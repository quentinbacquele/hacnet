"""
Multi-lag, multi-band router for the minimal HAC-Net stack.

The design follows Supplement S4 of the HAC-Net proposal:
    - compares each new frame against several causal lags and frequency bands;
    - aggregates change evidence with simplex weights learned from the encoder features;
    - applies either a fixed threshold or a rolling percentile rule to convert scores to cuts;
    - enforces a causal minimum-duration rule (refractory period).

The module is intentionally lightweight but keeps the same interface as H-Net's routing
blocks so it can plug into the existing chunking/dechunking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks


@dataclass
class RouterConfig:
    hidden_dim: int
    num_bands: int = 4
    band_dim: int = 32
    lags: Sequence[int] = (1, 2, 4, 8, 16, 32)
    threshold_mode: str = "fixed"  # {"fixed", "rolling", ...}
    fixed_threshold: float = 0.5
    rolling_percentile: float = 90.0
    rolling_window: int = 200  # frames
    smooth_window: int = 7
    pre_smooth_window: int = 5
    peak_height: float = 0.001
    peak_distance: int = 5
    peak_prominence: float = 0.0005
    peak_rel_height: float = 0.5
    refractory: int = 8
    eps: float = 1e-4
    energy_gate: bool = True
    energy_threshold: float = -6.0
    energy_gamma: float = 5.0


@dataclass
class RouterState:
    last_boundary: torch.Tensor  # (B,)


@dataclass
class RouterOutput:
    boundary_prob: torch.Tensor  # (B, L, 2)
    boundary_mask: torch.Tensor  # (B, L)
    selected_probs: torch.Tensor  # (B, L, 1)
    change_score: torch.Tensor  # (B, L)
    threshold: torch.Tensor  # (B, L)
    state: RouterState


class MultiLagRouter(nn.Module):
    """
    Router that scores change points using multiple frequency bands and time lags.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        self.num_lags = len(config.lags)
        self.q_proj = nn.Linear(config.hidden_dim, config.num_bands * config.band_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_bands * config.band_dim, bias=False)
        self.band_logits = nn.Linear(config.hidden_dim, config.num_bands, bias=True)
        self.lag_logits = nn.Linear(config.hidden_dim, self.num_lags, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        state: Optional[RouterState] = None,
    ) -> RouterOutput:
        """
        Args:
            hidden_states: (B, L, D) encoder features.
            mask: (B, L) boolean mask of valid frames.
            state: optional RouterState for streaming/continuation.
        """
        if mask is None:
            mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        change_score = self._multi_lag_change(hidden_states, mask)
        if self.config.energy_gate:
            energy = hidden_states.pow(2).mean(dim=-1)
            gate = torch.sigmoid(
                self.config.energy_gamma * (energy - self.config.energy_threshold)
            )
            change_score = change_score * gate

        mode = self.config.threshold_mode.lower()
        peak_mask: Optional[torch.Tensor] = None
        if mode == "derivative":
            change_score = self._derivative_change(change_score, mask)
        elif mode == "smooth_derivative":
            change_score = self._smooth_derivative(change_score, mask)
        elif mode == "smooth_derivative_peaks":
            change_score = self._smooth_derivative(change_score, mask)
            peak_mask = self._peak_mask(change_score, mask)

        thresholds = self._compute_thresholds(change_score, mask)
        boundary_mask = self._select_boundaries(
            change_score,
            thresholds,
            mask,
            state,
            peak_mask=peak_mask,
        )
        change_score = change_score.clone()
        thresholds = thresholds.clone()
        # Ensure the first frame is always a boundary to avoid empty chunks.
        boundary_mask[:, 0] = True
        change_score[:, 0] = 1.0
        thresholds[:, 0] = 0.0

        selected_probs = torch.where(
            boundary_mask, change_score, 1.0 - change_score
        ).unsqueeze(-1)
        boundary_prob = torch.stack((1.0 - change_score, change_score), dim=-1)

        new_state = RouterState(
            last_boundary=self._last_boundary_indices(boundary_mask, state),
        )

        return RouterOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
            change_score=change_score,
            threshold=thresholds,
            state=new_state,
        )

    def _multi_lag_change(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        num_bands = self.config.num_bands
        band_dim = self.config.band_dim

        q = self.q_proj(hidden_states).view(B, L, num_bands, band_dim)
        k = self.k_proj(hidden_states).view(B, L, num_bands, band_dim)

        lag_changes = []
        lag_valid = []
        for lag in self.config.lags:
            shifted = torch.zeros_like(k)
            if lag == 0:
                shifted.copy_(k)
            else:
                shifted[:, lag:, :, :] = k[:, :-lag, :, :]
            numerator = (q * shifted).sum(dim=-1)
            denom = (
                q.norm(dim=-1) * shifted.norm(dim=-1) + self.config.eps
            )
            cos = numerator / denom
            change = 0.5 * (1.0 - cos)
            valid = torch.zeros(B, L, device=k.device, dtype=torch.bool)
            if lag == 0:
                valid[:, :] = True
            else:
                valid[:, lag:] = True
            lag_changes.append(change)
            lag_valid.append(valid & mask)

        change_tensor = torch.stack(lag_changes, dim=-2)  # (B, L, num_lags, num_bands)
        valid_tensor = torch.stack(lag_valid, dim=-1)  # (B, L, num_lags)

        band_w = torch.softmax(self.band_logits(hidden_states), dim=-1).unsqueeze(-2)
        lag_logits = self.lag_logits(hidden_states)
        lag_logits = torch.where(
            valid_tensor,
            lag_logits,
            torch.full_like(lag_logits, -1e4),
        )
        lag_w = torch.softmax(lag_logits, dim=-1).unsqueeze(-1)

        weighted = change_tensor * band_w
        aggregated = (weighted * lag_w).sum(dim=(-1, -2))
        aggregated = torch.clamp(aggregated, min=self.config.eps, max=1.0 - self.config.eps)
        aggregated = torch.where(mask, aggregated, torch.zeros_like(aggregated))
        return aggregated

    def _compute_thresholds(
        self,
        change_score: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.config.threshold_mode.lower()
        if mode == "fixed":
            return torch.full_like(change_score, self.config.fixed_threshold)
        if mode in {"rolling", "derivative", "smooth_derivative"}:
            return self._rolling_threshold(change_score, mask)
        if mode == "smooth_derivative_peaks":
            return torch.zeros_like(change_score)
        raise ValueError(f"Unknown threshold_mode '{self.config.threshold_mode}'.")

    def _rolling_threshold(
        self,
        change_score: torch.Tensor,
        mask: torch.Tensor,
        window: Optional[int] = None,
        percentile: Optional[float] = None,
    ) -> torch.Tensor:
        B, L = change_score.shape
        thresholds = torch.zeros_like(change_score)
        window = max(1, int(window if window is not None else self.config.rolling_window))
        half = max(1, window) // 2
        percentile = float(percentile if percentile is not None else self.config.rolling_percentile) / 100.0

        for b in range(B):
            valid_len = int(mask[b].sum().item())
            if valid_len == 0:
                continue
            curve = change_score[b, :valid_len].detach()
            for t in range(valid_len):
                start = max(0, t - half)
                end = min(valid_len, t + half + 1)
                window_vals = curve[start:end]
                if window_vals.numel() == 0:
                    thresholds[b, t] = self.config.fixed_threshold
                else:
                    thresholds[b, t] = torch.quantile(window_vals, percentile)
            if valid_len < L:
                thresholds[b, valid_len:] = self.config.fixed_threshold
        return thresholds

    def _smooth_change(
        self,
        change_score: torch.Tensor,
        mask: torch.Tensor,
        window: Optional[int] = None,
    ) -> torch.Tensor:
        window = max(1, int(window if window is not None else self.config.smooth_window))
        if window == 1:
            return change_score
        pad_left = window // 2
        pad_right = window - pad_left - 1
        padded = F.pad(change_score.unsqueeze(1), (pad_left, pad_right), mode="replicate")
        kernel = torch.ones(1, 1, window, device=change_score.device, dtype=change_score.dtype) / window
        smoothed = F.conv1d(padded, kernel).squeeze(1)
        smoothed = torch.where(mask, smoothed, torch.zeros_like(smoothed))
        smoothed = torch.clamp(smoothed, min=self.config.eps, max=1.0 - self.config.eps)
        return smoothed

    def _smooth_derivative(
        self,
        change_score: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        smoothed = self._smooth_change(change_score, mask, window=self.config.pre_smooth_window)
        return self._derivative_change(smoothed, mask)

    def _derivative_change(
        self,
        change_score: torch.Tensor,
        mask: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        diff = torch.zeros_like(change_score)
        diff[:, 1:] = torch.abs(change_score[:, 1:] - change_score[:, :-1])
        diff = torch.where(mask, diff, torch.zeros_like(diff))
        if self.config.smooth_window > 1:
            diff = self._smooth_change(diff, mask, window=self.config.smooth_window)
        if clamp:
            diff = torch.clamp(diff, min=self.config.eps, max=1.0 - self.config.eps)
        return diff

    def _peak_mask(
        self,
        change_score: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = change_score.shape
        peaks = torch.zeros_like(change_score, dtype=torch.bool)
        height = self.config.peak_height
        prominence = self.config.peak_prominence
        distance = max(1, int(self.config.peak_distance))
        for b in range(B):
            valid = mask[b].detach().cpu().numpy()
            curve = change_score[b].detach().cpu().numpy()
            indices = np.where(valid)[0]
            if indices.size == 0:
                continue
            segment = curve[indices]
            candidate_peaks, props = find_peaks(
                segment,
                height=height,
                prominence=prominence,
                distance=distance,
            )
            if candidate_peaks.size > 0:
                heights = props["peak_heights"]
                max_height = heights.max() if heights.size > 0 else 0.0
                rel_mask = heights >= max_height * self.config.peak_rel_height
                selected = candidate_peaks[rel_mask]
                if selected.size > 0:
                    peaks[b, indices[selected]] = True
        peaks[:, 0] = False
        return peaks & mask

    def _select_boundaries(
        self,
        change_score: torch.Tensor,
        thresholds: torch.Tensor,
        mask: torch.Tensor,
        state: Optional[RouterState],
        peak_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = change_score.shape
        refractory = self.config.refractory
        if state is None:
            last_boundary = change_score.new_full((B,), -refractory, dtype=torch.long)
        else:
            last_boundary = state.last_boundary.clone()

        boundary_mask = torch.zeros_like(change_score, dtype=torch.bool)
        for t in range(L):
            allow = (t - last_boundary) >= refractory
            if peak_mask is None:
                comparison = change_score[:, t] >= thresholds[:, t]
            else:
                comparison = peak_mask[:, t]
            proposed = comparison & allow & mask[:, t]
            boundary_mask[:, t] = proposed
            last_boundary = torch.where(proposed, torch.full_like(last_boundary, t), last_boundary)
        return boundary_mask

    def _last_boundary_indices(
        self,
        boundary_mask: torch.Tensor,
        state: Optional[RouterState],
    ) -> torch.Tensor:
        B, L = boundary_mask.shape
        if state is None:
            last_boundary = boundary_mask.new_full((B,), -self.config.refractory, dtype=torch.long)
        else:
            last_boundary = state.last_boundary.clone()
        for t in range(L):
            last_boundary = torch.where(
                boundary_mask[:, t],
                torch.full_like(last_boundary, t),
                last_boundary,
            )
        return last_boundary
