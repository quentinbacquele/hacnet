"""
EMA smoothing and frame-rate restoration for HAC-Net.

Implements:
    hat{z}_j = P_j z_j + (1 - P_j) hat{z}_{j-1}
and holds hat{z}_j constant between consecutive boundaries to recreate the
frame-level stream expected by the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMADechunker(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        unit_embeddings: torch.Tensor,
        unit_confidence: torch.Tensor,
        unit_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            unit_embeddings: (B, J, D) output of the main network (J varies across batch).
            unit_confidence: (B, J) confidence P_j derived from router scores.
            boundary_mask: (B, L) bool mask with boundaries in the original frame sequence.
        """
        smoothed = self._ema(unit_embeddings, unit_confidence, unit_mask)
        frames = self._upsample(smoothed, boundary_mask)
        return frames

    def _ema(
        self,
        unit_embeddings: torch.Tensor,
        unit_confidence: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, J, D = unit_embeddings.shape
        smoothed = torch.zeros_like(unit_embeddings)
        prev = torch.zeros(B, D, device=unit_embeddings.device, dtype=unit_embeddings.dtype)
        for j in range(J):
            valid = unit_mask[:, j].unsqueeze(-1)
            p = torch.clamp(unit_confidence[:, j].unsqueeze(-1), self.eps, 1 - self.eps)
            current = torch.where(valid, unit_embeddings[:, j], torch.zeros_like(unit_embeddings[:, j]))
            new_val = torch.where(valid, p * current + (1 - p) * prev, prev)
            smoothed[:, j] = new_val
            prev = new_val
        return smoothed

    def _upsample(self, smoothed_units: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        B, L = boundary_mask.shape
        J = smoothed_units.shape[1]
        unit_indices = torch.cumsum(boundary_mask, dim=1) - 1
        unit_indices = unit_indices.clamp(min=0, max=J - 1)
        frames = torch.gather(
            smoothed_units,
            dim=1,
            index=unit_indices.unsqueeze(-1).expand(-1, -1, smoothed_units.shape[-1]),
        )
        return frames
