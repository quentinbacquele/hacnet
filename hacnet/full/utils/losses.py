"""
Loss utilities for minimal HAC-Net training.

Includes:
    - waveform reconstruction (L1/MSE);
    - multi-resolution spectral loss;
    - refractory penalty (L_refr).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class SpectralConfig:
    fft_sizes: Sequence[int] = (256, 512, 1024)
    hop_sizes: Sequence[int] = (64, 128, 256)
    win_lengths: Sequence[int] = (256, 512, 1024)
    power: float = 1.0


def waveform_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.l1_loss(pred, target, reduction=reduction) + F.mse_loss(pred, target, reduction=reduction)


def spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: SpectralConfig,
) -> torch.Tensor:
    total = 0.0
    for fft, hop, win in zip(config.fft_sizes, config.hop_sizes, config.win_lengths):
        window = torch.hann_window(win, device=pred.device, dtype=pred.dtype)
        pred_spec = torch.stft(
            pred.view(-1, pred.shape[-1]),
            n_fft=fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
        )
        tgt_spec = torch.stft(
            target.view(-1, target.shape[-1]),
            n_fft=fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
        )
        total += F.l1_loss(torch.abs(pred_spec) ** config.power, torch.abs(tgt_spec) ** config.power)
    return total / len(config.fft_sizes)


def refractory_loss(change_scores: torch.Tensor, refractory: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Soft penalty encouraging a minimum duration between cuts.
    """
    if refractory <= 1:
        return torch.tensor(0.0, device=change_scores.device)
    penalty = 0.0
    norm = 0.0
    for k in range(1, refractory):
        cs1 = change_scores[:, k:]
        cs2 = change_scores[:, :-k]
        if mask is not None:
            valid = mask[:, k:].float() * mask[:, :-k].float()
            penalty += (cs1 * cs2 * valid).sum()
            norm += valid.sum()
        else:
            penalty += (cs1 * cs2).sum()
            norm += cs1.numel()
    norm = max(norm, 1.0)
    return penalty / norm
