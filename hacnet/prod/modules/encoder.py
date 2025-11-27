"""
Minimal HAC-Net encoder components.

This file houses two building blocks:
    1. MelSpectrogramFrontEnd: converts raw waveforms into causal log-mel frames.
    2. CausalConvEncoder: a lightweight causal stack that produces frame-wise embeddings.

The design mirrors the HAC-Net paper section 2.1.3, where a causal encoder provides
short-range spectral summaries that the router and subsequent modules can consume.
Both modules are intentionally lightweight so they can run on a laptop GPU/CPU while
sharing the same interfaces we will later scale up on larger accelerators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float,
    f_max: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Construct a triangular mel filterbank compatible with torch.stft outputs.
    """
    if f_max is None:
        f_max = sample_rate / 2

    mel_points = torch.linspace(
        _hz_to_mel(torch.tensor(f_min)),
        _hz_to_mel(torch.tensor(f_max)),
        n_mels + 2,
        dtype=dtype,
        device=device,
    )
    hz_points = _mel_to_hz(mel_points)

    fft_freqs = torch.linspace(
        0.0,
        sample_rate / 2,
        n_fft // 2 + 1,
        dtype=dtype,
        device=device,
    )

    filterbank = torch.zeros(n_mels, n_fft // 2 + 1, device=device, dtype=dtype)
    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]

        left_slope = (fft_freqs - left) / (center - left + 1e-10)
        right_slope = (right - fft_freqs) / (right - center + 1e-10)

        filterbank[m] = torch.clamp(torch.min(left_slope, right_slope), min=0.0)

    # Normalize each filter to have unit area so energy is preserved on average.
    filterbank = filterbank / (filterbank.sum(dim=-1, keepdim=True) + 1e-10)
    return filterbank


@dataclass
class MelConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 64
    f_min: float = 150.0
    f_max: Optional[float] = None
    log_offset: float = 1e-4


class MelSpectrogramFrontEnd(nn.Module):
    """
    Causal log-mel feature extractor built on torch.stft.
    """

    def __init__(self, config: MelConfig):
        super().__init__()
        self.config = config
        window = torch.hann_window(
            config.win_length, periodic=False, dtype=torch.float32
        )
        self.register_buffer("window", window, persistent=False)
        filterbank = build_mel_filterbank(
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            f_min=config.f_min,
            f_max=config.f_max,
            device=window.device,
            dtype=window.dtype,
        )
        self.register_buffer("mel_filterbank", filterbank, persistent=False)

    def forward(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            waveforms: Tensor shaped (B, T) in float32/float16.

        Returns:
            features: (B, n_mels, n_frames) log-mel energies.
            frame_hop: seconds per hop, useful for mapping indices to wall-clock time.
        """
        if waveforms.dim() != 2:
            raise ValueError("Expected waveforms of shape (B, T)")

        spec = torch.stft(
            waveforms,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=False,  # ensures strict causality
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spec.real**2 + spec.imag**2
        mel = torch.einsum("mf,bft->bmt", self.mel_filterbank, power)
        log_mel = torch.log(self.config.log_offset + mel)
        frame_hop = self.config.hop_length / self.config.sample_rate
        return log_mel, frame_hop


class CausalConvBlock(nn.Module):
    """
    Depthwise-separable causal convolution with GLU gating.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size - 1
        x = F.pad(x, (pad, 0))
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)
        x, gate = x.chunk(2, dim=1)
        return x * torch.sigmoid(gate)


class CausalConvEncoder(nn.Module):
    """
    Lightweight frame encoder that runs a causal conv stack over log-mel inputs.
    """

    def __init__(
        self,
        mel_config: MelConfig,
        hidden_dim: int,
        num_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frontend = MelSpectrogramFrontEnd(mel_config)
        self.proj_in = nn.Linear(mel_config.n_mels, hidden_dim)
        self.blocks = nn.ModuleList(
            CausalConvBlock(hidden_dim, kernel_size) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            waveforms: (B, T) waveform tensor in float.

        Returns:
            frame_embeddings: (B, L, hidden_dim) causal features for routing.
            frame_hop_sec: hop duration in seconds for downstream timestamping.
        """
        mel, frame_hop = self.frontend(waveforms)
        x = self.proj_in(mel.transpose(1, 2))  # (B, L, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, L) for conv processing
        for block in self.blocks:
            x = x + self.dropout(block(x))
        x = x.transpose(1, 2)
        return x, frame_hop
