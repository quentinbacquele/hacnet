"""
Causal decoder for the minimal HAC-Net stack.

Maps dechunked frame embeddings back to waveform frames using a lightweight
transposed-convolution generator, optionally conditioned on encoder skip
connections. Designed to be simple enough for laptop training while remaining
faithful to HAC-Net's reconstruction objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    hidden_dim: int
    output_channels: int = 1
    kernel_size: int = 9
    num_layers: int = 4
    stride: int = 2
    upsample_initial: int = 2


class CausalDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        layers = []
        in_channels = config.hidden_dim
        stride = config.stride
        padding = config.kernel_size // 2
        for _ in range(config.num_layers - 1):
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=stride - 1,
                )
            )
            layers.append(nn.GELU())
        layers.append(
            nn.ConvTranspose1d(
                in_channels,
                config.output_channels,
                kernel_size=config.kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (B, L, D) dechunked features.
            skip: optional (B, L, D_skip) encoder skip features to concatenate.
        Returns:
            waveform reconstruction (B, C, T).
        """
        x = frame_embeddings
        if skip is not None:
            x = x + skip[..., : x.shape[-1]]
        x = x.transpose(1, 2)
        audio = self.net(x)
        return audio
