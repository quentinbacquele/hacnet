"""
Archived causal convolutional main net (for reference).

Kept here so we can revisit the lightweight TCN implementation if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MainNetConvConfig:
    hidden_dim: int
    num_layers: int = 12
    kernel_size: int = 3
    dilation_growth: int = 2
    dilation_cycle: int = 6
    dropout: float = 0.1
    ffn_mult: float = 2.0


class GatedDilatedConvLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        ffn_mult: float,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pre_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=d_model,
            bias=False,
        )
        self.pointwise = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(d_model, d_model)
        self.skip_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        ffn_hidden = max(1, int(d_model * ffn_mult))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        residual = x
        x = self.pre_norm(x)
        x = x.transpose(1, 2)
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        filt, gate = x.chunk(2, dim=-1)
        x = torch.tanh(filt) * torch.sigmoid(gate)
        x = self.dropout(x)

        skip = self.skip_proj(x)
        x = residual + self.residual_proj(x)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            x = x * mask_exp
            skip = skip * mask_exp
        return x, skip


class CausalConvMainNet(nn.Module):
    def __init__(self, config: MainNetConvConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        dilation = 1
        for idx in range(config.num_layers):
            self.layers.append(
                GatedDilatedConvLayer(
                    d_model=config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    ffn_mult=config.ffn_mult,
                )
            )
            dilation *= config.dilation_growth
            if config.dilation_cycle > 0 and (idx + 1) % config.dilation_cycle == 0:
                dilation = 1

        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.skip_norm = nn.LayerNorm(config.hidden_dim)
        self.skip_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.final_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip_total = None
        for layer in self.layers:
            x, skip = layer(x, mask=mask)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            skip_total = torch.zeros_like(x)

        skip_total = skip_total / len(self.layers)
        skip_total = self.skip_proj(self.skip_norm(skip_total))
        x = self.final_norm(x)
        x = x + self.final_dropout(skip_total)

        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
