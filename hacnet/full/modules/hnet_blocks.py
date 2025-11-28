"""Shared H-Net style blocks for the full HAC-Net variant.

These blocks intentionally mirror the building blocks described in the
H-Net paper: a lightweight state-space inspired module (implemented here as
gated depthwise separable convolutions) and a causal Transformer block that
operates on the boundary-indexed unit stream. They are designed so the
encoder, main network, and decoder can mix and match the same primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from mamba_ssm.modules.mamba2 import Mamba2

    _HAS_MAMBA = True
except ImportError:  # pragma: no cover - optional dependency
    Mamba2 = None
    _HAS_MAMBA = False
    warnings.warn(
        "mamba-ssm not installed; falling back to a lightweight causal conv block. "
        "Install mamba-ssm on CUDA hardware to match the official H-Net blocks.",
        RuntimeWarning,
    )


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


@dataclass
class BlockConfig:
    hidden_dim: int
    dropout: float = 0.1
    kernel_size: int = 5
    expansion: int = 2
    num_heads: int = 4
    ffn_mult: float = 4.0
    d_state: int = 128
    d_conv: int = 4
    chunk_size: int = 256
    norm_eps: float = 1e-5


def build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


class MambaBlock(nn.Module):
    """Wraps the official Mamba2 block with a fallback depthwise SSM."""

    def __init__(self, config: BlockConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.norm = SimpleRMSNorm(config.hidden_dim, eps=config.norm_eps)
        if _HAS_MAMBA:
            self.mixer = Mamba2(
                d_model=config.hidden_dim,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expansion,
                chunk_size=config.chunk_size,
            )
        else:
            # Lightweight causal convolution fallback to keep local dev working.
            inner_dim = config.hidden_dim * config.expansion
            self.in_proj = nn.Linear(config.hidden_dim, inner_dim * 2)
            self.depthwise = nn.Conv1d(
                inner_dim,
                inner_dim,
                kernel_size=config.kernel_size,
                groups=inner_dim,
            )
            self.out_proj = nn.Linear(inner_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        if _HAS_MAMBA:
            y = self.mixer(y)
        else:
            y1, y2 = self.in_proj(y).chunk(2, dim=-1)
            y = torch.tanh(y1) * torch.sigmoid(y2)
            y = y.transpose(1, 2)
            y = F.pad(y, (self.config.kernel_size - 1, 0))
            y = self.depthwise(y)
            y = y.transpose(1, 2)
            y = self.out_proj(y)
        return residual + self.dropout(y)


class CausalTransformerBlock(nn.Module):
    """Causal Transformer + FFN block reused across encoder/main/decoder."""

    def __init__(self, config: BlockConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.norm1 = SimpleRMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm2 = SimpleRMSNorm(config.hidden_dim, eps=config.norm_eps)
        ffn_dim = int(config.hidden_dim * config.ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.hidden_dim),
        )
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        attn_out, _ = self.attn(
            y,
            y,
            y,
            attn_mask=attn_mask,
            key_padding_mask=None if key_padding_mask is None else ~key_padding_mask,
        )
        x = residual + self.dropout1(attn_out)
        residual = x
        y = self.norm2(x)
        y = self.ffn(y)
        return residual + self.dropout2(y)
