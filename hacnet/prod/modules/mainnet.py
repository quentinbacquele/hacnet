"""
Transformer-based main network for HAC-Net minimal stack.

Replaces the earlier causal convolutional block with a lightweight causal
Transformer so unit embeddings can capture longer context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MainNetConfig:
    hidden_dim: int
    num_layers: int = 8
    num_heads: int = 4
    ffn_mult: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 4096


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_mult: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(
            y,
            y,
            y,
            attn_mask=attn_mask,
            key_padding_mask=None if key_padding_mask is None else ~key_padding_mask,
        )
        x = x + self.dropout1(attn_out)
        y = self.norm2(x)
        x = x + self.dropout2(self.ffn(y))
        return x


class TransformerMainNet(nn.Module):
    def __init__(self, config: MainNetConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                ffn_mult=config.ffn_mult,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.positional_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.hidden_dim)
        )
        nn.init.normal_(self.positional_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            mask: optional (B, L) bool mask of valid units
        """
        B, L, _ = x.shape
        if L > self.config.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len={self.config.max_seq_len}")

        pos = self.positional_embed[:, :L, :].to(x.device)
        x = x + pos
        attn_mask = self.causal_mask[:L, :L].to(x.device)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=mask,
            )

        x = self.final_norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
