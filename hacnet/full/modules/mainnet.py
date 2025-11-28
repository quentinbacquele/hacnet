"""H-Net inspired main network for the full HAC-Net variant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .hnet_blocks import BlockConfig, MambaBlock, CausalTransformerBlock, build_causal_mask


@dataclass
class FullMainNetConfig:
    hidden_dim: int
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    ffn_mult: float = 4.0
    expansion: int = 2
    kernel_size: int = 5
    ssm_ratio: float = 0.5
    max_seq_len: int = 8192
    mamba_d_state: int = 256
    mamba_d_conv: int = 4
    mamba_chunk_size: int = 256


class HNetMainNet(nn.Module):
    def __init__(self, config: FullMainNetConfig):
        super().__init__()
        self.config = config
        block_cfg = BlockConfig(
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            kernel_size=config.kernel_size,
            expansion=config.expansion,
            num_heads=config.num_heads,
            ffn_mult=config.ffn_mult,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            chunk_size=config.mamba_chunk_size,
        )
        total_layers = max(1, config.num_layers)
        ssm_layers = max(1, int(total_layers * config.ssm_ratio))
        ssm_layers = min(ssm_layers, total_layers)
        attn_layers = total_layers - ssm_layers
        pattern = []
        ssm_remaining = ssm_layers
        attn_remaining = attn_layers
        while len(pattern) < total_layers:
            if ssm_remaining > 0:
                pattern.append("ssm")
                ssm_remaining -= 1
            if attn_remaining > 0 and len(pattern) < total_layers:
                pattern.append("attn")
                attn_remaining -= 1
        layers = []
        for kind in pattern:
            if kind == "ssm":
                layers.append(MambaBlock(block_cfg))
            else:
                layers.append(CausalTransformerBlock(block_cfg))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.register_buffer(
            "causal_mask",
            build_causal_mask(config.max_seq_len, torch.device("cpu")),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("MainNet expects inputs shaped (B, L, D)")
        B, L, _ = x.shape
        if L > self.causal_mask.shape[0]:
            raise ValueError(
                f"MainNet sequence length {L} exceeds max_seq_len={self.causal_mask.shape[0]}"
            )
        attn_mask = self.causal_mask[:L, :L].to(x.device)
        for layer in self.layers:
            if isinstance(layer, CausalTransformerBlock):
                x = layer(x, attn_mask=attn_mask, key_padding_mask=mask)
            else:
                x = layer(x)
        x = self.final_norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
