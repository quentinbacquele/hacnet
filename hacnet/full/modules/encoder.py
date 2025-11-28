"""H-Net style encoder used by the full HAC-Net variant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn

from hacnet.minimal.modules.encoder import MelConfig, MelSpectrogramFrontEnd
from .hnet_blocks import BlockConfig, MambaBlock, CausalTransformerBlock, build_causal_mask


@dataclass
class FullEncoderConfig:
    mel: MelConfig = field(default_factory=MelConfig)
    ssm_layers: int = 4
    attn_layers: int = 2
    kernel_size: int = 5
    dropout: float = 0.1
    expansion: int = 2
    num_heads: int = 4
    ffn_mult: float = 2.0
    max_seq_len: int = 4096
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_chunk_size: int = 256

    @property
    def hop_length(self) -> int:
        return self.mel.hop_length

    @property
    def sample_rate(self) -> int:
        return self.mel.sample_rate


class HNetEncoder(nn.Module):
    """Mel frontend + H-Net inspired stack mixing SSM and attention blocks."""

    def __init__(self, config: FullEncoderConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        self.frontend = MelSpectrogramFrontEnd(config.mel)
        self.input_proj = nn.Linear(config.mel.n_mels, hidden_dim)
        block_cfg = BlockConfig(
            hidden_dim=hidden_dim,
            dropout=config.dropout,
            kernel_size=config.kernel_size,
            expansion=config.expansion,
            num_heads=config.num_heads,
            ffn_mult=config.ffn_mult,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            chunk_size=config.mamba_chunk_size,
        )
        pattern = []
        ssm_remaining = config.ssm_layers
        attn_remaining = config.attn_layers
        while ssm_remaining > 0 or attn_remaining > 0:
            if ssm_remaining > 0:
                pattern.append("ssm")
                ssm_remaining -= 1
            if attn_remaining > 0:
                pattern.append("attn")
                attn_remaining -= 1
        layers = []
        for kind in pattern:
            if kind == "ssm":
                layers.append(MambaBlock(block_cfg))
            else:
                layers.append(CausalTransformerBlock(block_cfg))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer(
            "causal_mask",
            build_causal_mask(config.max_seq_len, torch.device("cpu")),
            persistent=False,
        )

    def forward(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if waveforms.dim() != 2:
            raise ValueError("Expected waveforms shaped (B, T)")
        mel, hop = self.frontend(waveforms)
        x = self.input_proj(mel.transpose(1, 2))
        B, L, _ = x.shape
        if L > self.causal_mask.shape[0]:
            raise ValueError(
                f"Encoder sequence length {L} exceeds max_seq_len={self.causal_mask.shape[0]}"
            )
        attn_mask = self.causal_mask[:L, :L].to(x.device)
        for layer in self.layers:
            if isinstance(layer, CausalTransformerBlock):
                x = layer(x, attn_mask=attn_mask)
            else:
                x = layer(x)
        x = self.final_norm(x)
        return x, hop
