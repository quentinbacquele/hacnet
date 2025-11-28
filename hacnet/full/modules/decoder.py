"""Decoder for the full HAC-Net model built from H-Net style blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .hnet_blocks import BlockConfig, MambaBlock, CausalTransformerBlock, build_causal_mask


@dataclass
class FullDecoderConfig:
    hidden_dim: int
    output_channels: int = 1
    ssm_layers: int = 2
    attn_layers: int = 1
    kernel_size: int = 5
    dropout: float = 0.1
    expansion: int = 2
    num_heads: int = 4
    ffn_mult: float = 2.0
    upsample_layers: int = 4
    upsample_stride: int = 2
    conv_kernel: int = 9
    max_seq_len: int = 8192
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_chunk_size: int = 256


class HNetDecoder(nn.Module):
    def __init__(self, config: FullDecoderConfig):
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
        prenet = []
        for kind in pattern:
            if kind == "ssm":
                prenet.append(MambaBlock(block_cfg))
            else:
                prenet.append(CausalTransformerBlock(block_cfg))
        self.prenet = nn.ModuleList(prenet)
        self.prenet_norm = nn.LayerNorm(config.hidden_dim)
        self.register_buffer(
            "causal_mask",
            build_causal_mask(config.max_seq_len, torch.device("cpu")),
            persistent=False,
        )

        up_layers = []
        in_ch = config.hidden_dim
        for idx in range(max(1, config.upsample_layers - 1)):
            up_layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    in_ch,
                    kernel_size=config.conv_kernel,
                    stride=config.upsample_stride,
                    padding=config.conv_kernel // 2,
                    output_padding=config.upsample_stride - 1,
                )
            )
            up_layers.append(nn.GELU())
        up_layers.append(
            nn.ConvTranspose1d(
                in_ch,
                config.output_channels,
                kernel_size=config.conv_kernel,
                stride=config.upsample_stride,
                padding=config.conv_kernel // 2,
                output_padding=config.upsample_stride - 1,
            )
        )
        self.upsampler = nn.Sequential(*up_layers)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if frame_embeddings.dim() != 3:
            raise ValueError("Decoder expects (B, L, D) inputs")
        x = frame_embeddings
        if skip is not None:
            x = x + skip[..., : x.shape[-1]]
        L = x.shape[1]
        if L > self.causal_mask.shape[0]:
            raise ValueError(
                f"Decoder sequence length {L} exceeds max_seq_len={self.causal_mask.shape[0]}"
            )
        attn_mask = self.causal_mask[:L, :L].to(x.device)
        for layer in self.prenet:
            if isinstance(layer, CausalTransformerBlock):
                x = layer(x, attn_mask=attn_mask)
            else:
                x = layer(x)
        x = self.prenet_norm(x)
        x = x.transpose(1, 2)
        return self.upsampler(x)
