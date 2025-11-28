from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from hacnet.full.modules.encoder import HNetEncoder
from hacnet.full.modules.mainnet import HNetMainNet
from hacnet.full.modules.decoder import HNetDecoder
from hacnet.minimal.modules.router import MultiLagRouter, RouterOutput
from hacnet.minimal.modules.chunker import BoundaryChunker, ChunkerOutput
from hacnet.minimal.modules.dechunker import EMADechunker
from .config import HACNetMinimalConfig


class _StraightThroughOnes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _ste_ones(x: torch.Tensor) -> torch.Tensor:
    return _StraightThroughOnes.apply(x)


@dataclass
class HACNetMinimalOutput:
    reconstruction: torch.Tensor
    frame_embeddings: torch.Tensor
    unit_embeddings: torch.Tensor
    router: RouterOutput
    chunker: ChunkerOutput
    hop_seconds: float


class HACNetMinimal(nn.Module):
    def __init__(self, config: HACNetMinimalConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.encoder = HNetEncoder(config.encoder, hidden_dim=hidden_dim)
        self.router = MultiLagRouter(config.router)
        self.chunker = BoundaryChunker(config.chunker)
        self.mainnet = HNetMainNet(config.mainnet)
        self.dechunker = EMADechunker()
        self.decoder = HNetDecoder(config.decoder)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        nn.init.zeros_(self.residual_proj.weight)
        if self.residual_proj.bias is not None:
            nn.init.zeros_(self.residual_proj.bias)
        self.residual_proj.weight._no_reinit = True

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> HACNetMinimalOutput:
        """
        Args:
            waveforms: (B, T) float tensor of audio samples.
            lengths: optional (B,) tensor with valid sample counts.
        """
        frame_features, hop = self.encoder(waveforms)
        mask = self._frame_mask(frame_features.shape[0], frame_features.shape[1], lengths, waveforms.device)

        router_out = self.router(frame_features, mask=mask)
        chunk_out = self.chunker(
            frame_features,
            router_out.boundary_mask,
            router_out.change_score,
            router_out.selected_probs.squeeze(-1),
        )
        unit_embeddings = self.mainnet(chunk_out.chunks, mask=chunk_out.mask)
        frame_stream = self.dechunker(
            unit_embeddings,
            chunk_out.confidence,
            chunk_out.mask,
            router_out.boundary_mask,
        )
        residual = self.residual_proj(frame_features.to(self.residual_proj.weight.dtype))
        frame_stream = self._ste_residual_mix(frame_stream, residual, router_out.selected_probs)
        reconstruction = self.decoder(frame_stream, skip=frame_features)

        return HACNetMinimalOutput(
            reconstruction=reconstruction,
            frame_embeddings=frame_stream,
            unit_embeddings=unit_embeddings,
            router=router_out,
            chunker=chunk_out,
            hop_seconds=hop,
        )

    def _frame_mask(
        self,
        batch_size: int,
        num_frames: int,
        lengths: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)
        hop_samples = self.config.encoder.hop_length
        frame_lengths = torch.ceil(lengths.float() / hop_samples).long()
        frame_lengths = frame_lengths.clamp(max=num_frames)
        frame_idx = torch.arange(num_frames, device=device)
        return frame_idx.unsqueeze(0) < frame_lengths.unsqueeze(1)

    def _ste_residual_mix(
        self,
        frame_stream: torch.Tensor,
        residual: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        gate = _ste_ones(confidence).to(frame_stream.dtype).expand_as(frame_stream)
        mixed = frame_stream * gate + residual.to(frame_stream.dtype)
        return mixed
