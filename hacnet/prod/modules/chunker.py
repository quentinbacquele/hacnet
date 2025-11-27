"""
Boundary readout and packing utilities for HAC-Net.

Compared to the original H-Net chunker (which simply gathered boundary tokens),
this module implements the "boundary-anchored micro-pool" described in the HAC-Net
proposal (Section 2.1.3, Supplement S4). Around each detected boundary we aggregate
a short causal window of encoder frames, optionally weighted by the router's change
scores, to form compact unit representations that feed the main network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ChunkerConfig:
    hidden_dim: int
    pool_size: int = 5  # number of frames to aggregate per boundary (causal)
    use_change_score: bool = True
    normalize: bool = True


@dataclass
class ChunkerOutput:
    chunks: torch.Tensor  # (B, J, D)
    mask: torch.Tensor  # (B, J) boolean mask for valid units
    boundary_indices: torch.Tensor  # (B, J) frame indices of emitted units
    confidence: torch.Tensor  # (B, J) boundary confidence per unit


class BoundaryChunker(nn.Module):
    """
    Implements boundary-anchored pooling and packing.
    """

    def __init__(self, config: ChunkerConfig):
        super().__init__()
        self.config = config
        self.pool_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        nn.init.eye_(self.pool_proj.weight)
        if self.pool_proj.bias is not None:
            nn.init.zeros_(self.pool_proj.bias)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        boundary_mask: torch.Tensor,
        change_score: Optional[torch.Tensor] = None,
        boundary_confidence: Optional[torch.Tensor] = None,
    ) -> ChunkerOutput:
        """
        Args:
            frame_embeddings: (B, L, D) encoder outputs.
            boundary_mask: (B, L) bool mask where True marks a boundary frame.
            change_score: (B, L) optional score to weight the pooled vectors.
        """
        B, L, D = frame_embeddings.shape
        device = frame_embeddings.device

        if change_score is None:
            change_score = torch.ones(B, L, device=device)
        if boundary_confidence is None:
            boundary_confidence = change_score

        window = self.config.pool_size
        chunks = []
        change_vals = []
        indices = []
        max_units = boundary_mask.sum(dim=1).max().item() if boundary_mask.any() else 0

        pooled = torch.zeros(B, max_units or 1, D, device=device)
        mask = torch.zeros(B, max_units or 1, dtype=torch.bool, device=device)
        idx_tensor = torch.zeros(B, max_units or 1, dtype=torch.long, device=device)
        score_tensor = torch.zeros(B, max_units or 1, device=device)

        for b in range(B):
            unit_idx = 0
            for t in torch.nonzero(boundary_mask[b], as_tuple=False).flatten():
                t = t.item()
                start = max(0, t - window + 1)
                segment = frame_embeddings[b, start : t + 1]
                weights = torch.ones_like(change_score[b, start : t + 1])
                if self.config.use_change_score:
                    weights = change_score[b, start : t + 1]
                weights = weights.unsqueeze(-1)
                pooled_vec = (segment * weights).sum(dim=0)
                if self.config.normalize:
                    pooled_vec = pooled_vec / (weights.sum() + 1e-6)
                pooled_vec = self.pool_proj(pooled_vec)
                pooled[b, unit_idx] = pooled_vec
                mask[b, unit_idx] = True
                idx_tensor[b, unit_idx] = t
                score_tensor[b, unit_idx] = boundary_confidence[b, t]
                unit_idx += 1

        return ChunkerOutput(
            chunks=pooled,
            mask=mask,
            boundary_indices=idx_tensor,
            confidence=score_tensor,
        )
