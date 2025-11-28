from __future__ import annotations

from dataclasses import dataclass, field

from hacnet.full.modules.encoder import FullEncoderConfig
from hacnet.full.modules.mainnet import FullMainNetConfig
from hacnet.full.modules.decoder import FullDecoderConfig
from hacnet.minimal.modules.router import RouterConfig
from hacnet.minimal.modules.chunker import ChunkerConfig


@dataclass
class HACNetMinimalConfig:
    hidden_dim: int = 128
    encoder: FullEncoderConfig = field(default_factory=FullEncoderConfig)
    router: RouterConfig | None = None
    chunker: ChunkerConfig | None = None
    mainnet: FullMainNetConfig | None = None
    decoder: FullDecoderConfig | None = None

    def __post_init__(self):
        if self.router is None:
            self.router = RouterConfig(hidden_dim=self.hidden_dim)
        if self.chunker is None:
            self.chunker = ChunkerConfig(hidden_dim=self.hidden_dim)
        if self.mainnet is None:
            self.mainnet = FullMainNetConfig(hidden_dim=self.hidden_dim)
        if self.decoder is None:
            self.decoder = FullDecoderConfig(hidden_dim=self.hidden_dim)
