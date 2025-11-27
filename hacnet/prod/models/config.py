from __future__ import annotations

from dataclasses import dataclass, field

from hacnet.minimal.modules.encoder import MelConfig
from hacnet.minimal.modules.router import RouterConfig
from hacnet.minimal.modules.chunker import ChunkerConfig
from hacnet.minimal.modules.mainnet import MainNetConfig
from hacnet.minimal.modules.decoder import DecoderConfig


@dataclass
class HACNetMinimalConfig:
    hidden_dim: int = 128
    encoder: MelConfig = field(default_factory=MelConfig)
    router: RouterConfig | None = None
    chunker: ChunkerConfig | None = None
    mainnet: MainNetConfig | None = None
    decoder: DecoderConfig | None = None

    def __post_init__(self):
        if self.router is None:
            self.router = RouterConfig(hidden_dim=self.hidden_dim)
        if self.chunker is None:
            self.chunker = ChunkerConfig(hidden_dim=self.hidden_dim)
        if self.mainnet is None:
            self.mainnet = MainNetConfig(hidden_dim=self.hidden_dim)
        if self.decoder is None:
            self.decoder = DecoderConfig(hidden_dim=self.hidden_dim)
