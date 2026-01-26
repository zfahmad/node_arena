from dataclasses import dataclass
from typing import Mapping

from python.players.random_player import RandomPlayer


@dataclass
class RandomPlayerConfig:
    seed: int | None = None


class RandomPlayerFactory:
    def coerce_random_config(self, raw_config: Mapping) -> RandomPlayerConfig:
        return RandomPlayerConfig(
            seed=raw_config.get("seed"),
        )

    def make_random_player(self, cfg: RandomPlayerConfig) -> RandomPlayer:
        return RandomPlayer(
            seed=cfg.seed,
        )

    def __call__(self, raw_config: Mapping) -> RandomPlayer:
        cfg = self.coerce_random_config(raw_config)
        player = self.make_random_player(cfg)
        return player
