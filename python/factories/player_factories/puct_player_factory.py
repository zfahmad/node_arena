from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from python.players.puct_inference_server import InferenceClient
from python.players.puct_player import PUCTPlayer, PUCB, SoftArgmax


@dataclass
class PUCBConfig:
    seed: int | None = None
    C: float = 1.0


@dataclass
class SoftArgmaxConfig:
    seed: int | None = None


@dataclass
class PUCTConfig:
    inf_client: InferenceClient | None = None
    seed: int | None = None
    num_samples: int = 256
    gamma: float = 1.0
    exploitation_threshold: int = 0
    training: bool = False
    dirichlet_epsilon: float = 0.1
    dirichlet_alpha: float = 0.3
    tree_policy: PUCBConfig = field(default_factory=lambda: PUCBConfig())
    final_policy: SoftArgmaxConfig = field(default_factory=lambda: SoftArgmaxConfig())


class PUCTPlayerFactory:
    def coerce_puct_config(self, raw_config: Mapping):
        # Base seed for PUCT
        base_seed = raw_config.get("seed")

        # Coerce raw subconfigs
        tree_policy_raw = raw_config["tree_policy"]
        final_policy_raw = raw_config["final_policy"]

        # If PUCT seed is provided, use SeedSequence to generate child seeds
        if base_seed is not None:
            ss = np.random.SeedSequence(base_seed)
            # Request three child seeds: tree, final, eval
            child_seeds = ss.spawn(3)

            def make_unique_int_seed(child_ss: np.random.SeedSequence) -> int:
                # generate a single uint32 state from this child
                return int(child_ss.generate_state(1)[0])

            # Assign seeds only if not already provided
            if tree_policy_raw.get("seed") is None:
                tree_policy_raw["seed"] = make_unique_int_seed(child_seeds[0])
            if final_policy_raw.get("seed") is None:
                final_policy_raw["seed"] = make_unique_int_seed(child_seeds[1])

        pucb_cfg = PUCBConfig(
            seed=tree_policy_raw.get("seed"), C=tree_policy_raw.get("C", 1.0)
        )
        sa_cfg = SoftArgmaxConfig(seed=tree_policy_raw.get("seed"))

        return PUCTConfig(
            inf_client=None,
            seed=raw_config.get("seed"),
            num_samples=raw_config.get("num_samples", 256),
            gamma=raw_config.get("gamma", 0.1),
            exploitation_threshold=raw_config.get("exploitation_threshold", 1),
            training=raw_config.get("training", False),
            dirichlet_epsilon=raw_config.get("dirichlet_epsilon", 0.1),
            dirichlet_alpha=raw_config.get("dirichlet_alpha", 0.3),
            tree_policy=pucb_cfg,
            final_policy=sa_cfg
        )

    def make_puct_player(self, cfg: PUCTConfig) -> PUCTPlayer:
        tree_policy = PUCB(cfg.tree_policy.seed, cfg.tree_policy.C)
        final_policy = SoftArgmax(cfg.tree_policy.seed)
        return PUCTPlayer(
            inf_client=None,
            seed=cfg.seed,
            num_samples=cfg.num_samples,
            gamma=cfg.gamma,
            exploitation_threshold=cfg.exploitation_threshold,
            training=cfg.training,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
            dirichlet_alpha=cfg.dirichlet_alpha,
            tree_policy=tree_policy,
            final_policy=final_policy,
        )

    def __call__(self, raw_config: Mapping) -> PUCTPlayer:
        cfg = self.coerce_puct_config(raw_config)
        player = self.make_puct_player(cfg)
        return player
