from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from python.players.mcts_helpers import *
from python.players.mcts_player import MCTSPlayer


class PolicyType(Enum):
    UCB1 = "ucb1"
    LCB = "lcb"
    EPSILON_GREEDY = "epsilon_greedy"
    MOST_SAMPLED = "most_sampled"
    GREEDY = "greedy"


class EvalFuncType(Enum):
    RANDOM_ROLLOUT = "random_rollout"


@dataclass
class PolicyConfig:
    type_: PolicyType
    seed: int | None = None
    C: float = 1.0
    epsilon: float = 0.1


@dataclass
class EvalFuncConfig:
    type_: EvalFuncType
    seed: int | None = None
    max_depth: int = 30


@dataclass
class MCTSConfig:
    seed: int | None = None
    num_samples: int = 256
    gamma: float = 1.0
    tree_policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(type_=PolicyType.UCB1)
    )
    final_policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(type_=PolicyType.LCB)
    )
    eval_func: EvalFuncConfig = field(
        default_factory=lambda: EvalFuncConfig(type_=EvalFuncType.RANDOM_ROLLOUT)
    )


POLICY_REGISTRY = {
    PolicyType.UCB1: UCB1,
    PolicyType.LCB: LCB,
    PolicyType.EPSILON_GREEDY: EpsilonGreedy,
    PolicyType.MOST_SAMPLED: MostSampled,
}

EVAL_FUNC_REGISTRY = {EvalFuncType.RANDOM_ROLLOUT: RandomRollout}


class MCTSPlayerFactory:
    def coerce_policy_config(self, raw_config: Mapping) -> PolicyConfig:
        if not isinstance(raw_config, Mapping):
            raise TypeError("policy config must be a mapping")

        # Required key
        try:
            policy_type = PolicyType(raw_config["type_"])
        except KeyError:
            raise KeyError("policy.type_ is required")

        # Reject unknown keys early
        allowed_keys = {"type_", "seed", "C", "epsilon"}
        unknown = set(raw_config) - allowed_keys
        if unknown:
            raise KeyError(f"Unknown policy config keys: {unknown}")

        # Construct config (defaults applied here)
        return PolicyConfig(
            type_=policy_type,
            seed=raw_config.get("seed"),
            C=raw_config.get("C", 1.0),
            epsilon=raw_config.get("epsilon", 0.1),
        )

    def coerce_eval_func_config(self, raw_config: Mapping) -> EvalFuncConfig:
        if not isinstance(raw_config, Mapping):
            raise TypeError("eval func config must be a mapping")

        # Required key
        try:
            eval_func_type = EvalFuncType(raw_config["type_"])
        except KeyError:
            raise KeyError("policy.type_ is required")

        # Reject unknown keys early
        allowed_keys = {"type_", "seed", "max_depth"}
        unknown = set(raw_config) - allowed_keys
        if unknown:
            raise KeyError(f"Unknown eval_func config keys: {unknown}")

        # Construct config (defaults applied here)
        return EvalFuncConfig(
            type_=eval_func_type,
            seed=raw_config.get("seed"),
            max_depth=raw_config.get("max_depth", 30),
        )

    def coerce_mcts_config(self, raw_config: Mapping) -> MCTSConfig:
        # Base seed for MCTS
        base_seed = raw_config.get("seed")

        # Coerce raw subconfigs
        tree_policy_raw = raw_config["tree_policy"]
        final_policy_raw = raw_config["final_policy"]
        eval_func_raw = raw_config["eval_func"]

        # If MCTS seed is provided, use SeedSequence to generate child seeds
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
            if eval_func_raw.get("seed") is None:
                eval_func_raw["seed"] = make_unique_int_seed(child_seeds[2])

        return MCTSConfig(
            seed=raw_config.get("seed"),
            num_samples=raw_config.get("num_samples", 256),
            gamma=raw_config.get("gamma", 0.1),
            tree_policy=self.coerce_policy_config(raw_config["tree_policy"]),
            final_policy=self.coerce_policy_config(raw_config["final_policy"]),
            eval_func=self.coerce_eval_func_config(raw_config["eval_func"]),
        )

    def make_policy(self, cfg: PolicyConfig) -> EdgePolicy:
        policy = POLICY_REGISTRY[cfg.type_]
        if cfg.type_ == PolicyType.UCB1:
            return policy(cfg.seed, cfg.C)
        if cfg.type_ in [PolicyType.LCB, PolicyType.MOST_SAMPLED]:
            return policy(cfg.seed)
        if cfg.type_ == PolicyType.EPSILON_GREEDY:
            return policy(cfg.seed, cfg.epsilon)
        raise ValueError(f"Unknown policy type: {cfg.type_}")

    def make_eval_func(self, cfg: EvalFuncConfig):
        eval_func = EVAL_FUNC_REGISTRY[cfg.type_]
        if cfg.type_ == EvalFuncType.RANDOM_ROLLOUT:
            return eval_func(cfg.seed, cfg.max_depth)
        raise ValueError(f"Unknown eval func type: {cfg.type_}")

    def make_mcts_player(self, cfg: MCTSConfig) -> MCTSPlayer:
        tree_policy = self.make_policy(cfg.tree_policy)
        final_policy = self.make_policy(cfg.final_policy)
        eval_func = self.make_eval_func(cfg.eval_func)

        return MCTSPlayer(
            seed=cfg.seed,
            num_samples=cfg.num_samples,
            gamma=cfg.gamma,
            tree_policy=tree_policy,
            final_policy=final_policy,
            eval_func=eval_func,
        )

    def __call__(self, raw_config: Mapping) -> MCTSPlayer:
        cfg = self.coerce_mcts_config(raw_config)
        player = self.make_mcts_player(cfg)
        return player
