from dataclasses import dataclass, field
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from typing import Any, Optional

import numpy as np


@dataclass
class PlayerConfig:
    type_: str
    params: dict
    server: Optional[str] = None


@dataclass
class GameConfig:
    type_: str
    params: list[int]
    size: list[int]
    initial_state: str = ""


@dataclass
class InferenceServerConfig:
    game_str: str
    name: str
    type_: str
    num_actors: int
    batch_size: int
    model_type: str
    ckpt_dir: str
    seed: int
    dims: list[int]


@dataclass
class TrainingConfig:
    game: str
    size: list[int]
    dataset_path: str
    policy_type: str
    optimizer_cfg: OptimizerConfig
    model_cfg: ModelConfig
    eval_interval: int = 1
    num_iterations: int = 0
    num_epochs: int = 1
    batch_size: int = 32
    ckpt_path: str = "./"


@dataclass
class ModelConfig:
    type_: str
    name: str
    seed: int
    hypers: list[Any]


@dataclass
class OptimizerConfig:
    name: str
    kwargs: dict


@dataclass
class LearnerConfig:
    seed: int
    working_dir: str
    batch_size: int
    buffer_size: int
    ckpt_path: str
    save_interval: int
    update_model: Event
    game_cfg: GameConfig
    model_cfg: ModelConfig
    optimizer_cfg: OptimizerConfig


@dataclass
class Config:
    output: str
    verbose: bool
    max_turns: int
    num_procs: int
    game: GameConfig
    player: PlayerConfig
    inference_servers: list[InferenceServerConfig] = field(default_factory=list)


@dataclass
class InferenceEndpoints:
    request_queue: Queue
    response_queues: list[Queue]


@dataclass
class DensePolicy:
    mask: np.ndarray
    policy: np.ndarray


@dataclass
class SparsePolicy:
    actions: np.ndarray
    weights: np.ndarray


@dataclass
class Batch:
    states: np.ndarray
    values: np.ndarray
    dense_policy: Optional[DensePolicy] = None
    sparse_policy: Optional[SparsePolicy] = None
