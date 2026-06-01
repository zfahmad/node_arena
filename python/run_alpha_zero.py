"""
run_alpha_zero.py
Author: Zaheen Farraz Ahmad

Runs AlphaZero --- creates config file for a run given a set of passed
parameters. Makes a directory in a specified path and outputs the data from the
run in the directory.

Need to specify the output directory. The base config file used is
data/train_alpha_zero_template.yaml.

Usage:
    run_alpha_zero.py [options] <game> <size> <output-dir>

Options:
    -h --help                   # Show this screen
    --base-config=bc            # Path to Yaml config file to use as a base [default: ./data/train_alpha_zero_template.yaml]
    --base-eval-config=bec      # Path to Yaml config file to use as a base [default: ./data/evaluate_alpha_zero_template.yaml]
    --seed=s                    # Seed for random number generator [default: 0]
    --num-procs=n               # Number of actor processes [default: 1]
    --num-samples=n             # PUCT budget [default: 16]
    --max-turns=n               # Max number of turns before truncation [default: 50]
    --gamma=gamma               # Discount factor [default: 1.0]
    --epsilon=epsilon           # Dirichlet weight [default: 0.25]
    --alpha=alpha               # Dirichlet alpha [default: 0.2]
    --C=c                       # Exploration constant [default: 1.0]
    --num-res-blocks=nrs        # Number of residual blocks [default: 1]
    --batch-size=batc           # Inference batch size [default: 32]
    --buffer-size=bs            # Size of replay buffer [default: 1000]
    --learning-rate=lr          # Learning rate for optimizer [default: 0.001]
    --num-iterations=ni         # Learning rate for optimizer [default: 500]
    --save-interval=si          # Checkpointing interval [default: 500]
    --resume                    # Resume training [default: False]
    --mcts-samples=ms           # List of MCTS sampling budgets [default: 8,16,32]
    --eval-games-per-proc=gpp   # Number of games to play per process [default: 10]
"""

import os
import pathlib
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pprint import pprint

import docopt
import yaml


@dataclass
class AlphaZeroTrainingConfig:
    game: str
    size: list[int]
    params: list[int]
    seed: int
    num_procs: int
    num_samples: int
    max_turns: int
    gamma: float
    dirichlet_epsilon: float
    dirichlet_alpha: float
    C: float
    num_residual_blocks: int
    batch_size: int
    buffer_size: int
    learning_rate: float
    num_iterations: int
    save_interval: int


@dataclass
class AlphaZeroEvaluationConfig:
    game: str
    size: list[int]
    params: list[int]
    seed: int
    num_procs: int
    num_samples: int
    max_turns: int
    gamma: float
    dirichlet_epsilon: float
    dirichlet_alpha: float
    C: float
    num_residual_blocks: int
    batch_size: int
    mcts_samples: list[int]
    games_per_proc: int


def read_base_json(file_path: pathlib.Path) -> dict:
    with open(file_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    f.close()
    return raw_cfg


def create_training_config(base_cfg: dict, cfg: AlphaZeroTrainingConfig) -> dict:
    new_raw_cfg = deepcopy(base_cfg)
    new_raw_cfg["game"]["type_"] = cfg.game
    new_raw_cfg["game"]["size"] = cfg.size
    new_raw_cfg["game"]["params"] = cfg.params
    new_raw_cfg["master_seed"] = cfg.seed
    new_raw_cfg["num_procs"] = cfg.num_procs
    new_raw_cfg["max_turns"] = cfg.max_turns
    new_raw_cfg["player"]["params"]["num_samples"] = cfg.num_samples
    new_raw_cfg["player"]["params"]["gamma"] = cfg.gamma
    new_raw_cfg["player"]["params"]["dirichlet_epsilon"] = cfg.dirichlet_epsilon
    new_raw_cfg["player"]["params"]["dirichlet_alpha"] = cfg.dirichlet_alpha
    new_raw_cfg["player"]["params"]["C"] = cfg.C
    new_raw_cfg["inference_servers"][0]["model_cfg"]["hypers"][
        "num_blocks"
    ] = cfg.num_residual_blocks
    new_raw_cfg["inference_servers"][0]["model_cfg"]["hypers"]["dims"] = cfg.size
    new_raw_cfg["inference_servers"][0]["model_cfg"]["batch_size"] = cfg.batch_size
    new_raw_cfg["learner"]["batch_size"] = cfg.batch_size
    new_raw_cfg["learner"]["optimizer"]["kwargs"]["learning_rate"] = cfg.learning_rate
    new_raw_cfg["num_iterations"] = cfg.num_iterations
    new_raw_cfg["learner"]["save_interval"] = cfg.save_interval

    return new_raw_cfg


def create_evaluation_config(base_cfg: dict, cfg: AlphaZeroEvaluationConfig) -> dict:
    new_raw_cfg = deepcopy(base_cfg)
    new_raw_cfg["game"]["type_"] = cfg.game
    new_raw_cfg["game"]["size"] = cfg.size
    new_raw_cfg["game"]["params"] = cfg.params
    new_raw_cfg["master_seed"] = cfg.seed
    new_raw_cfg["num_procs"] = cfg.num_procs
    new_raw_cfg["max_turns"] = cfg.max_turns
    new_raw_cfg["player"]["params"]["num_samples"] = cfg.num_samples
    new_raw_cfg["player"]["params"]["gamma"] = cfg.gamma
    new_raw_cfg["player"]["params"]["dirichlet_epsilon"] = cfg.dirichlet_epsilon
    new_raw_cfg["player"]["params"]["dirichlet_alpha"] = cfg.dirichlet_alpha
    new_raw_cfg["player"]["params"]["C"] = cfg.C
    new_raw_cfg["inference_servers"][0]["model_cfg"]["hypers"][
        "num_blocks"
    ] = cfg.num_residual_blocks
    new_raw_cfg["num_samples"] = cfg.mcts_samples
    new_raw_cfg["games_per_proc"] = cfg.games_per_proc
    new_raw_cfg["inference_servers"][0]["model_cfg"]["hypers"]["dims"] = cfg.size
    new_raw_cfg["inference_servers"][0]["model_cfg"]["batch_size"] = cfg.batch_size
    return new_raw_cfg


def parse_training_args(args: dict) -> AlphaZeroTrainingConfig:
    size = args["<size>"]
    size = [int(x) for x in size.split(",")]
    game = args["<game>"]
    if game == "chinese_checkers":
        params = size
    else:
        params = []

    cfg = AlphaZeroTrainingConfig(
        game=str(args["<game>"]),
        size=size,
        params=params,
        seed=int(args["--seed"]),
        num_procs=int(args["--num-procs"]),
        num_samples=int(args["--num-samples"]),
        max_turns=int(args["--max-turns"]),
        gamma=float(args["--gamma"]),
        dirichlet_epsilon=float(args["--epsilon"]),
        dirichlet_alpha=float(args["--alpha"]),
        C=float(args["--C"]),
        num_residual_blocks=int(args["--num-res-blocks"]),
        batch_size=int(args["--batch-size"]),
        buffer_size=int(args["--buffer-size"]),
        learning_rate=float(args["--learning-rate"]),
        num_iterations=int(args["--num-iterations"]),
        save_interval=int(args["--save-interval"]),
    )
    return cfg


def parse_evaluation_args(args: dict) -> AlphaZeroEvaluationConfig:
    size = args["<size>"]
    size = [int(x) for x in size.split(",")]
    game = args["<game>"]
    if game == "chinese_checkers":
        params = size
    else:
        params = []
    mcts_samples = args["--mcts-samples"]
    mcts_samples = [int(x) for x in mcts_samples.split(",")]

    cfg = AlphaZeroEvaluationConfig(
        game=str(args["<game>"]),
        size=size,
        params=params,
        seed=int(args["--seed"]),
        num_procs=int(args["--num-procs"]),
        num_samples=int(args["--num-samples"]),
        max_turns=int(args["--max-turns"]),
        gamma=float(args["--gamma"]),
        dirichlet_epsilon=float(args["--epsilon"]),
        dirichlet_alpha=float(args["--alpha"]),
        C=float(args["--C"]),
        num_residual_blocks=int(args["--num-res-blocks"]),
        batch_size=int(args["--batch-size"]),
        mcts_samples=mcts_samples,
        games_per_proc=int(args["--eval-games-per-proc"]),
    )
    return cfg


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    # Read base config file or template file
    # Create new config file with experiment specs
    base_file = pathlib.Path(args["--base-config"])
    base_training_cfg = read_base_json(base_file)
    training_cfg = parse_training_args(args)
    raw_training_cfg = create_training_config(base_training_cfg, training_cfg)
    base_eval_file = pathlib.Path(args["--base-eval-config"])
    base_evaluation_cfg = read_base_json(base_eval_file)
    evaluation_cfg = parse_evaluation_args(args)
    raw_evaluation_cfg = create_evaluation_config(base_evaluation_cfg, evaluation_cfg)

    # Create output directory for games and checkpoints
    output_dir = pathlib.Path(args["<output-dir>"])
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "alpha_zero.yaml", "w") as f:
        yaml.dump(raw_training_cfg, f)
    f.close()

    with open(output_dir / "eval_alpha_zero.yaml", "w") as f:
        yaml.dump(raw_evaluation_cfg, f)
    f.close()

    subprocess.run(
        [
            "python",
            "python/train_alpha_zero.py",
            str(output_dir / "alpha_zero.yaml"),
            "--output=" + str(output_dir),
        ]
    )

    subprocess.run(
        [
            "python",
            "python/evaluate_alpha_zero.py",
            str(output_dir / "eval_alpha_zero.yaml"),
            "--output=" + str(output_dir),
        ]
    )
