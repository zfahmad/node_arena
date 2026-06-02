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
    --base-train-config=bc      # Path to Yaml config file to use as a base
    --base-eval-config=bec      # Path to Yaml config file to use as a base
    --seed=s                    # Seed for random number generator
    --learning-rate=lr          # Learning rate for optimizer
    --resume                    # Resume training
"""

import os
import pathlib
import subprocess
from copy import deepcopy
from pprint import pprint

import docopt
import yaml


def read_base_yaml(file_path: pathlib.Path) -> dict:
    with open(file_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    f.close()
    return raw_cfg


def save_config(file_path: pathlib.Path, cfg: dict) -> None:
    with open(file_path, "w") as f:
        yaml.safe_dump(cfg, f)
    f.close()


def create_training_config(base_cfg: dict, args: dict) -> dict:
    new_cfg = deepcopy(base_cfg)

    if args["--seed"] is not None:
        seed = int(args["--seed"])
        new_cfg["master_seed"] = seed
    if args["--learning-rate"] is not None:
        learning_rate = float(args["--learning-rate"])
        new_cfg["learning_rate"]["optimizer"]["kwargs"]["learning_rate"] = learning_rate

    return new_cfg


def create_evaluation_config(base_cfg: dict, args: dict) -> dict:
    new_cfg = deepcopy(base_cfg)

    if args["--seed"] is not None:
        seed = int(args["--seed"])
        new_cfg["master_seed"] = seed

    return new_cfg


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    # Read base config file or template file
    # Create new config file with experiment specs
    base_file = pathlib.Path(args["--base-train-config"])
    base_training_cfg = read_base_yaml(base_file)
    base_eval_file = pathlib.Path(args["--base-eval-config"])
    base_evaluation_cfg = read_base_yaml(base_eval_file)

    new_training_cfg = create_training_config(base_training_cfg, args)
    new_eval_cfg = create_evaluation_config(base_evaluation_cfg, args)

    # Create output directory for games and checkpoints
    output_dir = pathlib.Path(args["<output-dir>"])
    os.makedirs(output_dir, exist_ok=True)
    save_config(output_dir / "alpha_zero.yaml", new_training_cfg)
    save_config(output_dir / "eval_alpha_zero.yaml", new_eval_cfg)

    # subprocess.run(
    #     [
    #         "python",
    #         "python/train_alpha_zero.py",
    #         str(output_dir / "alpha_zero.yaml"),
    #         "--output=" + str(output_dir),
    #     ]
    # )
    #
    # subprocess.run(
    #     [
    #         "python",
    #         "python/evaluate_alpha_zero.py",
    #         str(output_dir / "eval_alpha_zero.yaml"),
    #         "--output=" + str(output_dir),
    #     ]
    # )
