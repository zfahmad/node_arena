"""
evaluate_model.py

Usage:
    evaluate_model.py [options] <config-file>

Options:
    -v --verbose             # Print to screen
    -l FILE --logging=LOG_FILE    # Log program to file [default: ]
"""

import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
from flax import nnx
from jax import Array, jit
from jax.typing import ArrayLike

from python.configs import Batch, ModelConfig, OptimizerConfig, TrainingConfig
from python.configure_logging import configure_logging
from python.data_reader import DataReader
from python.utils.seed_gen import generate_random_seeds


@dataclass
class EvaluationConfig:
    game: str
    size: list[int]
    policy_type: str
    batch_size: int
    dataset_path: str
    ckpt_path: str
    master_seed: int
    optimizer_cfg: OptimizerConfig
    model_cfg: ModelConfig


def load_model(
    cfg: ModelConfig,
) -> tuple[
    nnx.Module,
    Callable[[ArrayLike], Array],
    Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    model_module = importlib.import_module(f"python.models.{cfg.type_}_nn")
    Model = getattr(model_module, cfg.name)
    create_input_fn = getattr(model_module, "create_batch_input")
    preprocess_batch = getattr(model_module, "preprocess_batch")
    return Model(cfg.seed, **cfg.hypers), create_input_fn, preprocess_batch


def make_evaluation(create_input_fn, dims: Sequence[int]) -> Callable:
    dims = tuple(dims)

    def evaluate_model(
        model: nnx.Module,
        batch_states: Array,
        batch_values: Array,
        batch_masks: Array,
        batch_policies: Array,
    ) -> dict:
        batch_inputs = create_input_fn(batch_states, dims)
        predicted_values, output_policies = model(batch_inputs)
        policies = output_policies * batch_masks
        loss = jnp.mean(
            optax.squared_error(predicted_values, batch_values)
            + optax.softmax_cross_entropy(policies, batch_policies)
        )
        bins = jnp.array([-0.33, 0.33])
        predicted_outcomes = jnp.digitize(predicted_values, bins) - 1
        value_prediction_acc = jnp.mean(predicted_outcomes == batch_values)
        action_selection = jnp.argmax(policies, axis=1)
        selection_acc = jnp.count_nonzero(
            batch_policies[jnp.arange(batch_policies.shape[0]), action_selection]
        )

        return {
            "loss": loss,
            "pred_acc": value_prediction_acc,
            "sel_acc": selection_acc,
        }

    # return jit(evaluate_model)
    return evaluate_model


def main():
    from pprint import pprint

    from docopt import docopt

    args = docopt(__doc__)
    if args["--logging"]:
        configure_logging(args["--logging"])

    try:
        logging.info("Opening config file.")
        with open(args["<config-file>"]) as f:
            raw_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file does not exist: {args['<config-file>']}")
        sys.exit(1)

    ss = generate_random_seeds(raw_cfg["master_seed"], 1)

    cfg = TrainingConfig(
        game=raw_cfg["game"],
        size=raw_cfg["size"],
        dataset_path=raw_cfg["dataset_path"],
        policy_type=raw_cfg["policy_type"],
        batch_size=raw_cfg["batch_size"],
        ckpt_path=raw_cfg["ckpt_path"],
        optimizer_cfg=OptimizerConfig(**raw_cfg["optimizer_config"]),
        model_cfg=ModelConfig(
            raw_cfg["game"],
            raw_cfg["model_config"]["name"],
            ss[0],
            raw_cfg["model_config"]["hypers"],
        ),
    )

    pprint(cfg)

    if not os.path.exists(cfg.ckpt_path):
        logging.info(f"Path: {cfg.ckpt_path} does not exist.")
        logging.info("Making directory.")
        os.mkdir(cfg.ckpt_path)

    logging.info(f"Initializing model: {cfg.model_cfg.name}")
    model, create_input_fn, preprocess_batch = load_model(cfg.model_cfg)
    graphdef, params = nnx.split(model)
    if os.path.exists(cfg.ckpt_path):
        checkpointer = ocp.StandardCheckpointer()
        params = checkpointer.restore(
            os.path.join(cfg.ckpt_path, "state"),
            target=params,
        )
    model = nnx.merge(graphdef, params)

    evaluate_model = make_evaluation(create_input_fn, cfg.size)

    test_dr = DataReader(cfg.dataset_path, "testing", cfg.policy_type, cfg.batch_size)
    num_eval_batches = test_dr.size // cfg.batch_size
    metrics_sum = {"loss": 0.0, "pred_acc": 0.0, "sel_acc": 0.0}
    for _ in range(num_eval_batches):
        batch: Batch = test_dr.get_next_batch()
        batch_states, batch_values, batch_masks, batch_policies = preprocess_batch(batch, cfg.size)  # type: ignore
        metrics = evaluate_model(
            model, batch_states, batch_values, batch_masks, batch_policies
        )
        for k in metrics:
            metrics_sum[k] += metrics[k] * cfg.batch_size
    mean_metrics = {
        k: v / (cfg.batch_size * num_eval_batches) for k, v in metrics_sum.items()
    }
    print(mean_metrics)


if __name__ == "__main__":
    main()
