"""
train_unsupervised_learning.py

Usage:
    train_unsupervised_learning.py [options] <config-file>

Options:
    -v --verbose                  # Print to screen
    -l FILE --logging=LOG_FILE    # Log program to file [default: ]
"""

import importlib
import logging
import os
import sys
from collections.abc import Sequence
from typing import Callable

import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import nnx
from jax import Array, jit
from jax.typing import ArrayLike
from orbax import checkpoint as ckp

from python.configs import Batch, ModelConfig, OptimizerConfig, TrainingConfig
from python.configure_logging import configure_logging
from python.data_reader import DataReader
from python.utils.seed_gen import generate_random_seeds


def make_train_step(create_input_fn, dims: Sequence[int]) -> Callable:
    dims = tuple(dims)

    def loss_fn(
        model: nnx.Module,
        batch_inputs: Array,
        batch_values: Array,
        batch_masks: Array,
        batch_policies: Array,
    ) -> Array:
        predicted_values, output_logits = model(batch_inputs)
        masked_logits = output_logits + (1 - batch_masks) * (-1e9)
        batch_policies = batch_policies / jnp.sum(batch_policies, axis=1, keepdims=True)
        entropy = -jnp.sum(
            nnx.softmax(output_logits) * nnx.log_softmax(output_logits),
            axis=-1,
        )
        value_loss = jnp.mean(optax.squared_error(predicted_values, batch_values))
        policy_loss = jnp.mean(
            optax.softmax_cross_entropy(masked_logits, batch_policies)
        )
        # TODO: Use command line arg for regularization
        loss = value_loss + policy_loss - 1e-3 * jnp.mean(entropy)

        return loss

    grad_fn = nnx.value_and_grad(loss_fn)

    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch_states: ArrayLike,
        batch_values: ArrayLike,
        batch_masks: ArrayLike,
        batch_policies: ArrayLike,
    ) -> tuple[nnx.Module, Array]:
        batch_inputs = create_input_fn(batch_states, dims)
        loss, grads = grad_fn(
            model, batch_inputs, batch_values, batch_masks, batch_policies
        )
        optimizer.update(model, grads)
        return model, loss

    # NOTE: Should I jit compile?
    return train_step


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
        predicted_values, output_logits = model(batch_inputs)
        masked_logits = output_logits + (1 - batch_masks) * (-1e9)
        batch_policies = batch_policies / jnp.sum(batch_policies, axis=1, keepdims=True)
        value_loss = jnp.mean(optax.squared_error(predicted_values, batch_values))
        policy_loss = jnp.mean(
            optax.softmax_cross_entropy(masked_logits, batch_policies)
        )
        loss = value_loss + policy_loss
        bins = jnp.array([-0.33, 0.33])
        predicted_outcomes = jnp.digitize(predicted_values, bins) - 1
        value_prediction_acc = jnp.mean(predicted_outcomes == batch_values)
        action_selection = jnp.argmax(masked_logits, axis=1)
        selection_acc = (
            jnp.count_nonzero(
                batch_policies[jnp.arange(batch_policies.shape[0]), action_selection]
            )
            / batch_policies.shape[0]
        )

        return {
            "loss": loss,
            "pred_acc": value_prediction_acc,
            "sel_acc": selection_acc,
        }

    return jit(evaluate_model)


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


def main():
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
        game=raw_cfg["training_config"]["game"],
        size=raw_cfg["training_config"]["size"],
        dataset_path=raw_cfg["training_config"]["dataset_path"],
        policy_type=raw_cfg["training_config"]["policy_type"],
        num_iterations=raw_cfg["training_config"]["num_iterations"],
        num_epochs=raw_cfg["training_config"]["num_epochs"],
        batch_size=raw_cfg["training_config"]["batch_size"],
        ckpt_path=raw_cfg["training_config"]["ckpt_path"],
        eval_interval=raw_cfg["training_config"]["eval_interval"],
        optimizer_cfg=OptimizerConfig(**raw_cfg["optimizer_config"]),
        model_cfg=ModelConfig(
            raw_cfg["training_config"]["game"],
            raw_cfg["model_config"]["name"],
            ss[0],
            raw_cfg["model_config"]["hypers"],
        ),
    )

    if not os.path.exists(cfg.ckpt_path):
        logging.info(f"Path: {cfg.ckpt_path} does not exist.")
        logging.info("Making directory.")
        os.mkdir(cfg.ckpt_path)
    ckpt_opts = ckp.CheckpointManagerOptions(
        max_to_keep=3, save_interval_steps=cfg.eval_interval
    )
    ckptrs = {"state": ckp.StandardCheckpointer()}
    mngr = ckp.CheckpointManager(cfg.ckpt_path, options=ckpt_opts, checkpointers=ckptrs)

    logging.info(f"Initializing model: {cfg.model_cfg.name}")
    model, create_input_fn, preprocess_batch = load_model(cfg.model_cfg)
    logging.info(f"Loading optimizer: {cfg.optimizer_cfg.name}")
    O = getattr(optax, cfg.optimizer_cfg.name)
    optimizer = nnx.Optimizer(model, O(**cfg.optimizer_cfg.kwargs), wrt=nnx.Param)
    train_step = make_train_step(create_input_fn, cfg.size)
    evaluate_model = make_evaluation(create_input_fn, cfg.size)

    # Load the datasets
    # Each data sample in the sets must consist of a compact state, a value
    # prediction and a policy
    logging.info(f"Loading dataset: {cfg.dataset_path}")
    train_dr = DataReader(cfg.dataset_path, "training", cfg.policy_type, cfg.batch_size)
    test_dr = DataReader(cfg.dataset_path, "testing", cfg.policy_type, cfg.batch_size)

    if cfg.num_iterations != 0:
        num_iterations = cfg.num_iterations
    else:
        num_iterations = train_dr.size // cfg.batch_size

    logging.info("Begin training...")
    iteration = 0
    for epoch in range(cfg.num_epochs):
        logging.info(f"Starting epoch: {epoch}")
        for step in range(num_iterations):

            batch: Batch = train_dr.get_next_batch()
            batch_states, batch_values, batch_masks, batch_policies = preprocess_batch(batch, cfg.size)  # type: ignore
            model, _ = train_step(
                model,
                optimizer,
                batch_states,
                batch_values,
                batch_masks,
                batch_policies,
            )

            # Save model checkpoint
            _, state = nnx.split(model)
            mngr.save(iteration, {"state": state})

            # Evaluate accuracy of the model
            if not (step % cfg.eval_interval):
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
                    k: v / (cfg.batch_size * num_eval_batches)
                    for k, v in metrics_sum.items()
                }
                logging.info(
                    f"epoch: {epoch:3} step: {step:4} "
                    f"loss: {mean_metrics['loss']:.4f} "
                    f"pred_acc: {mean_metrics['pred_acc']:.4f} "
                    f"sel_acc: {mean_metrics['sel_acc']:.4f}"
                )
                if args["--verbose"]:
                    print(
                        f"epoch: {epoch:3} step: {step:4} "
                        f"loss: {mean_metrics['loss']:.4f} "
                        f"pred_acc: {mean_metrics['pred_acc']:.4f} "
                        f"sel_acc: {mean_metrics['sel_acc']:.4f}"
                    )
            iteration += 1

        if args["--verbose"]:
            print(30 * "-")

    logging.info("Ending training.")
    mngr.wait_until_finished()
    logging.info("Checkpoint manager finished.")


if __name__ == "__main__":
    main()
