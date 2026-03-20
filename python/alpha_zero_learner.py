import os
from multiprocessing.synchronize import Event
from pathlib import Path

import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from python.configs import Batch, GameConfig, ModelConfig, OptimizerConfig
from python.train_supervised_learning import load_model, make_train_step


class Learner:
    def __init__(
        self,
        game_cfg: GameConfig,
        model_cfg: ModelConfig,
        optimizer_cfg: OptimizerConfig,
        working_dir: str,
        ckpt_path: str,
        save_interval: int,
        update_model: Event,
    ) -> None:
        self.model, self.create_input_fn, self.preprocess_batch = load_model(model_cfg)
        self.graphdef, state = nnx.split(self.model)
        self.save_interval = save_interval
        self.working_dir = os.path.join(working_dir, "checkpoints")
        ckpt_opts = ocp.CheckpointManagerOptions(
            save_interval_steps=self.save_interval
        )
        ckptrs = {"state": ocp.StandardCheckpointer()}
        self.mngr = ocp.CheckpointManager(
            self.working_dir, options=ckpt_opts, checkpointers=ckptrs
        )
        step = 0
        if ckpt_path != "":
            if os.path.exists(ckpt_path):
                checkpointer = ocp.StandardCheckpointer()
                state = checkpointer.restore(ckpt_path + "state", target=state)
                self.model = nnx.merge(self.graphdef, state)
                step = self.mngr.latest_step()
        if step is None:
            self.step = 0
        else:
            self.step = step

        self.dims = game_cfg.size
        O = getattr(optax, optimizer_cfg.name)
        self.optimizer = nnx.Optimizer(
            self.model, O(**optimizer_cfg.kwargs), wrt=nnx.Param
        )
        self.train_step = make_train_step(self.create_input_fn, game_cfg.size)
        self.update_model = update_model

    def _update_latest_symlink(self, step: int):
        ckpt_dir = Path(self.working_dir)
        latest = ckpt_dir / "latest"
        target = ckpt_dir / str(step)

        if latest.exists() or latest.is_symlink():
            latest.unlink()

        latest.symlink_to(target, target_is_directory=True)

    def __call__(self, batch: Batch):
        batch_states, batch_values, batch_masks, batch_policies = self.preprocess_batch(batch, self.dims)  # type: ignore
        self.model, _ = self.train_step(
            self.model,
            self.optimizer,
            batch_states,
            batch_values,
            batch_masks,
            batch_policies,
        )
        if self.mngr.should_save(self.step):
            _, state = nnx.split(self.model)
            self.mngr.save(self.step, {"state": state})
            self.mngr.wait_until_finished()
            self._update_latest_symlink(self.step)
            self.update_model.set()
        self.step += 1
