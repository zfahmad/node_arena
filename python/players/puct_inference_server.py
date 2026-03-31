import os
import queue
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array, jit
from jax.typing import ArrayLike

from python.game_protocols import StateProtocol


class InferenceClient:
    def __init__(self, actor_id: int, request_q: Queue, response_q: Queue):
        self.actor_id = actor_id
        self.request_q = request_q
        self.response_q = response_q

    def __call__(self, state: StateProtocol) -> tuple[Any, Any]:
        self.request_q.put((state.to_array(), self.actor_id))
        value, policy = self.response_q.get()
        return value[0], policy

    def shutdown(self):
        self.request_q.put((None, self.actor_id))


class InferenceServer:
    def __init__(
        self,
        batch_size: int,
        num_actors: int,
        create_batch_input: Callable[[ArrayLike, Sequence[int]], Array],
        create_padding: Callable[..., Array],
        model: nnx.Module,
        ckpt_path: str,
        dims: Sequence[int],
    ) -> None:
        # Set the mini-batch size to be used for inference.
        self.batch_size = batch_size
        # Define the function to create batched inputs given a batch of state
        # arrays.
        self.create_batch_input = create_batch_input
        self.create_padding = create_padding

        self.num_actors = num_actors
        self.ckpt_path = ckpt_path
        self.model = model
        graphdef, params = nnx.split(self.model)
        self.graphdef = graphdef
        self.params = params
        self.infer = self._compile_inference()
        self._load_model()
        self.dims = tuple(dims)

    def _compile_inference(self):
        @jit
        def infer(params, x):
            model_instance = nnx.merge(self.graphdef, params)
            return model_instance(x)

        return infer

    def _load_model(self):
        # Load the model checkpoint at the path specified by ckpt_path.
        if os.path.exists(self.ckpt_path):
            checkpointer = ocp.StandardCheckpointer()
            self.params = checkpointer.restore(
                os.path.join(self.ckpt_path, "state"),
                target=self.params,
            )

    def __call__(
        self,
        request_q: Queue,
        response_qs: list[Queue],
        update_model: Event | None = None,
    ) -> None:
        terminated_actors: int = 0
        # Loop while actor processes are actively playing
        while terminated_actors < self.num_actors:
            batch = []
            actor_ids = []
            # Receive batch of data from request queue
            while len(batch) < self.batch_size:
                try:
                    state, actor_id = request_q.get(timeout=0.025)
                    if state is None:
                        terminated_actors += 1
                        continue
                    batch.append(state)
                    actor_ids.append(actor_id)
                except queue.Empty:
                    break
            if not batch:  # if no requests were received, start over the loop
                continue

            # Pad mini-batch if partially filled.
            # Allows the server to use a jit-compiled model without having to
            # recompile it
            effective_batch_size = len(batch)
            if effective_batch_size < self.batch_size:
                for _ in range(self.batch_size - effective_batch_size):
                    batch.append(self.create_padding(self.dims))

            input_batch = self.create_batch_input(jnp.array(batch), self.dims)
            values, policies = self.infer(self.params, input_batch)

            # Place the values and policies into the appropriate response queues
            for index, actor_id in enumerate(actor_ids):
                response_qs[actor_id].put(
                    (np.asarray(values[index]), np.asarray(policies[index]))
                )

            # Reloads the model weights given a signal.
            # Mainly used in AlphaZero self-play training.
            if (update_model is not None) and (update_model.is_set()):
                self._load_model()
                update_model.clear()
