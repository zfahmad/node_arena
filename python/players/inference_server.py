import os
import queue
from collections import deque
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from typing import Any, Callable, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array, jit
from jax.typing import ArrayLike

from python.game_protocols import StateProtocol


class InferenceClient:
    def __init__(self, actor_id, request_q, response_q):
        self.actor_id = actor_id
        self.request_q = request_q
        self.response_q = response_q

    def submit(self, states: np.ndarray) -> None:
        """states: [k, ...]. Returns immediately."""
        self.request_q.put((states, self.actor_id))

    def collect(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        value_parts, policy_parts, got = [], [], 0
        while got < k:
            value_part, policy_part = self.response_q.get()  # [n, ...] slice
            value_parts.append(value_part)
            policy_parts.append(policy_part)
            got += len(value_part)
        return np.concatenate(value_parts), np.concatenate(policy_parts)

    def evaluate_many(self, states):
        self.submit(states)
        return self.collect(len(states))

    def evaluate_states(
        self, states: Sequence[StateProtocol]
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch = np.stack(
            [s.to_array() for s in states]
        )  # list of State -> [k, ...] array
        return self.evaluate_many(batch)

    def __call__(self, states: Sequence[StateProtocol]):
        return self.evaluate_states(states)

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
        per_actor_cap=None,
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
        self.per_actor_cap = per_actor_cap or batch_size

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

        pending = [deque() for _ in range(self.num_actors)]  # entries: [states, offset]
        n_pending = 0

        while terminated_actors < self.num_actors or n_pending > 0:
            # Fill the buffer
            while n_pending < self.batch_size:
                try:
                    states, actor_id = request_q.get(timeout=0.025)
                except queue.Empty:
                    break
                if states is None:
                    terminated_actors += 1
                    continue
                pending[actor_id].append([states, 0])
                n_pending += len(states)

            if n_pending == 0:
                continue

            # Build one mini-batch, round-robin with a per-actor cap
            rows, runs, budget = [], [], self.batch_size
            while budget > 0 and any(pending):
                for actor_id, dq in enumerate(pending):
                    if not dq or budget == 0:
                        continue
                    entry = dq[0]
                    states, off = entry
                    take = min(len(states) - off, budget, self.per_actor_cap)
                    rows.append(states[off : off + take])
                    runs.append((actor_id, take))
                    budget -= take
                    n_pending -= take
                    if off + take == len(states):
                        dq.popleft()
                    else:
                        entry[1] += take

            batch = np.concatenate(rows)
            n_real = len(batch)
            if n_real < self.batch_size:
                pad = np.stack(
                    [
                        self.create_padding(self.dims)
                        for _ in range(self.batch_size - n_real)
                    ]
                )
                batch = np.concatenate([batch, pad])

            input_batch = self.create_batch_input(jnp.array(batch), self.dims)
            values, policies = self.infer(self.params, input_batch)
            values = np.asarray(values)  # one transfer
            policies = np.asarray(policies)

            start = 0
            for actor_id, n in runs:  # same order the batch was built
                response_qs[actor_id].put(
                    (values[start : start + n], policies[start : start + n])
                )
                start += n

            # Reloads the model weights given a signal.
            # Mainly used in AlphaZero self-play training.
            if (update_model is not None) and (update_model.is_set()):
                self._load_model()
                update_model.clear()
