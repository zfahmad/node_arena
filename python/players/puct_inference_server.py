import importlib
import os
import queue
from multiprocessing import Process, Queue
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

        # Load the model checkpoint at the path specified by ckpt_path.
        if ckpt_path != "":
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            graphdef, state = nnx.split(model)
            checkpointer = ocp.StandardCheckpointer()
            state = checkpointer.restore(ckpt_path + "state", target=state)
            model = nnx.merge(graphdef, state)
        # jit compile the loaded model
        self.jit_model = jit(model)
        self.dims = tuple(dims)

    def __call__(self, request_q: Queue, response_qs: list[Queue]) -> None:
        terminated_actors: int = 0
        # Loop while actor processes are actively playing
        while terminated_actors < self.num_actors:
            batch = []
            actor_ids = []
            # Receive batch of data from request queue
            while len(batch) < self.batch_size:
                try:
                    state, actor_id = request_q.get(timeout=0.5)
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
            values, policies = self.jit_model(input_batch)

            for index, actor_id in enumerate(actor_ids):
                response_qs[actor_id].put(
                    (np.asarray(values[index]), np.asarray(policies[index]))
                )


def run_inference(request_q: Queue, response_qs: list[Queue], params: dict) -> None:
    # TODO: Maybe take model specs as dict?

    # Load model for specified game
    gm = importlib.import_module(f"python.models.{params["game_str"]}_nn")
    model = getattr(gm, params["model_type"])
    m = model(params["seed"], params["hypers"])

    # Create inference server
    print(f"Creating inference server: {os.getpid()}")
    inference_server = InferenceServer(
        params["batch_size"],
        params["num_actors"],
        gm.create_batch_input,
        gm.create_padding,
        m,
        params["model_ckpt_path"],
        [3, 3],
    )

    # Await requests and process mini-batches until all actors terminate
    # while not shutdown_event.is_set():
    inference_server(request_q, response_qs)


def run_clients(id: int, request_q: Queue, response_q: Queue):
    import random

    import python.wrappers.tic_tac_toe_wrapper as gm

    game = gm.Game()
    state = gm.State()
    game.reset(state)
    state = game.get_next_state(state, random.choice(game.get_actions(state)))
    state.print_board()
    print(f"Creating inference client: {os.getpid()}")
    client = InferenceClient(id, request_q, response_q)
    client(state)
    client.shutdown()


def main():
    import python.wrappers.tic_tac_toe_wrapper as gm

    game = gm.Game()
    print(game.get_id())
    state = gm.State()
    game.reset(state)
    num_actors = 5
    batch_size = 5
    request_q = Queue()
    response_qs = [Queue() for _ in range(num_actors)]
    model_type = "CNN"
    # ckpt_path = "/Users/zaheen/projects/node_arena/python/checkpoints/"
    ckpt_path = "/Users/zaheen/Documents/node_arena/test/checkpoints/60/"
    game_str = "tic_tac_toe"
    model_params = {
        "game_str": game_str,
        "batch_size": batch_size,
        "num_actors": num_actors,
        "model_type": model_type,
        "model_ckpt_path": ckpt_path,
        "seed": 0,
        "hypers": [3, 3],
    }

    inference_proc = Process(
        target=run_inference,
        args=(request_q, response_qs, model_params),
    )
    inference_proc.start()

    clients = [
        Process(target=run_clients, args=(i, request_q, response_qs[i]))
        for i in range(num_actors)
    ]

    for i in range(num_actors):
        clients[i].start()

    for i in range(num_actors):
        clients[i].join()

    inference_proc.join()


# if __name__ == "__main__":
#     main()
