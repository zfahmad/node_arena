import importlib
import os
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from typing import Callable

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike

# TODO: Implement response queues so that inference results are sent to
# InferenceClient objects


class InferenceServer:
    def __init__(
        self,
        batch_size: int,
        num_actors: int,
        create_batch_input: Callable[[ArrayLike], Array],
        create_padding: Callable[..., Array],
        model: nnx.Module,
        ckpt_path: str,
    ):
        # Set the mini-batch size to be used for inference.
        self.batch_size = batch_size
        # Define the function to create batched inputs given a batch of state
        # arrays.
        self.create_batch_input = create_batch_input
        self.create_padding = create_padding
        self.num_actors = num_actors

        # Load the model at the path specified by ckpt_path.
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.model = model
        graphdef, state = nnx.split(self.model)
        checkpointer = ocp.StandardCheckpointer()
        state = checkpointer.restore(ckpt_path + "state", target=state)
        self.model = nnx.merge(graphdef, state)

    def __call__(self, request_q: Queue):
        terminated_actors: int = 0

        # Loop while actor processes are actively playing
        while terminated_actors < self.num_actors:
            batch = []
            # Receive batch of data from request queue
            while len(batch) < self.batch_size:
                try:
                    item = request_q.get(timeout=0.5)
                    if item is None:
                        terminated_actors += 1
                        continue
                    batch.append(item)
                except:
                    break

            # Pad mini-batch if partially filled
            if len(batch) < self.batch_size:
                for _ in range(self.batch_size - len(batch)):
                    batch.append(self.create_padding())

            input_batch = self.create_batch_input(jnp.array(batch))
            values = self.model(input_batch)
            print(values)
            print(terminated_actors)


def run_inference(
    request_q: Queue,
    shutdown_event: EventClass,
    num_actors: int,
    batch_size: int,
    game_str: str,
    ckpt_path: str,
):
    # Load model for specified game
    gm = importlib.import_module(f"python.models.{game_str}_nn")
    model = gm.CNN(nnx.Rngs(0))

    # Create inference server
    inference_server = InferenceServer(
        batch_size,
        num_actors,
        gm.create_batch_input,
        gm.create_padding,
        model,
        ckpt_path,
    )

    # Await requests and process mini-batches until all actors terminate
    while not shutdown_event.is_set():
        inference_server(request_q)


if __name__ == "__main__":
    import random
    import time

    import python.wrappers.tic_tac_toe_wrapper as gm

    game = gm.Game()
    print(game.get_id())
    state = gm.State()
    game.reset(state)
    state_batch = []
    request_q = Queue()
    shutdown_event = Event()
    num_actors = 5
    batch_size = 5
    ckpt_path = "/Users/zaheen/projects/node_arena/python/checkpoints/"
    game_str = "tic_tac_toe"
    for _ in range(batch_size):
        state_batch.append(state.to_array())
        state = game.get_next_state(state, random.choice(game.get_actions(state)))

    inference_proc = Process(
        target=run_inference,
        args=(request_q, shutdown_event, num_actors, batch_size, game_str, ckpt_path),
    )
    inference_proc.start()

    for state in state_batch:
        request_q.put(state)
    for i in range(3):
        request_q.put(state_batch[i])

    time.sleep(2)
    for _ in range(batch_size):
        request_q.put(None)
    shutdown_event.set()
    inference_proc.join()
