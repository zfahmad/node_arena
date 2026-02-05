import os
import random
import time
from multiprocessing import Array, Event, Process


def actor(actor_id: int, states, values, ready, done):

    print(f"Actor {actor_id} pid={os.getpid()}")

    for step in range(3):
        # Simulate MCTS work
        time.sleep(random.uniform(0.2, 0.6))

        # Write state
        state = actor_id * 10 + step
        states[actor_id] = state

        # Signal inference
        ready[actor_id].set()

        # Block until result arrives
        done[actor_id].wait()
        done[actor_id].clear()

        value = values[actor_id]
        print(f"Actor {actor_id} got value {value}")


def inference(states, values, ready, done, shutdown, batch_size):
    print(f"Inference pid={os.getpid()}")
    N = len(ready)

    while not shutdown.is_set():
        batch = []

        while len(batch) < batch_size and not shutdown.is_set():
            for i in range(N):
                if ready[i].is_set():
                    ready[i].clear()
                    batch.append(i)
                    if len(batch) == batch_size:
                        break
            time.sleep(0.01)

        if not batch:
            continue

        inputs = [states[i] for i in batch]
        print(f"Inference evaluating {inputs}")
        time.sleep(1.0)

        outputs = [x**2 for x in inputs]

        for i, v in zip(batch, outputs):
            values[i] = v
            done[i].set()

    print("Inference shutting down")


if __name__ == "__main__":
    N = 4

    states = Array("i", N)
    values = Array("i", N)

    ready = [Event() for _ in range(N)]
    done = [Event() for _ in range(N)]
    shutdown = Event()

    infer = Process(target=inference, args=(states, values, ready, done, shutdown, 3))
    infer.start()

    actors = [
        Process(target=actor, args=(i, states, values, ready, done)) for i in range(N)
    ]

    for p in actors:
        p.start()

    for p in actors:
        p.join()

    shutdown.set()
    infer.join()
