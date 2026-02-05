import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import random

# -------------------------------------------------
# Shared slot descriptor (passed to child processes)
# -------------------------------------------------
class SharedSlotsDesc:
    def __init__(
        self,
        num_slots,
        state_dim,
        state_name,
        value_name,
        ready,
        done,
        free_slots,
    ):
        self.num_slots = num_slots
        self.state_dim = state_dim
        self.state_name = state_name
        self.value_name = value_name
        self.ready = ready
        self.done = done
        self.free_slots = free_slots


# -------------------------------------------------
# Shared memory creator (main process only)
# -------------------------------------------------
class SharedSlots:
    def __init__(self, num_slots, state_dim):
        self.num_slots = num_slots
        self.state_dim = state_dim

        state_bytes = num_slots * state_dim * 4
        value_bytes = num_slots * 4

        self.state_shm = shared_memory.SharedMemory(
            create=True, size=state_bytes
        )
        self.value_shm = shared_memory.SharedMemory(
            create=True, size=value_bytes
        )

        self.states = np.ndarray(
            (num_slots, state_dim),
            dtype=np.float32,
            buffer=self.state_shm.buf,
        )
        self.values = np.ndarray(
            (num_slots,),
            dtype=np.float32,
            buffer=self.value_shm.buf,
        )

        self.ready = [mp.Event() for _ in range(num_slots)]
        self.done = [mp.Event() for _ in range(num_slots)]

        self.free_slots = mp.Queue()
        for i in range(num_slots):
            self.free_slots.put(i)

    def descriptor(self):
        return SharedSlotsDesc(
            self.num_slots,
            self.state_dim,
            self.state_shm.name,
            self.value_shm.name,
            self.ready,
            self.done,
            self.free_slots,
        )

    def close(self):
        self.state_shm.close()
        self.value_shm.close()

    def unlink(self):
        self.state_shm.unlink()
        self.value_shm.unlink()


# -------------------------------------------------
# Attach helper (used inside child processes)
# -------------------------------------------------
def attach_slots(desc: SharedSlotsDesc):
    state_shm = shared_memory.SharedMemory(name=desc.state_name)
    value_shm = shared_memory.SharedMemory(name=desc.value_name)

    states = np.ndarray(
        (desc.num_slots, desc.state_dim),
        dtype=np.float32,
        buffer=state_shm.buf,
    )
    values = np.ndarray(
        (desc.num_slots,),
        dtype=np.float32,
        buffer=value_shm.buf,
    )

    return state_shm, value_shm, states, values


# -------------------------------------------------
# Inference object
# -------------------------------------------------
class Inference:
    def __init__(self, desc: SharedSlotsDesc):
        self.desc = desc

    def run(self, shutdown_event):
        state_shm, value_shm, states, values = attach_slots(self.desc)

        print(f"[Inference] pid={mp.current_process().pid}")

        while not shutdown_event.is_set():
            batch = [
                i for i, e in enumerate(self.desc.ready) if e.is_set()
            ]

            if not batch:
                time.sleep(0.001)
                continue

            # Collect states
            batch_states = states[batch, 0]

            # Dummy "network": square
            time.sleep(0.2)
            batch_results = batch_states ** 2

            # Write back
            for i, r in zip(batch, batch_results):
                values[i] = r
                self.desc.ready[i].clear()
                self.desc.done[i].set()

        state_shm.close()
        value_shm.close()
        print("[Inference] shutdown")


# -------------------------------------------------
# Actor object
# -------------------------------------------------
class Actor:
    def __init__(self, actor_id, desc: SharedSlotsDesc):
        self.actor_id = actor_id
        self.desc = desc

    def run(self, steps=5):
        state_shm, value_shm, states, values = attach_slots(self.desc)

        print(f"[Actor {self.actor_id}] pid={mp.current_process().pid}")

        for step in range(steps):
            state = self.actor_id * 10 + step

            slot = self.desc.free_slots.get()

            states[slot, 0] = state
            self.desc.ready[slot].set()
            self.desc.done[slot].wait()

            result = values[slot]
            print(f"[Actor {self.actor_id}] state={state}, result={result}")

            self.desc.done[slot].clear()
            self.desc.free_slots.put(slot)

            time.sleep(random.uniform(0.1, 0.3))

        state_shm.close()
        value_shm.close()
        print(f"[Actor {self.actor_id}] finished")


# -------------------------------------------------
# Main (external process creation)
# -------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn")

    NUM_ACTORS = 3
    NUM_SLOTS = 4
    STATE_DIM = 1

    slots = SharedSlots(NUM_SLOTS, STATE_DIM)
    desc = slots.descriptor()

    shutdown_event = mp.Event()

    # Inference process
    inference = Inference(desc)
    infer_proc = mp.Process(
        target=inference.run,
        args=(shutdown_event,),
    )
    infer_proc.start()

    # Actor processes
    actors = [Actor(i, desc) for i in range(NUM_ACTORS)]
    actor_procs = [
        mp.Process(target=a.run)
        for a in actors
    ]

    for p in actor_procs:
        p.start()
    for p in actor_procs:
        p.join()

    # Shutdown inference
    shutdown_event.set()
    infer_proc.join()

    # Cleanup shared memory
    slots.close()
    slots.unlink()

    print("All done.")

