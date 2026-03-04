import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import h5py
import numpy as np
import numpy.random as rnd


# TODO: Update read_file to handle sparse policies


@dataclass
class FileIndex:
    files: deque[str] = field(default_factory=deque)
    num_states: deque[int] = field(default_factory=deque)


@dataclass
class DensePolicy:
    mask: np.ndarray
    policy: np.ndarray


@dataclass
class SparsePolicy:
    actions: np.ndarray
    weights: np.ndarray


@dataclass
class Batch:
    states: np.ndarray
    values: np.ndarray
    dense_policy: Optional[DensePolicy] = None
    sparse_policy: Optional[SparsePolicy] = None


class ReplayBuffer:
    def __init__(
        self, seed: int, data_dir: str, batch_size: int, buffer_size: int
    ) -> None:
        self.rand = rnd.default_rng(seed)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.file_idx = FileIndex()
        self.self_play_dir = os.path.join(self.data_dir, "self_play")
        self.archive_dir = os.path.join(self.data_dir, "archive")
        os.makedirs(self.self_play_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
    
    # NOTE: Read file currently does not handle sparse policies
    def read_file(self, file_idx: int, num_tuples: int) -> Tuple[np.ndarray, ...]:
        # Read a file an collect a random sample of num_tuples data tuples
        file_path = self.file_idx.files[file_idx]
        size = self.file_idx.num_states[file_idx]

        f = h5py.File(file_path, "r")
        idx = sorted(self.rand.integers(low=0, high=size, size=num_tuples))  # type: ignore
        states: np.ndarray = f["states"][idx]  # type: ignore
        masks: np.ndarray = f["masks"][idx]  # type: ignore
        policies: np.ndarray = f["policies"][idx]  # type: ignore
        values: np.ndarray = f["values"][idx]  # type: ignore
        f.close()

        return states, masks, policies, values

    def get_next_batch(self) -> Batch:
        weights = np.array(self.file_idx.num_states)
        weights = weights / np.sum(weights)
        idxs = np.sort(
            self.rand.choice(np.arange(len(weights)), size=self.batch_size, p=weights)
        )
        counts = Counter(idxs)

        batch_states = []
        batch_masks = []
        batch_policies = []
        batch_values = []
        for idx, count in counts.items():
            states, masks, policies, values = self.read_file(idx, count)
            batch_states.append(states)
            batch_masks.append(masks)
            batch_policies.append(policies)
            batch_values.append(values)

        return Batch(
            states=np.concatenate(batch_states),
            values=np.concatenate(batch_values),
            dense_policy=DensePolicy(
                np.concatenate(batch_masks), np.concatenate(batch_policies)
            ),
        )

    def add_to_index(self):
        """
        Detect new files in self_play, add them to the buffer index,
        enforce buffer size, and optionally move to archive.
        """
        # List new files
        files = sorted(
            os.listdir(self.self_play_dir)
        )  # Optional sort for chronological order

        for file in files:
            file_path = os.path.join(self.self_play_dir, file)

            # Skip files already in the buffer
            if file_path in self.file_idx.files:
                continue

            # Open file to get number of positions
            with h5py.File(file_path, "r") as f:
                num_states = f["states"].shape[0]  # type: ignore

            # Enforce buffer sliding window
            while len(self.file_idx.files) >= self.buffer_size:
                self.file_idx.files.popleft()
                self.file_idx.num_states.popleft()

            # Add new file to index
            self.file_idx.files.append(file_path)
            self.file_idx.num_states.append(num_states)

            # Move file to archive
            archived_path = os.path.join(self.archive_dir, file)
            os.rename(file_path, archived_path)
            # Update index to point to new location
            self.file_idx.files[-1] = archived_path


if __name__ == "__main__":
    data_dir: str = "/Users/zaheen/Documents/node_arena/az_test"
    rb = ReplayBuffer(0, data_dir, 32, 100)
    rb.add_to_index()
    print(rb.file_idx.files)
    print(rb.file_idx.num_states)
    batch = rb.get_next_batch()
    print(batch.states.shape)
    print(batch.values.shape)
    print(batch.dense_policy.policy.shape)
    # time.sleep(3)
    for file in os.listdir(f"{data_dir}/archive"):
        os.rename(
            f"{data_dir}/archive/{file}",
            f"{data_dir}/self_play/{file}",
        )
