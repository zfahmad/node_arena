import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import h5py
import numpy as np
import numpy.random as rnd

from python.configs import Batch, DensePolicy, SparsePolicy

# TODO: Update read_file to handle sparse policies


@dataclass
class FileIndex:
    files: deque[str] = field(default_factory=deque)
    num_states: deque[int] = field(default_factory=deque)


class ReplayBuffer:
    def __init__(
        self,
        seed: int,
        data_dir: str,
        batch_size: int,
        buffer_size: int,
    ) -> None:
        self.rand = rnd.default_rng(seed)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.file_idx = FileIndex()
        self._lock = threading.Lock()

        self.self_play_dir = os.path.join(self.data_dir, "self_play")
        self.archive_dir = os.path.join(self.data_dir, "archive")

        os.makedirs(self.self_play_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

    def _snapshot_index(self) -> Tuple[list[str], np.ndarray]:
        """
        Copy index under lock so training thread works on a stable snapshot.
        """
        with self._lock:
            files = list(self.file_idx.files)
            counts = np.array(self.file_idx.num_states, dtype=np.int64)

        return files, counts

    def _read_file(
        self,
        file_path: str,
        file_size: int,
        num_tuples: int,
    ) -> Tuple[np.ndarray, ...]:

        num_tuples = min(num_tuples, file_size)
        idx = np.sort(self.rand.choice(file_size, size=num_tuples, replace=False))

        with h5py.File(file_path, "r") as f:
            states = f["states"][idx]  # type: ignore
            masks = f["masks"][idx]  # type: ignore
            policies = f["policies"][idx]  # type: ignore
            values = f["values"][idx]  # type: ignore

        return states, masks, policies, values  # type: ignore

    def get_next_batch(self) -> Batch:
        files, counts = self._snapshot_index()

        if len(files) == 0:
            raise RuntimeError("ReplayBuffer is empty.")

        total = counts.sum()
        if total == 0:
            raise RuntimeError("ReplayBuffer contains zero states.")

        weights = counts / total

        sampled_file_idxs = np.sort(
            self.rand.choice(
                np.arange(len(files)),
                size=self.batch_size,
                p=weights,
            )
        )

        file_sample_counts = Counter(sampled_file_idxs)

        batch_states = []
        batch_masks = []
        batch_policies = []
        batch_values = []

        for file_idx, count in file_sample_counts.items():
            file_path = files[file_idx]
            file_size = int(counts[file_idx])

            states, masks, policies, values = self._read_file(
                file_path,
                file_size,
                count,
            )

            batch_states.append(states)
            batch_masks.append(masks)
            batch_policies.append(policies)
            batch_values.append(values)

        return Batch(
            states=np.concatenate(batch_states, axis=0),
            values=np.concatenate(batch_values, axis=0),
            dense_policy=DensePolicy(
                mask=np.concatenate(batch_masks, axis=0),
                policy=np.concatenate(batch_policies, axis=0),
            ),
        )

    def add_to_index(self) -> None:
        """
        Detect new files in self_play, add to index, enforce sliding window,
        then atomically move to archive.
        """

        files = sorted(os.listdir(self.self_play_dir))

        for file in files:
            file_path = os.path.join(self.self_play_dir, file)

            # Skip temporary / partially written files
            if not file.endswith(".h5"):
                continue

            # Check already indexed
            with self._lock:
                if file_path in self.file_idx.files:
                    continue

            # Read metadata outside lock
            try:
                with h5py.File(file_path, "r") as f:
                    num_states = f["states"].shape[0]  # type: ignore
            except OSError:
                # File may still be being written; skip this round
                continue

            archived_path = os.path.join(self.archive_dir, file)

            # Atomic move first (so actors can't touch it anymore)
            os.rename(file_path, archived_path)

            # Now mutate index under lock
            with self._lock:
                while len(self.file_idx.files) >= self.buffer_size:
                    self.file_idx.files.popleft()
                    self.file_idx.num_states.popleft()

                self.file_idx.files.append(archived_path)
                self.file_idx.num_states.append(num_states)

    def start_indexing_thread(self, interval_seconds: float = 5.0):
        def loop():
            while True:
                self.add_to_index()
                time.sleep(interval_seconds)

        t = threading.Thread(target=loop, daemon=True)
        t.start()


if __name__ == "__main__":
    data_dir: str = "/Users/zaheen/Documents/node_arena/az_test"
    rb = ReplayBuffer(0, data_dir, 32, 100)
    rb.start_indexing_thread()
    print(rb.file_idx.files)
    # print(rb.file_idx.num_states)
    time.sleep(6)
    for _ in range(5):
        batch = rb.get_next_batch()
        print(batch.states.shape)
        print(batch.values.shape)
        print(batch.dense_policy.policy.shape)
    for file in os.listdir(f"{data_dir}/archive"):
        os.rename(
            f"{data_dir}/archive/{file}",
            f"{data_dir}/self_play/{file}",
        )
