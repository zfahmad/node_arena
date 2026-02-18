from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


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


class DataReader:
    def __init__(
        self, file_path: str, group: str, policy_type: str, batch_size: int
    ) -> None:
        self.file_path: str = file_path
        self.batch_size: int = batch_size
        self.file: h5py.File = h5py.File(self.file_path, "r")
        self.dataset: h5py.Group = self.file[group]  # type: ignore
        self.pos: int = 0
        self.size: int = self.dataset["states"].shape[0]  # type: ignore
        if policy_type not in ["dense", "sparse"]:
            raise ValueError("Unknown policy type")
        self.policy_type = policy_type

    def get_next_batch(self) -> Batch:
        # Check if the next batch exceeds the number of datapoints.
        # This data reader drops extra data points.
        if self.pos + self.batch_size > self.size:
            self.pos = 0

        idx: np.ndarray = np.arange(self.pos, self.pos + self.batch_size)
        self.pos += self.batch_size

        batch_states: np.ndarray = self.dataset["states"][idx]  # type: ignore
        batch_values: np.ndarray = self.dataset["values"][idx]  # type: ignore

        # Save dense policies for policies that are small.
        # Save sparse representation of only legal actions and weights when the
        # action space is too large.

        if self.policy_type == "dense":
            batch_masks: np.ndarray = self.dataset["masks"][idx]  # type: ignore
            batch_policies: np.ndarray = self.dataset["policies"][idx]  # type: ignore
            return Batch(
                states=batch_states,
                values=batch_values,
                dense_policy=DensePolicy(batch_masks, batch_policies),
            )
        else:
            batch_actions: np.ndarray = self.dataset["actions"][idx]  # type: ignore
            batch_weights: np.ndarray = self.dataset["weights"][idx]  # type: ignore
            return Batch(
                states=batch_states,
                values=batch_values,
                sparse_policy=SparsePolicy(batch_actions, batch_weights),
            )

    def close(self) -> None:
        self.file.close()


if __name__ == "__main__":
    file_path = "/Users/zaheen/projects/node_arena/python/ttt_dataset.h5"
    group = "training"
    policy_type = "dense"
    batch_size = 64
    dr = DataReader(file_path, group, policy_type, batch_size)
    print(dr.dataset.keys())  # type: ignore
    dr.get_next_batch()
    dr.get_next_batch()
    dr.close()
