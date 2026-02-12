import h5py
import numpy as np


class DataReader:
    def __init__(self, file_path: str, group: str, policy_type: str, batch_size: int) -> None:
        self.file_path: str = file_path
        self.batch_size: int = batch_size
        self.file: h5py.File = h5py.File(self.file_path, "r")
        self.dataset: h5py.Group = self.file[group]  # type: ignore
        self.pos: int = 0
        self.size: int = self.dataset["states"].shape[0]  # type: ignore
        if policy_type not in ["dense", "sparse"]:
            raise ValueError("Unknown policy type")
        self.policy_type = policy_type

    def get_next_batch(self) -> tuple[np.ndarray, ...]:
        if self.pos + self.batch_size > self.size:
            self.pos = 0

        idx = np.arange(self.pos, self.pos + self.batch_size)
        self.pos += self.batch_size

        batch_states: np.ndarray = self.dataset["states"][idx]  # type: ignore
        batch_values: np.ndarray = self.dataset["values"][idx]  # type: ignore

        if self.policy_type == "dense":
            batch_policies: np.ndarray = self.dataset["policies"][idx]  # type: ignore
            batch = (batch_states, batch_values, batch_policies,)
        else:
            batch_actions: np.ndarray = self.dataset["actions"][idx]  # type: ignore
            batch_weights: np.ndarray = self.dataset["weights"][idx]  # type: ignore
            batch = (batch_states, batch_values, batch_actions, batch_weights,)

        return batch

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
