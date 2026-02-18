from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from python.data_reader import Batch


@nnx.jit
def create_input_repr(state_arr: ArrayLike) -> Array:
    # Takes a batch of state arrays and generates the appropriate input
    # representation.
    # For Tic-tac-toe, the input for a state is a 3x3 matrix for each player
    # representing each player's tokens.
    input_repr = jnp.reshape(state_arr, [2, 3, 3])
    input_repr = jnp.transpose(input_repr, (1, 2, 0))
    return input_repr


@nnx.jit
def create_batch_input(batch_state_arrs: ArrayLike) -> Array:
    return vmap(create_input_repr, in_axes=0)(batch_state_arrs)


@nnx.jit
def create_padding() -> Array:
    # Creates padding for partially filled batches
    return jnp.zeros((2, 9))


def preprocess_batch(batch: Batch) -> tuple[Array, Array, Array, Array]:
    batch_states = jnp.asarray(batch.states)
    batch_values = jnp.asarray(batch.values)
    if batch.dense_policy is not None:
        batch_masks = jnp.asarray(batch.dense_policy.mask)
        batch_policies = jnp.asarray(batch.dense_policy.policy)
    elif batch.sparse_policy is not None:
        batch_masks, batch_policies = from_sparse_policy_numpy(
            batch.sparse_policy.actions, batch.sparse_policy.weights
        )
    else:
        batch_masks = batch_policies = jnp.zeros((batch_states.shape[0], 9))
    return batch_states, batch_values, batch_masks, batch_policies


def to_sparse_policy(policy: Array) -> tuple[Array, Array]:
    actions = jnp.nonzero(policy)[0]
    weights = policy[actions]

    return actions, weights


def to_sparse_policy_numpy(policy: Array) -> tuple[np.ndarray, np.ndarray]:
    actions, weights = to_sparse_policy(policy)
    return (np.asarray(actions, dtype=np.uint8), np.asarray(weights))


def from_sparse_policy(actions: Array, weights: Array):
    unnormed_policy = np.zeros(9)
    unnormed_policy[actions] = weights
    mask = np.zeros(9)
    mask[actions] = 1
    policy = unnormed_policy / np.sum(unnormed_policy)
    return jnp.array(mask), jnp.array(policy)


def from_sparse_policy_numpy(
    actions: np.ndarray, weights: np.ndarray
) -> tuple[Array, Array]:
    mask, policy = vmap(from_sparse_policy)(jnp.array(actions), jnp.array(weights))
    return mask, policy


class CNN(nnx.Module):
    def __init__(self, seed: int):
        rngs = nnx.Rngs(seed)
        self.conv1 = nnx.Conv(2, 32, kernel_size=(3, 3), rngs=rngs, padding="SAME")
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs, padding="SAME")
        self.linear1 = nnx.Linear(576, 128, rngs=rngs)
        self.value_head = nnx.Linear(128, 1, rngs=rngs)
        self.policy_head = nnx.Linear(128, 9, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        v = nnx.tanh(self.value_head(x))
        p = self.policy_head(x)
        return v, p


if __name__ == "__main__":
    import jax.random as rnd

    # key = rnd.key(0)
    # dummy_arr = rnd.normal(key, shape=(32, 2, 9))
    # print(dummy_arr.shape)
    # dummy_input = create_batch_input(dummy_arr)
    # print(dummy_input.shape)

    actions = np.array([1, 3, 4], dtype=np.uint8)
    weights = np.array([1, 1, 1])
    policy = from_sparse_policy_numpy(actions, weights)
    print(policy)
    actions_np, weights_np = to_sparse_policy_numpy(policy)
    print(actions_np, weights_np)
