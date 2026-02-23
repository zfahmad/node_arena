from functools import partial

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array, vmap
from jax.typing import ArrayLike

from python.data_reader import Batch


@partial(nnx.jit, static_argnums=(1,))
def create_input_repr(state_arr: ArrayLike, dims: list[int]) -> Array:
    # Takes a batch of state arrays and generates the appropriate input
    # representation.
    # For Connect Four, the input for a state is a matrix for each player
    # representing each player's tokens.
    input_repr = jnp.reshape(state_arr, [2, dims[0], dims[1]])
    input_repr = jnp.transpose(input_repr, (1, 2, 0))
    return input_repr


@partial(nnx.jit, static_argnums=(1,))
def create_batch_input(batch_state_arrs: ArrayLike, dims: list[int]) -> Array:
    return vmap(create_input_repr, in_axes=(0, None))(batch_state_arrs, dims)


def create_padding(dims: list[int]) -> Array:
    # Creates padding for partially filled batches
    return jnp.zeros((2, dims[0] * dims[1]))


def preprocess_batch(
    batch: Batch, dims: list[int]
) -> tuple[Array, Array, Array, Array]:
    # Receives a batch of data read by DataReader
    # It returns:
    #   - a state array representation
    #   - a target value
    #   - a policy mask
    #   - a target policy
    batch_states = jnp.asarray(batch.states)
    batch_values = jnp.asarray(batch.values)
    if batch.dense_policy is not None:
        batch_masks = jnp.asarray(batch.dense_policy.mask)
        batch_policies = jnp.asarray(batch.dense_policy.policy)
    elif batch.sparse_policy is not None:
        batch_masks, batch_policies = from_sparse_policy_numpy(
            batch.sparse_policy.actions, batch.sparse_policy.weights, dims
        )
    else:
        batch_masks = batch_policies = jnp.zeros((batch_states.shape[0], dims[1]))
    return batch_states, batch_values, batch_masks, batch_policies


def to_sparse_policy(policy: Array) -> tuple[Array, Array]:
    actions = jnp.nonzero(policy)[0]
    weights = policy[actions]

    return actions, weights


def to_sparse_policy_numpy(policy: Array) -> tuple[np.ndarray, np.ndarray]:
    actions, weights = to_sparse_policy(policy)
    return (np.asarray(actions, dtype=np.uint8), np.asarray(weights))


def from_sparse_policy(actions: Array, weights: Array, dims: list[int]):
    policy_len = dims[1]
    unnormed_policy = np.zeros(policy_len)
    unnormed_policy[actions] = weights
    mask = np.zeros(policy_len)
    mask[actions] = 1
    policy = unnormed_policy / np.sum(unnormed_policy)
    return jnp.array(mask), jnp.array(policy)


def from_sparse_policy_numpy(
    actions: np.ndarray, weights: np.ndarray, dims: list[int]
) -> tuple[Array, Array]:
    mask, policy = vmap(from_sparse_policy, in_axes=(0, 0, None))(
        jnp.array(actions), jnp.array(weights), dims
    )
    return mask, policy


class CNN(nnx.Module):
    def __init__(self, seed: int, dims: list[int]):
        rngs = nnx.Rngs(seed)
        self.conv1 = nnx.Conv(2, 32, kernel_size=(3, 3), rngs=rngs, padding="SAME")
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs, padding="SAME")
        self.linear1 = nnx.Linear(1024, 128, rngs=rngs)
        self.value_head = nnx.Linear(128, 1, rngs=rngs)
        self.policy_len = dims[1]
        self.policy_head = nnx.Linear(128, self.policy_len, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        v = nnx.tanh(self.value_head(x))
        p = self.policy_head(x)
        return v, p
