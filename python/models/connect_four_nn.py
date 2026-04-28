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


class ResidualBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(32, rngs=rngs)
        self.conv1 = nnx.Conv(32, 32, kernel_size=(1, 1), rngs=rngs, padding="SAME")
        self.ln2 = nnx.LayerNorm(32, rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs, padding="SAME")
        self.ln3 = nnx.LayerNorm(64, rngs=rngs)
        self.conv3 = nnx.Conv(64, 32, kernel_size=(1, 1), rngs=rngs, padding="SAME")
        self.ln_out = nnx.LayerNorm(32, rngs=rngs)

    def __call__(self, input_):
        x = input_
        x = self.conv1(x)
        x = nnx.relu(x)

        # x = self.ln2(x)
        x = self.conv2(x)
        x = nnx.relu(x)

        # x = self.ln3(x)
        x = self.conv3(x)
        x = nnx.relu(x)

        y = input_ + x

        return y


class ValueHead(nnx.Module):
    def __init__(self, in_features: int, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(in_features, rngs=rngs)
        self.linear1 = nnx.Linear(in_features, 128, rngs=rngs)
        self.ln2 = nnx.LayerNorm(128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 64, rngs=rngs)
        self.ln3 = nnx.LayerNorm(64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, input_):
        x = input_.reshape(input_.shape[0], -1)
        x = self.linear1(nnx.relu(x))
        x = self.linear2(nnx.relu(x))
        v = nnx.tanh(self.linear3(x))
        return v


class PolicyHead(nnx.Module):
    def __init__(self, in_features: int, policy_len: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(8, 8, (1, 1), rngs=rngs)
        self.ln1 = nnx.LayerNorm(in_features, rngs=rngs)
        self.linear1 = nnx.Linear(in_features, 128, rngs=rngs)
        self.ln2 = nnx.LayerNorm(128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 64, rngs=rngs)
        self.ln3 = nnx.LayerNorm(64, rngs=rngs)
        self.linear3 = nnx.Linear(64, policy_len, rngs=rngs)

    def __call__(self, input_):
        x = self.conv(input_)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(nnx.relu(x))
        x = self.linear2(nnx.relu(x))
        p = self.linear3(x)
        return p


class ResNet(nnx.Module):
    def __init__(self, seed: int, dims: list[int], num_blocks: int):
        rngs = nnx.Rngs(seed)
        # self.policy_len = dims[0] * dims[1]
        self.ln1 = nnx.LayerNorm(2, rngs=rngs)
        self.conv1 = nnx.Conv(2, 32, kernel_size=(3, 3), rngs=rngs, padding="SAME")
        self.ln2 = nnx.LayerNorm(32, rngs=rngs)
        self.conv2 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs, padding="SAME")
        self.blocks = nnx.List([ResidualBlock(rngs) for _ in range(num_blocks)])
        self.ch_red = nnx.Conv(32, 8, kernel_size=(1, 1), rngs=rngs, padding="SAME")
        in_features = dims[0] * dims[1] * 8
        self.policy_head = PolicyHead(in_features, dims[1], rngs)
        self.value_head = ValueHead(in_features, rngs)

    def __call__(self, input_):
        x = nnx.relu(input_)
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)

        for res_block in self.blocks:
            x = res_block(x)

        x = self.ch_red(x)
        p = self.policy_head(x)
        v = self.value_head(x)

        return v, p
