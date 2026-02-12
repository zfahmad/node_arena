import h5py
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import Array, jit
from jax.typing import ArrayLike
from orbax import checkpoint as ckp

from python.models.tic_tac_toe_nn import CNN, create_batch_input


