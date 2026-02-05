import h5py
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import Array, jit
from jax.typing import ArrayLike
from orbax import checkpoint as ckp

from python.models.tic_tac_toe_nn import CNN, create_batch_input


def loss_fn(model: nnx.Module, batch_inputs: ArrayLike, batch_labels: ArrayLike):
    batch = create_batch_input(batch_inputs)
    targets = jnp.asarray(batch_labels)
    predicted_values = model(jnp.asarray(batch))
    loss = jnp.mean(optax.squared_error(predicted_values, targets))
    return loss


@jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_inputs: ArrayLike,
    batch_labels: ArrayLike,
):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch_inputs, batch_labels)
    optimizer.update(model, grads)

    return model, loss


def main():
    print("main")
    dataset_file = "./data/tic_tac_toe_dataset.h5"
    f = h5py.File(dataset_file, "r")
    training_set = f["training"]
    if not isinstance(training_set, h5py.Group):
        raise TypeError("training is not a group")
    testing_set = f["testing"]
    if not isinstance(testing_set, h5py.Group):
        raise TypeError("testing is not a group")

    training_inputs = np.array(training_set["states"])
    training_labels = np.reshape(np.array(training_set["values"]), (-1, 1))
    testing_inputs = np.array(testing_set["states"])
    testing_labels = np.reshape(np.array(testing_set["values"]), (-1, 1))
    print(training_inputs.shape)
    print(training_labels.shape)

    mini_batch_size = 64
    num_epochs = 5
    num_iterations = training_inputs.shape[0] // mini_batch_size
    learning_rate = 0.1
    momentum = 0.99

    model = CNN(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate, momentum), wrt=nnx.Param)

    for j in range(num_epochs):
        for i in range(num_iterations):
            start_ind = i * mini_batch_size
            end_ind = (i * mini_batch_size) + mini_batch_size
            model, loss = train_step(
                model,
                optimizer,
                jnp.asarray(training_inputs[start_ind:end_ind,:], dtype=jnp.float32),
                jnp.asarray(training_labels[start_ind:end_ind,:], dtype=jnp.float32)
            )
            print(j, i, loss)

    _, state = nnx.split(model)
    ckpt_dir = ckp.test_utils.erase_and_create_empty('/Users/zaheen/projects/node_arena/python/checkpoints/')
    ckpr = ckp.StandardCheckpointer()
    ckpr.save(ckpt_dir / 'state', state)
    ckpr.wait_until_finished()


if __name__ == "__main__":
    main()
