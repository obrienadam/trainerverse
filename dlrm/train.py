from encodings.punycode import T
from math import inf
from random import seed
from flax.struct import dataclass
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import DLRMConfig, DLRM
import tensorflow as tf
from data import load_dataset
import jax
import data
import functools
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import jax_metrics
from absl import logging

jax.config.update("jax_enable_x64", True)


@dataclass
class TrainConfig:
    model_config: DLRMConfig
    dense_learning_rate: float = 0.01
    sparse_learning_rate: float = 0.1
    num_epochs: int = 1
    batch_size: int = 2048
    seed: int = 4753


class TrainState(train_state.TrainState):
    pass


def loss_fn(params, x_dense, x_sparse, labels, apply_fn, training):
    logits = apply_fn({"params": params}, x_dense, x_sparse)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)

    if training:
        return jnp.mean(loss)
    return loss, logits


grad_fn = jax.value_and_grad(functools.partial(loss_fn, training=True))


@jax.jit
def train_step(state, x_dense, x_sparse, labels):
    loss, grad = grad_fn(state.params, x_dense, x_sparse, labels, state.apply_fn)
    state = state.apply_gradients(grads=grad)
    return state, loss


@jax.jit
def eval_step(state, x_dense, x_sparse, labels, accuracy):
    loss, preds = loss_fn(
        state.params, x_dense, x_sparse, labels, state.apply_fn, training=False
    )
    accuracy = accuracy.update(
        preds=jnp.stack((1.0 - preds, preds), axis=1), target=labels
    )
    return loss, accuracy, accuracy.compute()


def train(config: TrainConfig):
    model = DLRM(config.model_config)
    key = jax.random.PRNGKey(config.seed)

    dummy_dense = jnp.zeros((1, 13), dtype=jnp.float32)
    dummy_sparse = jnp.zeros((1, 26), dtype=jnp.int64)
    params = model.init(key, dummy_dense, dummy_sparse)["params"]
    key, subkey = jax.random.split(key, num=2)

    dense_optimizer = optax.adam(config.dense_learning_rate)
    sparse_optimizer = optax.adagrad(config.sparse_learning_rate)

    dense_mask = jax.tree_util.tree_map_with_path(
        lambda path, _: "EmbeddingLayer" not in str(path), params
    )

    optimizer = optax.chain(
        optax.masked(dense_optimizer, dense_mask),
        optax.masked(
            sparse_optimizer, jax.tree_util.tree_map(lambda x: not x, dense_mask)
        ),
    )

    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    train_step(train_state, dummy_dense, dummy_sparse, jnp.ones((config.batch_size, 1)))

    ds_train, ds_test = load_dataset(
        config.batch_size, shuffle_seed=jax.random.randint(subkey, (), 0, 1e6)
    )

    accuracy = jax_metrics.metrics.Accuracy(multiclass=True, num_classes=2)

    loss = inf
    acc = 0.0
    with Progress(
        TextColumn("[bold blue]Epoch {task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        "{task.completed}/{task.total}",
        "•",
        TextColumn("loss={task.fields[loss]:.4f}"),
        "•",
        TextColumn("acc={task.fields[acc]:.4f}"),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        for epoch in range(config.num_epochs):
            task = progress.add_task(
                f"{epoch + 1}/{config.num_epochs}", total=800_000, loss=loss, acc=acc
            )

            for x_dense, x_sparse, labels in ds_train.as_numpy_iterator():
                train_state, loss = train_step(train_state, x_dense, x_sparse, labels)
                progress.update(task, advance=config.batch_size, loss=loss, acc=acc)

            task = progress.add_task(
                f"{epoch + 1}/{config.num_epochs}", total=200_000, loss=loss, acc=acc
            )
            for x_dense, x_sparse, labels in ds_test.as_numpy_iterator():
                progress.update(task, advance=config.batch_size, loss=loss, acc=acc)
                _, accuracy, acc = eval_step(
                    train_state, x_dense, x_sparse, labels, accuracy
                )

            accuracy.reset()
