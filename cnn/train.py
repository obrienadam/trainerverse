from model import ModelParams, Model
import optax
import jax
from flax.struct import dataclass
from flax.training import train_state
import chex
from data import load_data
import jax.numpy as jnp
from absl import logging
import jax_metrics
import tensorflow as tf


@dataclass
class HyperParams:
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 1
    seed: int = 9234578


class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree


def loss_fn(params, batch_stats, x, y, apply_fn):
    logits, new_variables = apply_fn(
        {"params": params, "batch_stats": batch_stats},
        x,
        training=True,
        mutable=["batch_stats"],
    )
    return (
        optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(),
        (logits, new_variables),
    )


@jax.jit
def train_step(train_state, x, y):
    (loss, (_, new_variables)), grads = jax.value_and_grad(
        loss_fn, allow_int=True, has_aux=True
    )(train_state.params, train_state.batch_stats, x, y, train_state.apply_fn)
    train_state = train_state.apply_gradients(
        grads=grads, batch_stats=new_variables["batch_stats"]
    )
    return train_state, loss


@jax.jit
def eval_step(train_state, x, y):
    loss, (logits, _) = loss_fn(
        train_state.params, train_state.batch_stats, x, y, train_state.apply_fn
    )
    return loss, logits


def train(
    params: ModelParams = ModelParams(), hyperparams: HyperParams = HyperParams()
):
    model = Model(params)
    rng = jax.random.PRNGKey(hyperparams.seed)

    variables = model.init(rng, jnp.ones([1, 32, 32, 3]))
    tx = optax.adam(learning_rate=hyperparams.learning_rate)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
    )

    train_ds, test_ds, ds_info = load_data("cifar10")
    train_ds = (
        train_ds.shuffle(1_000_000)
        .batch(hyperparams.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(hyperparams.batch_size).prefetch(tf.data.AUTOTUNE)

    for epoch in range(hyperparams.num_epochs):
        logging.info("Starting epoch %d / %d.", epoch + 1, hyperparams.num_epochs)

        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            images, labels = batch
            train_state, loss = train_step(train_state, images, labels)
            logging.info("Epoch %d, step %d, loss: %.4f", epoch + 1, step + 1, loss)

        accuracy = jax_metrics.metrics.Accuracy()
        for batch in test_ds.as_numpy_iterator():
            images, labels = batch
            loss, logits = eval_step(train_state, images, labels)
            accuracy = accuracy.update(preds=logits, target=labels)
            logging.info("Eval loss: %.4f", loss)
        logging.info("Eval accuracy: %.4f", accuracy.compute())
