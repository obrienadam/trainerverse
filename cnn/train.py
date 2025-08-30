from model import ModelParams, Model
import optax
import jax
from flax.struct import dataclass
from flax.training.train_state import TrainState
import chex
from data import load_data
import jax.numpy as jnp
from absl import logging
import jax_metrics


@dataclass
class HyperParams:
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 1
    seed: int = 9234578


def loss_fn(params, x, y, apply_fn):
    logits = apply_fn(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), logits


@jax.jit
def train_step(train_state, x, y):
    (loss, _), grads = jax.value_and_grad(loss_fn, allow_int=True, has_aux=True)(
        train_state.params, x, y, train_state.apply_fn
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


@jax.jit
def eval_step(train_state, x, y):
    loss, logits = train_state.apply_fn(train_state.params, x, y, train_state.apply_fn)
    return loss, logits


def train(
    params: ModelParams = ModelParams(), hyperparams: HyperParams = HyperParams()
):
    model = Model(params)
    rng = jax.random.PRNGKey(hyperparams.seed)

    variables = model.init(rng, jnp.ones([1, 32, 32, 3]))
    tx = optax.adam(learning_rate=hyperparams.learning_rate)

    train_state = TrainState.create(apply_fn=model.apply, params=variables, tx=tx)

    train_ds, test_ds, ds_info = load_data("cifar10")

    for epoch in range(hyperparams.num_epochs):
        logging.info("Starting epoch %d / %d.", epoch + 1, hyperparams.num_epochs)

        train_ds = train_ds.shuffle(1024).batch(hyperparams.batch_size)

        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            images, labels = batch
            train_state, loss = train_step(train_state, images, labels)
            logging.info("Epoch %d, step %d, loss: %.4f", epoch + 1, step + 1, loss)

        train_ds = train_ds.unbatch()

        accuracy = jax_metrics.metrics.Accuracy()
        for batch in test_ds.batch(hyperparams.batch_size).as_numpy_iterator():
            images, labels = batch
            loss, logits = eval_step(train_state, images, labels)
            accuracy = accuracy.update(preds=logits, target=labels)
            logging.info("Eval loss: %.4f", loss)
        logging.info("Eval accuracy: %.4f", accuracy.compute())
