from random import seed
from flax.training import train_state
from flax.struct import dataclass
import optax
import jax.numpy as jnp
import chex
import jax
from model import Resnet18, ModelConfig
from data import load_data
import tensorflow as tf
from absl import logging


@dataclass
class TrainConfig:
    learning_rate: float = 0.001
    num_epochs: int = 10
    batch_size: int = 128
    seed: int = 1425


class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree


def loss_fn(params, batch_stats, x, y, apply_fn, training):
    logits, new_variables = apply_fn(
        {"params": params, "batch_stats": batch_stats},
        x,
        training=training,
        mutable=["batch_stats"],
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, (new_variables, logits)


grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)


@jax.jit
def train_step(
    state: TrainState, x: chex.Array, y: chex.Array
) -> tuple[TrainState, float]:
    (loss, (new_variables, _)), grads = grad_fn(
        state.params, state.batch_stats, x, y, state.apply_fn, training=True
    )
    state = state.apply_gradients(grads=grads, batch_stats=new_variables["batch_stats"])
    return state, loss


@jax.jit
def eval_step(state: TrainState, x: chex.Array, y: chex.Array) -> tuple[float, float]:
    loss, (new_variables, logits) = loss_fn(
        state.params, state.batch_stats, x, y, state.apply_fn, training=False
    )
    accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
    return loss, accuracy


def train(train_config: TrainConfig):
    model = Resnet18(ModelConfig(num_output_classes=10))
    rng = jax.random.PRNGKey(train_config.seed)
    dummy_batch = jnp.ones((1, 32, 32, 3), jnp.uint8)
    variables = model.init(rng, dummy_batch, training=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=optax.adam(
            optax.cosine_decay_schedule(
                train_config.learning_rate, train_config.num_epochs * 50_000
            )
        ),
    )

    ds_train, ds_test, _ = load_data("cifar10")

    rng, _ = jax.random.split(rng)
    ds_train = (
        ds_train.shuffle(1_000_000, seed=rng[0])
        .batch(train_config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_test = ds_test.batch(train_config.batch_size).prefetch(tf.data.AUTOTUNE)

    for epoch in range(1, train_config.num_epochs + 1):
        logging.info(f"Starting epoch {epoch}/{train_config.num_epochs}.")
        for step, batch in enumerate(ds_train.as_numpy_iterator()):
            x, y = batch
            state, loss = train_step(state, x, y)
            logging.log_every_n(
                logging.INFO, f"Epoch {epoch} Step {step}, Loss: {loss:.4f}", 100
            )

        logging.info(f"Evaluating epoch {epoch}/{train_config.num_epochs}.")

        total_loss = 0.0
        total_correct_predictions = 0
        total_samples = 0
        for batch in ds_test.as_numpy_iterator():
            x, y = batch
            loss, accuracy = eval_step(state, x, y)
            num_samples_in_batch = y.shape[0]
            total_loss += loss * num_samples_in_batch
            total_correct_predictions += accuracy * num_samples_in_batch
            total_samples += num_samples_in_batch

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct_predictions / total_samples

        logging.info(
            f"Epoch {epoch} Final Evaluation, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
        )
