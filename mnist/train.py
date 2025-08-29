import model
from data import load_dataset, batch_generator
import jax
import optax
import jax.numpy as jnp
import jax_metrics
from flax.struct import dataclass
from absl import logging as log


@dataclass
class HyperParams:
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 1


def loss_fn(func, variables, x, labels):
    logits = func(variables, x)
    return (
        jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(logits, labels)
        ),
        logits,
    )


def train_step(func, variables, x, labels, update_fn, opt_state):
    (loss, _), grads = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(
        func, variables, x, labels
    )

    updates, opt_state = update_fn(grads, opt_state)
    variables = optax.apply_updates(variables, updates)

    return loss, variables, opt_state


train_step = jax.jit(train_step, static_argnames=("func", "update_fn"))


def eval_step(func, variables, x, labels):
    return loss_fn(func, variables, x, labels)


eval_step = jax.jit(eval_step, static_argnames=("func",))


def train(hparams: HyperParams = HyperParams()):
    df_train, df_test = load_dataset()

    m = model.Model()
    key = jax.random.key(2342)

    optimizer = optax.adam(learning_rate=hparams.learning_rate)
    variables = m.init(key, jnp.empty((1, 28, 28, 1)))
    opt_state = optimizer.init(variables)

    for epoch in range(hparams.num_epochs):
        log.info(f"Epoch {epoch+1}/{hparams.num_epochs}")
        log.info("Shuffling dataset...")
        df_train = df_train.sample(frac=1.0)

        for batch, labels in batch_generator(df_train, hparams.batch_size):
            loss, variables, opt_state = train_step(
                m.apply, variables, batch, labels, optimizer.update, opt_state
            )
            log.info(f"Loss: {loss:.4f}")

        accuracy = jax_metrics.metrics.Accuracy(num_classes=10)

        for batch, labels in batch_generator(df_test, hparams.batch_size):
            loss, logits = eval_step(m.apply, variables, batch, labels)
            accuracy = accuracy.update(preds=logits, target=labels)

        log.info(f"Test accuracy: {accuracy.compute():.4f}")


if __name__ == "__main__":
    train()
