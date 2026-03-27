from math import inf
from flax.struct import dataclass
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import DLRMConfig, DLRM
from data import load_dataset
import jax
import functools
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import jax_metrics
from absl import logging

# Disable x64 as it's typically not needed for DLRM and can be slower
jax.config.update("jax_enable_x64", False)


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
    # Ensure labels have correct shape for optax
    labels = labels.astype(jnp.float32)
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
    loss, logits = loss_fn(
        state.params, x_dense, x_sparse, labels, state.apply_fn, training=False
    )
    # Convert logits to probabilities
    probs = jax.nn.sigmoid(logits)
    # Stack probabilities for multiclass Accuracy metric (2 classes)
    preds_2d = jnp.stack((1.0 - probs, probs), axis=1)
    accuracy = accuracy.update(preds=preds_2d, target=labels)
    return jnp.mean(loss), accuracy


def train(config: TrainConfig):
    model = DLRM(config.model_config)
    key = jax.random.PRNGKey(config.seed)

    dummy_dense = jnp.zeros((1, 13), dtype=jnp.float32)
    dummy_sparse = jnp.zeros((1, 26), dtype=jnp.int32)
    params = model.init(key, dummy_dense, dummy_sparse)["params"]
    key, subkey = jax.random.split(key, num=2)

    dense_optimizer = optax.adam(config.dense_learning_rate)
    # Use optax.adam for sparse as well, or keep adagrad
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

    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    ds_train, ds_test = load_dataset(
        config.batch_size, shuffle_seed=jax.random.randint(subkey, (), 0, 1000000)
    )

    accuracy = jax_metrics.metrics.Accuracy(multiclass=True, num_classes=2)

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
            # Training Phase
            train_task = progress.add_task(
                f"Train {epoch + 1}/{config.num_epochs}", total=None, loss=inf, acc=0.0
            )

            for x_dense, x_sparse, labels in ds_train:
                state, loss = train_step(state, x_dense, x_sparse, labels)
                progress.update(train_task, advance=1, loss=float(loss))

            # Evaluation Phase
            eval_task = progress.add_task(
                f"Eval  {epoch + 1}/{config.num_epochs}", total=None, loss=inf, acc=0.0
            )
            
            epoch_loss = 0.0
            num_steps = 0
            for x_dense, x_sparse, labels in ds_test:
                loss, accuracy = eval_step(state, x_dense, x_sparse, labels, accuracy)
                epoch_loss += loss
                num_steps += 1
                progress.update(eval_task, advance=1, loss=float(epoch_loss / num_steps), acc=float(accuracy.compute()))

            accuracy.reset()
