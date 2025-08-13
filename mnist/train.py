import model
from data import load_dataset, batch_generator
import jax
import optax
import jax.numpy as jnp
import jax_metrics

def loss_fn(func, variables, x, labels):
    logits = func(variables, x)
    return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, labels)), logits

def train_step(func, variables, x, labels, update_fn, opt_state):
    (loss, _), grads = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(func, variables, x, labels)

    updates, opt_state = update_fn(grads, opt_state)
    variables = optax.apply_updates(variables, updates)

    return loss, variables, opt_state

train_step = jax.jit(train_step, static_argnames=('func', 'update_fn'))

def eval_step(func, variables, x, labels):
    return loss_fn(func, variables, x, labels)

eval_step = jax.jit(eval_step, static_argnames=('func',))

def train():
    df_train, df_test = load_dataset()
    data_itr = batch_generator(df_train, 64)

    m = model.Model()
    key = jax.random.key(2342)

    optimizer = optax.adam(learning_rate=1e-3)
    variables = m.init(key, jnp.empty((1, 28, 28, 1)))
    opt_state = optimizer.init(variables)

    for batch, labels in data_itr:
        loss, variables, opt_state = train_step(m.apply, variables, batch, labels, optimizer.update, opt_state)
        print(loss)

    metrics = jax_metrics.metrics.Accuracy(num_classes=10)

    for batch, labels in batch_generator(df_test, 64):
        loss, logits = eval_step(m.apply, variables, batch, logits)
        metrics.update(preds=logits, target=labels)

    print(metrics.compute())


if __name__ == "__main__":
    train()



