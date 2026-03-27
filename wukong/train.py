import jax
import jax.numpy as jnp
import optax
from model import WukongConfig, Wukong
from data import get_dataset
from flax.struct import dataclass
from flax.training import train_state
import jax_metrics
from absl import logging

@dataclass
class TrainConfig:
    model_config: WukongConfig
    vocab_sizes: list[int]
    learning_rate: float = 0.001
    num_epochs: int = 1
    batch_size: int = 128
    seed: int = 4753
    shuffle_seed: int = 5342

class TrainState(train_state.TrainState):
    pass

def loss_fn(params, x_dense, x_sparse, labels, apply_fn):
    logits = apply_fn({"params": params}, x_dense, x_sparse)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels.astype(jnp.float32)).mean()
    return loss, logits

@jax.jit
def train_step(state, x_dense, x_sparse, labels):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, x_dense, x_sparse, labels, state.apply_fn)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(config: TrainConfig):
    loader = get_dataset(batch_size=config.batch_size, shuffle_seed=config.shuffle_seed)

    model = Wukong(config.model_config, vocab_sizes=config.vocab_sizes)

    rng = jax.random.PRNGKey(config.seed)
    dummy_dense = jnp.zeros((1, 13))
    dummy_sparse = jnp.zeros((1, 26), dtype=jnp.uint32)
    
    variables = model.init(rng, dummy_dense, dummy_sparse)
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adam(config.learning_rate)
    )

    for epoch in range(config.num_epochs):
        logging.info(f"Epoch {epoch+1}/{config.num_epochs}")
        for step, (x_dense, x_sparse, labels) in enumerate(loader):
            state, loss = train_step(state, x_dense, x_sparse, labels)
            if step % 10 == 0:
                logging.info(f"Step {step}, Loss: {loss:.4f}")

    return state

if __name__ == "__main__":
    _VOCAB_SIZES = [2**20] * 26 # Consistent with hashing
    config = TrainConfig(
        model_config=WukongConfig(),
        vocab_sizes=_VOCAB_SIZES
    )
    train(config)
