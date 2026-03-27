import jax
import jax.numpy as jnp
import optax
from model import WukongConfig, Wukong
from data import get_dataloaders
from flax.struct import dataclass
from flax.training import train_state
import jax_metrics
from absl import logging
import numpy as np

@dataclass
class TrainConfig:
    model_config: WukongConfig
    vocab_sizes: list[int]
    learning_rate: float = 0.0001
    num_epochs: int = 5
    batch_size: int = 256
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
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, x_dense, x_sparse, labels):
    logits = state.apply_fn({"params": state.params}, x_dense, x_sparse)
    preds = (logits > 0.0).astype(jnp.int32)
    accuracy = jnp.mean(preds == labels)
    return accuracy

def train(config: TrainConfig):
    loader_train, loader_test = get_dataloaders(batch_size=config.batch_size, shuffle_seed=config.shuffle_seed)

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

    print(f"Starting Train/Test verification...")
    for epoch in range(config.num_epochs):
        epoch_losses = []
        for step, (x_dense, x_sparse, labels) in enumerate(loader_train):
            state, loss = train_step(state, x_dense, x_sparse, labels)
            epoch_losses.append(float(loss))
            
            if step % 20 == 0:
                acc_batch = eval_step(state, x_dense, x_sparse, labels)
                print(f"Epoch {epoch+1}, Step {step}, Train Loss: {loss:.4f}, Batch Acc: {acc_batch:.4f}")

        # Evaluation
        test_accuracies = []
        for x_dense, x_sparse, labels in loader_test:
            acc = eval_step(state, x_dense, x_sparse, labels)
            test_accuracies.append(float(acc))
        
        avg_test_acc = np.mean(test_accuracies)
        print(f"--- Epoch {epoch+1} Evaluation finished. Avg Test Accuracy: {avg_test_acc:.4f} ---")

    return state

if __name__ == "__main__":
    _VOCAB_SIZES = [2**16] * 26
    config = TrainConfig(
        model_config=WukongConfig(embedding_dim=8),
        vocab_sizes=_VOCAB_SIZES,
        num_epochs=5
    )
    train(config)
