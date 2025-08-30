from model import ModelParams, Model
import optax
import jax
from flax.struct import dataclass
from flax.training.train_state import TrainState
import chex
from data import load_data


@dataclass
class HyperParams:
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 1
    seed: int = 9234578


def loss_fn(variables, x, y):
    pass


def train_step(train_state):
    pass


def eval_step():
    pass


def train(params: ModelParams, hyperparams: HyperParams):
    model = Model(params)
    rng = jax.random.PRNGKey(hyperparams.seed)

    variables = model.init(rng, jax.numpy.ones([1, 32, 32, 3]))
    tx = optax.adam(learning_rate=hyperparams.learning_rate)

    train_state = TrainState.create(apply_fn=model.apply, params=variables, tx=tx)
