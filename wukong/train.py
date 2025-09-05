from scipy import sparse
from data import get_dataset
from model import WukongConfig, Wukong
import tensorflow as tf
from flax.struct import dataclass


@dataclass
class TrainConfig:
    model_config: WukongConfig
    dense_learning_rate: float = 0.001
    sparse_learning_rate: float = 0.01
    num_epochs: int = 1
    batch_size: int = 2048
    seed: int = 4753
    shuffle_seed: int = 5342

def train(config: TrainConfig):
    ds_train = get_dataset(batch_size=config.batch_size, shuffle_seed=config.shuffle_seed)

    model = Wukong(config.model_config)

    rng = jax.random.PRNGKey(config.seed)
    
