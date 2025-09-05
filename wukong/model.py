import flax.linen as nn
from flax.struct import dataclass
import jax.numpy as jnp

@dataclass
class WukongConfig:
    embedding_dim: in
