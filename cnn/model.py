from curses import window
import flax.linen as nn
from flax.struct import dataclass
import jax.numpy as jnp
from collections.abc import Sequence


@dataclass
class ModelParams:
    num_outputs: int = 10
    features: Sequence[int] = (32, 64, 128)
    kernel_size: tuple[int, int] = (3, 3)
    mlp_dims: Sequence[int] = (256, 128)
    window_shape: tuple[int, int] = (3, 3)


class Model(nn.Module):
    params: ModelParams

    @nn.compact
    def __call__(self, x):
        for feature in self.params.features:
            x = nn.Conv(features=feature, kernel_size=self.params.kernel_size)(x)
            x = nn.max_pool(x, window_shape=self.params.window_shape)

        x = x.reshape((x.shape[0], -1))
        for dim in self.params.mlp_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.relu(x)

        return nn.Dense(features=self.params.num_outputs)(x)
