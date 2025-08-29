import flax.linen as nn
import flax
import jax
import jax.numpy as jnp
from absl import logging as log


@flax.struct.dataclass
class ModelConfig:
    kernel_size = (5, 5)
    filters = (32, 32, 64, 64)
    mlp_dims = (64,)
    window_size = (2, 2)


class Model(nn.Module):
    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(self, x: jax.Array):

        for filter_size in self.config.filters:
            x = nn.Conv(features=filter_size, kernel_size=self.config.kernel_size)(x)
            x = nn.max_pool(x, self.config.window_size)

        x = jnp.reshape(x, (x.shape[0], -1))
        for dims in self.config.mlp_dims:
            x = nn.Dense(dims)(x)
            x = nn.relu(x)

        x = nn.Dense(10)(x)
        return x


if __name__ == "__main__":
    m = Model(ModelConfig())
    input = jnp.ones((1, 32, 32, 1))

    key = jax.random.key(42)
    param_key, other_key = jax.random.split(key, 2)

    variables = m.init(param_key, input)

    print(m.apply(variables, input))
