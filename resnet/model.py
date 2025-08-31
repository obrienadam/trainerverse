import flax.linen as nn
from flax.struct import dataclass
import jax.numpy as jnp
import jax


@dataclass
class ModelConfig:
    num_output_classes: int = 10


class ResidualBlock(nn.Module):
    reduce_dimensionality: bool

    @nn.compact
    def __call__(self, x, training: bool):
        num_channels = x.shape[-1] * (2 if self.reduce_dimensionality else 1)
        y = nn.Conv(
            num_channels,
            kernel_size=(3, 3),
            strides=(2, 2) if self.reduce_dimensionality else (1, 1),
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        y = nn.Conv(
            num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(y)
        y = nn.BatchNorm(use_running_average=not training)(y)

        if self.reduce_dimensionality:
            x = nn.Conv(
                num_channels,
                kernel_size=(1, 1),
                strides=(2, 2),
                padding="SAME",
                kernel_init=nn.initializers.he_normal(),
            )(x)

        return nn.relu(x + y)


class Resnet18(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, training: bool):
        x = x.astype(jnp.float32) / 255.0  # Normalize input to [0, 1]
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        for reduce_dimensionality in [False, False, True, False, True, False]:
            x = ResidualBlock(reduce_dimensionality=reduce_dimensionality)(x, training)

        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2])).squeeze()
        return nn.Dense(
            features=self.config.num_output_classes,
            kernel_init=nn.initializers.xavier_normal(),
        )(x)
