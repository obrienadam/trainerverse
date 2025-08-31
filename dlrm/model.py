import flax.linen as nn
from flax.struct import dataclass
import jax.numpy as jnp


@dataclass
class DLRMConfig:
    num_dense_features: int
    num_sparse_features: int
    embedding_dim: int
    bottom_mlp_dims: list[int]
    top_mlp_dims: list[int]


class MLP(nn.Module):
    layer_dims: list[int]
    num_outputs: int

    @nn.compact
    def __call__(self, x, sigmoid_final: bool = False):
        for dim in self.layer_dims:
            x = nn.Dense(dim, initializer=nn.initializers.xavier_normal())(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_outputs, initializer=nn.initializers.xavier_normal())(x)
        if sigmoid_final:
            return nn.sigmoid(x)
        return x


class InteractionLayer(nn.Module):
    @nn.compact
    def __call__(self, dense_x, sparse_x):
        dense_x = jnp.expand_dims(dense_x, axis=1)
        x = jnp.concatenate((dense_x, sparse_x), axis=1)
        x = jnp.matmul(x, jnp.transpose(x, (0, 2, 1)))

        return x.reshape((x.shape[0], -1))


class DLRM(nn.Module):
    config: DLRMConfig

    @nn.compact
    def __call__(self, dense_x, sparse_x):
        # Bottom MLP
        dense_x = MLP(self.config.bottom_mlp_dims, self.config.embedding_dim)(dense_x)
