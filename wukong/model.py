import flax.linen as nn
from flax.struct import dataclass
import jax.numpy as jnp
import jax
from collections.abc import Sequence
import chex


@dataclass
class WukongConfig:
    embedding_dim: int = 16


class EmbeddingLayer(nn.Module):
    vocab_sizes: Sequence[int]
    embedding_dim: int

    @nn.compact
    def __call__(self, x: chex.Array):
        pass


class MLP(nn.Module):
    hidden_layer_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for num_features in self.hidden_layer_dims:
            x = nn.Dense(num_features, kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.relu(x)

        return nn.Dense(self.output_dim, kernel_init=nn.initializers.lecun_normal())(x)


class LinearCompressBlock(nn.Module):
    num_embeddings: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            self.num_embeddings,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
        )(jnp.transpose(x, (0, 2, 1))).transpose((0, 2, 1))


class FactorizationMachine(nn.Module):
    num_compressed_embeddings: int

    @nn.compact
    def __call__(self, x):
        return jnp.matmul(
            x,
            LinearCompressBlock(self.num_compressed_embeddings)(x.transpose((0, 2, 1))),
        )


class FactorizationMachineBlock(nn.Module):
    num_embeddings: int
    embedding_dim: int
    num_compressed_embeddings: int
    mlp_hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: chex.Array):
        x = FactorizationMachine(self.num_compressed_embeddings)(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.LayerNorm(x)
        x = MLP(self.mlp_hidden_dims, self.num_embeddings * self.embedding_dim)(x)
        return x.reshape(-1, self.num_embeddings, self.embedding_dim)

class WukongLayer(nn.Module):

    @nn.compact
    def __call__(self, x):
        
