import flax.linen as nn
from flax.struct import dataclass
import jax.numpy as jnp
from numpy import stack


@dataclass
class DLRMConfig:
    num_embeddings: list[int]
    embedding_dim: int
    bottom_mlp_dims: list[int]
    top_mlp_dims: list[int]


class MLP(nn.Module):
    layer_dims: list[int]
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        for dim in self.layer_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.xavier_normal())(x)
            x = nn.relu(x)
        return nn.Dense(self.num_outputs, kernel_init=nn.initializers.xavier_normal())(
            x
        )


class EmbeddingLayer(nn.Module):
    num_embeddings: list[int]
    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        embeddings = []
        for i, num_embed in enumerate(self.num_embeddings):
            embed = nn.Embed(
                num_embed,
                self.embedding_dim,
                embedding_init=nn.initializers.glorot_uniform(),
            )(jnp.mod(x[:, i], num_embed))
            embeddings.append(embed)

        return jnp.stack(embeddings, axis=1)


class InteractionLayer(nn.Module):
    @nn.compact
    def __call__(self, dense_x, sparse_x):
        dense_x = jnp.expand_dims(dense_x, axis=1)
        x = jnp.concatenate((dense_x, sparse_x), axis=1)
        x = jnp.matmul(x, jnp.transpose(x, (0, 2, 1)))
        indices = jnp.triu_indices(x.shape[1], k=1)
        return x[:, indices[0], indices[1]]


class DLRM(nn.Module):
    config: DLRMConfig

    @nn.compact
    def __call__(self, dense_x, sparse_x):
        # Bottom MLP
        dense_x = MLP(self.config.bottom_mlp_dims, self.config.embedding_dim)(dense_x)

        sparse_x = EmbeddingLayer(
            self.config.num_embeddings, self.config.embedding_dim
        )(sparse_x)

        x = InteractionLayer()(dense_x, sparse_x)
        x = jnp.concatenate((x, dense_x), axis=1)

        # Top MLP
        x = MLP(self.config.top_mlp_dims, 1)(x)
        return x.squeeze()
