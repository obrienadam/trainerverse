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

    @property
    def total_num_embeddings(self) -> int:
        return sum(self.num_embeddings)

    @property
    def embedding_offsets(self) -> jnp.ndarray:
        return jnp.array([0] + list(jnp.cumsum(jnp.array(self.num_embeddings))[:-1]))


class MLP(nn.Module):
    layer_dims: list[int]
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        for dim in self.layer_dims:
            # He normal is generally better for ReLU
            x = nn.Dense(dim, kernel_init=nn.initializers.he_normal())(x)
            x = nn.relu(x)
        return nn.Dense(self.num_outputs, kernel_init=nn.initializers.he_normal())(
            x
        )


class EmbeddingLayer(nn.Module):
    num_embeddings: list[int]
    embedding_dim: int
    offsets: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        # x shape: (B, F) where F=26
        # Total size of the combined embedding table
        total_vocab = sum(self.num_embeddings)
        
        # Create a single large embedding table
        embedding_table = self.param(
            "embedding_table",
            nn.initializers.truncated_normal(stddev=1.0 / jnp.sqrt(total_vocab)),
            (total_vocab, self.embedding_dim)
        )

        # Offset each feature's index to point to its respective section in the table
        # x is (batch, num_features), offsets is (num_features,)
        # result: (batch, num_features)
        indices = x + self.offsets[None, :]

        # Perform one single gather operation for all features
        # result shape: (batch, num_features, embedding_dim)
        return jnp.take(embedding_table, indices, axis=0)


class InteractionLayer(nn.Module):
    @nn.compact
    def __call__(self, dense_x, sparse_x):
        dense_x = jnp.expand_dims(dense_x, axis=1)
        x = jnp.concatenate((dense_x, sparse_x), axis=1)
        # Dot product interaction
        x = jnp.matmul(x, jnp.transpose(x, (0, 2, 1)))
        indices = jnp.triu_indices(x.shape[1], k=1)
        return x[:, indices[0], indices[1]]


class DLRM(nn.Module):
    config: DLRMConfig

    @nn.compact
    def __call__(self, dense_x, sparse_x):
        # Bottom MLP maps dense features to the embedding dimension
        dense_x = MLP(self.config.bottom_mlp_dims, self.config.embedding_dim)(dense_x)

        # Sparse features use the stacked embedding table fix
        sparse_x = EmbeddingLayer(
            self.config.num_embeddings, 
            self.config.embedding_dim,
            self.config.embedding_offsets
        )(sparse_x)

        # Pairwise dot-product interations
        x = InteractionLayer()(dense_x, sparse_x)
        # Concatenate interaction results with the original dense features
        x = jnp.concatenate((x, dense_x), axis=1)

        # Top MLP for final classification
        x = MLP(self.config.top_mlp_dims, 1)(x)
        return x.squeeze(-1)
