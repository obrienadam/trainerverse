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
        # x is (batch, num_features)
        total_vocab = sum(self.vocab_sizes)
        offsets = jnp.array([0] + list(jnp.cumsum(jnp.array(self.vocab_sizes))[:-1]))
        
        embedding_table = self.param(
            "embedding_table",
            nn.initializers.truncated_normal(stddev=1.0 / jnp.sqrt(total_vocab)),
            (total_vocab, self.embedding_dim)
        )
        
        indices = x + offsets[None, :]
        return jnp.take(embedding_table, indices, axis=0)


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
    num_embeddings: int
    embedding_dim: int
    num_compressed_embeddings: int
    mlp_hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # x: (batch, num_embed, embed_dim)
        res = x
        x = FactorizationMachineBlock(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            num_compressed_embeddings=self.num_compressed_embeddings,
            mlp_hidden_dims=self.mlp_hidden_dims
        )(x)
        return x + res


class Wukong(nn.Module):
    config: WukongConfig
    vocab_sizes: Sequence[int]

    @nn.compact
    def __call__(self, dense_x, sparse_x):
        # Dense processing
        dense_x = MLP([64, self.config.embedding_dim], self.config.embedding_dim)(dense_x)
        dense_x = jnp.expand_dims(dense_x, axis=1) # (B, 1, D)

        # Sparse processing
        sparse_x = EmbeddingLayer(self.vocab_sizes, self.config.embedding_dim)(sparse_x) # (B, 26, D)

        # Combine
        x = jnp.concatenate([dense_x, sparse_x], axis=1) # (B, 27, D)

        # Wukong interaction layers
        for _ in range(3):
            x = WukongLayer(
                num_embeddings=x.shape[1],
                embedding_dim=self.config.embedding_dim,
                num_compressed_embeddings=16,
                mlp_hidden_dims=[128, 128]
            )(x)

        # Final MLP
        x = x.reshape(x.shape[0], -1)
        x = MLP([256, 128], 1)(x)
        return x.squeeze(-1)
