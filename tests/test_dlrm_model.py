import jax.numpy as jnp
import jax
from dlrm.model import DLRMConfig, DLRM

def test_dlrm_stacked_table():
    num_embeddings = [10, 20, 30]
    embedding_dim = 16
    config = DLRMConfig(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        bottom_mlp_dims=[32, 16],
        top_mlp_dims=[32, 16]
    )
    
    model = DLRM(config)
    key = jax.random.PRNGKey(0)
    
    x_dense = jnp.zeros((4, 13))
    x_sparse = jnp.zeros((4, len(num_embeddings)), dtype=jnp.int32)
    
    params = model.init(key, x_dense, x_sparse)["params"]
    
    # Check if we have one single embedding table in the params
    embedding_table = params["EmbeddingLayer_0"]["embedding_table"]
    
    # Total vocab should be sum of num_embeddings
    expected_vocab = sum(num_embeddings)
    assert embedding_table.shape == (expected_vocab, embedding_dim)
    
    # Check if offsets work correctly
    expected_offsets = [0, 10, 30]
    assert jnp.all(config.embedding_offsets == jnp.array(expected_offsets))

    # Verify output shape
    output = model.apply({"params": params}, x_dense, x_sparse)
    assert output.shape == (4,)
