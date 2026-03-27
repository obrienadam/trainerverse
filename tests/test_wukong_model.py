import jax.numpy as jnp
import jax
from wukong.model import WukongConfig, Wukong

def test_wukong_model_output():
    vocab_sizes = [100] * 26
    embedding_dim = 16
    config = WukongConfig(embedding_dim=embedding_dim)
    
    model = Wukong(config, vocab_sizes=vocab_sizes)
    key = jax.random.PRNGKey(42)
    
    # 4 examples, 13 dense, 26 sparse
    x_dense = jnp.zeros((4, 13))
    x_sparse = jnp.zeros((4, 26), dtype=jnp.uint32)
    
    params = model.init(key, x_dense, x_sparse)["params"]
    output = model.apply({"params": params}, x_dense, x_sparse)
    
    # Verify final shape
    assert output.shape == (4,)
