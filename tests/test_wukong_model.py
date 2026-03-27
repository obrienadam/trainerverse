from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import jax
from wukong.model import WukongConfig, Wukong

class WukongModelTest(parameterized.TestCase):

    def test_wukong_output_shape(self):
        vocab_sizes = [100] * 26
        embedding_dim = 16
        config = WukongConfig(embedding_dim=embedding_dim)
        
        model = Wukong(config, vocab_sizes=vocab_sizes)
        key = jax.random.PRNGKey(42)
        
        batch_size = 4
        x_dense = jnp.zeros((batch_size, 13))
        x_sparse = jnp.zeros((batch_size, 26), dtype=jnp.uint32)
        
        variables = model.init(key, x_dense, x_sparse)
        params = variables["params"]
        output = model.apply({"params": params}, x_dense, x_sparse)
        
        self.assertEqual(output.shape, (batch_size,))


if __name__ == "__main__":
    absltest.main()
