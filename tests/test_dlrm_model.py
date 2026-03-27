from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import jax
from dlrm.model import DLRMConfig, DLRM

class DLRMModelTest(parameterized.TestCase):

    def test_stacked_table_offsets(self):
        num_embeddings = [10, 20, 30]
        embedding_dim = 16
        config = DLRMConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            bottom_mlp_dims=[32, 16],
            top_mlp_dims=[32, 16]
        )
        
        # Verify offsets: [0, 10, 30]
        expected_offsets = jnp.array([0, 10, 30])
        self.assertSequenceAlmostEqual(config.embedding_offsets, expected_offsets)

    def test_model_initialization_and_output_shape(self):
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
        
        batch_size = 4
        x_dense = jnp.zeros((batch_size, 13))
        x_sparse = jnp.zeros((batch_size, len(num_embeddings)), dtype=jnp.int32)
        
        variables = model.init(key, x_dense, x_sparse)
        params = variables["params"]
        
        # Total vocab should be sum of num_embeddings
        expected_vocab = sum(num_embeddings)
        embedding_table = params["EmbeddingLayer_0"]["embedding_table"]
        self.assertEqual(embedding_table.shape, (expected_vocab, embedding_dim))
        
        # Verify final output shape
        output = model.apply({"params": params}, x_dense, x_sparse)
        self.assertEqual(output.shape, (batch_size,))


if __name__ == "__main__":
    absltest.main()
