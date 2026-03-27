from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from dlrm.data import load_dataset as load_dlrm_dataset
from wukong.data import get_dataset as load_wukong_dataset

class GrainLoaderTest(parameterized.TestCase):

    def test_dlrm_grain_loader_batch_yields_correct_shapes(self):
        batch_size = 16
        loader_train, _ = load_dlrm_dataset(batch_size, shuffle_seed=42)
        
        # Check first batch
        for batch in loader_train:
            dense, sparse, labels = batch
            self.assertEqual(dense.shape, (batch_size, 13))
            self.assertEqual(sparse.shape, (batch_size, 26))
            self.assertEqual(labels.shape, (batch_size,))
            self.assertEqual(dense.dtype, np.float32)
            break

    def test_wukong_grain_loader_batch_yields_correct_shapes(self):
        batch_size = 32
        loader = load_wukong_dataset(batch_size, shuffle_seed=5342)
        
        # Check first batch
        for batch in loader:
            dense, sparse, labels = batch
            self.assertEqual(dense.shape, (batch_size, 13))
            self.assertEqual(sparse.shape, (batch_size, 26))
            self.assertEqual(labels.shape, (batch_size,))
            self.assertEqual(dense.dtype, np.float32)
            self.assertEqual(sparse.dtype, np.uint32)
            break


if __name__ == "__main__":
    absltest.main()
