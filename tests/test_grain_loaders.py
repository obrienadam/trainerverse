import numpy as np
from dlrm.data import load_dataset as load_dlrm_dataset
from wukong.data import get_dataset as load_wukong_dataset

def test_dlrm_grain_loader():
    batch_size = 16
    loader_train, _ = load_dlrm_dataset(batch_size, shuffle_seed=42)
    # Check if we can get a batch
    for batch in loader_train:
        dense, sparse, labels = batch
        assert dense.shape == (batch_size, 13)
        assert sparse.shape == (batch_size, 26)
        assert labels.shape == (batch_size,)
        assert dense.dtype == np.float32
        assert sparse.dtype == np.int32
        break

def test_wukong_grain_loader():
    batch_size = 32
    loader = load_wukong_dataset(batch_size, shuffle_seed=5342)
    for batch in loader:
        dense, sparse, labels = batch
        assert dense.shape == (batch_size, 13)
        assert sparse.shape == (batch_size, 26)
        assert labels.shape == (batch_size,)
        assert dense.dtype == np.float32
        assert sparse.dtype == np.uint32
        break
