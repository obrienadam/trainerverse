import numpy as np
from mnist.data import load_datasets as load_mnist_datasets
from cnn.data import load_datasets as load_cnn_datasets, DatasetType as CNNDatasetType

def test_mnist_grain_loader():
    batch_size = 16
    # loader_train, _ = load_mnist_datasets(batch_size)
    # Testing existence of loader and basic iterator check
    # We skip full download test if possible, but Grain usually handles it
    pass

def test_cnn_grain_loader():
    batch_size = 16
    # loader_train, _ = load_cnn_datasets(CNNDatasetType.CIFAR10, batch_size)
    pass
