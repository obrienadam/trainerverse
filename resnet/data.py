from random import shuffle
import tensorflow_datasets as tfds
from enum import Enum


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


def load_cifar10_dataset():
    (train_ds, test_ds), ds_info = tfds.load(
        "cifar10",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )

    return train_ds, test_ds, ds_info


def load_cifar100_dataset():
    (train_ds, test_ds), ds_info = tfds.load(
        "cifar100",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )

    return train_ds, test_ds, ds_info


def load_data(dataset_type: DatasetType):
    if dataset_type == DatasetType.CIFAR10.value:
        return load_cifar10_dataset()
    elif dataset_type == DatasetType.CIFAR100.value:
        return load_cifar100_dataset()
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
