from random import shuffle
import tensorflow_datasets as tfds


def load_cifar10_dataset():
    (train_ds, test_ds), ds_info = tfds.load(
        "cifar10",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True
    )

    return train_ds, test_ds, ds_info
