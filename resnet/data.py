from random import shuffle
import tensorflow_datasets as tfds
from enum import Enum
import tensorflow as tf


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TF_FLOWERS = "tf_flowers"


def load_cifar10_dataset(batch_size: int, seed=213423):
    with tf.device("/CPU:0"):
        (train_ds, test_ds), ds_info = tfds.load(
            "cifar10",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
            shuffle_files=True,
        )

        train_ds = (
            train_ds.shuffle(10_000, seed=seed)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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


def load_tf_flowers(batch_size: int):
    with tf.device("/CPU:0"):
        ds_train, ds_info = tfds.load(
            "tf_flowers",
            split="train[:80%]",
            as_supervised=True,
            shuffle_files=False,
            with_info=True,
        )
        ds_test = tfds.load(
            "tf_flowers",
            split="train[80%:]",
            as_supervised=True,
            shuffle_files=False,
            with_info=False,
        )

        def preprocess(img, label):
            img = tf.image.resize_with_crop_or_pad(img, 224, 224)
            return img, label

        ds_train = (
            ds_train.shuffle(10_000)
            .map(preprocess)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds_train, ds_test, ds_info


def load_data(dataset_type: DatasetType, batch_size: int):
    if dataset_type == DatasetType.CIFAR10.value:
        return load_cifar10_dataset(batch_size=batch_size)
    elif dataset_type == DatasetType.CIFAR100.value:
        return load_cifar100_dataset()
    elif dataset_type == DatasetType.TF_FLOWERS.value:
        return load_tf_flowers(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
