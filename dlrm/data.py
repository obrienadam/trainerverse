from jax import config
import tensorflow as tf

_hashing_layer = tf.keras.layers.Hashing(num_bins=2**20)


def _preprocess(features, labels):
    dense_feature_names = [f"f{i}" for i in range(13)]
    sparse_feature_names = [f"c{i}" for i in range(26)]
    dense = tf.stack([features.pop(name) for name in dense_feature_names], axis=1)
    dense = tf.where(tf.math.is_nan(dense), 0.0, dense)
    dense = tf.where(dense < 0.0, 0.0, dense)
    dense = tf.math.log1p(dense)
    sparse = tf.stack([features.pop(name) for name in sparse_feature_names], axis=1)
    sparse = _hashing_layer(sparse)
    return dense, sparse, labels


def load_dataset(batch_size, shuffle_seed):
    with tf.device("/CPU:0"):
        column_names = (
            ["labels"] + [f"f{i}" for i in range(13)] + [f"c{i}" for i in range(26)]
        )
        column_defaults = [0] + [0.0] * 13 + ["*"] * 26

        ds = (
            tf.data.experimental.make_csv_dataset(
                "/home/aobrien/kaggle_datasets/train_1m.txt",
                field_delim="\t",
                column_names=column_names,
                column_defaults=column_defaults,
                batch_size=batch_size,
                label_name="labels",
                num_parallel_reads=tf.data.AUTOTUNE,
                sloppy=False,
                shuffle=False,
                num_epochs=1,
            )
            .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch()
        )

        ds_train = ds.take(800_000)
        ds_test = ds.skip(800_000)

        ds_train = (
            ds_train.shuffle(10_000, seed=shuffle_seed)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_test
