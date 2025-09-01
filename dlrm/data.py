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


def load_dataset(batch_size):
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
            sloppy=True,
            shuffle_buffer_size=10000,
        )
        .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds
