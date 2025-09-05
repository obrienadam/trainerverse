import pyarrow.parquet as pq
import tensorflow as tf
from etils import epath
import numpy as np
import random
from sklearn.utils import murmurhash3_32 as mhash

random.seed(5342)

files = list(
    map(
        str,
        epath.Path(
            "/home/aobrien/data/kaggle_datasets/criteo-display-advertising-challenge"
        ).glob("train*.parquet"),
    )
)


def get_dataset(batch_size=2048, shuffle_seed=5342):
    def file_generator():
        random.shuffle(files)
        for fn in files:
            pf = pq.ParquetFile(fn)
            for batch in pf.iter_batches(batch_size=batch_size):
                batch = batch.to_pandas()
                labels = batch["label"].values
                x_dense = np.log1p(
                    batch.iloc[:, 1:14].fillna(0.0).clip(0.0, None).values
                )
                x_sparse = (
                    batch.iloc[:, 14:]
                    .fillna("*")
                    .map(lambda col: mhash(col, seed=98237, positive=True))
                    .values
                )
                yield x_dense, x_sparse, labels

    output_signature = (
        tf.TensorSpec(shape=(None, 13), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 26), dtype=tf.uint32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        file_generator, output_signature=output_signature
    )

    ds = (
        ds.unbatch()
        .shuffle(10_000, seed=shuffle_seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds
