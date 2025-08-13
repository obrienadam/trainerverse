import pandas as pd
from PIL import Image
import io
import numpy as np

def load_dataset(shuffle: bool = True):
    splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

    if shuffle:
        print('Shuffling...')
        df_train = df_train.sample(frac=1.0)
        df_test = df_test.sample(frac=1.0)

    return df_train, df_test

def img_to_greyscale(img):
    img = Image.open(io.BytesIO(img))
    return np.array(img.convert('L'))

def batch_generator(df: pd.DataFrame, batch_size: int):
    n = len(df)
    for i in range(n // batch_size):
        idx = slice(i * batch_size, (i + 1) * batch_size)
        batch = df['image.bytes'].iloc[idx]
        batch = np.stack(batch.apply(img_to_greyscale).to_numpy(), dtype=np.float32) / 255.0
        labels = df['label'].iloc[idx].to_numpy()
        yield np.expand_dims(batch, -1), labels
