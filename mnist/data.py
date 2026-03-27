import grain.python as grain
import pandas as pd
from PIL import Image
import io
import numpy as np


class MNISTDataset(grain.MapDataset):
    def __init__(self, split: str = "train"):
        parquet_file = f"mnist/{split}-00000-of-00001.parquet"
        self.df = pd.read_parquet(f"hf://datasets/ylecun/mnist/{parquet_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._img_to_numpy(row["image"])
        label = int(row["label"])
        return image, label

    def _img_to_numpy(self, img_dict):
        img = Image.open(io.BytesIO(img_dict['bytes']))
        img_array = np.array(img.convert("L"), dtype=np.float32) / 255.0
        return np.expand_dims(img_array, -1)


def load_datasets(batch_size: int, shuffle_seed: int = 42):
    train_ds = MNISTDataset(split="train")
    test_ds = MNISTDataset(split="test")

    sampler_train = grain.IndexSampler(
        num_records=len(train_ds),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=shuffle_seed,
    )
    
    loader_train = grain.DataLoader(
        data_source=train_ds,
        sampler=sampler_train,
        worker_count=0,
        operations=[grain.Batch(batch_size=batch_size)],
    )

    sampler_test = grain.IndexSampler(
        num_records=len(test_ds),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=False,
    )
    
    loader_test = grain.DataLoader(
        data_source=test_ds,
        sampler=sampler_test,
        worker_count=0,
        operations=[grain.Batch(batch_size=batch_size)],
    )

    return loader_train, loader_test
