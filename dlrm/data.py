import grain.python as grain
import numpy as np
import os
import csv
import mmh3


class CriteoDataset(grain.MapDataset):
    def __init__(self, path, num_examples=None):
        self.path = path
        self.data = []
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for i, row in enumerate(reader):
                    if num_examples and i >= num_examples:
                        break
                    self.data.append(row)
        else:
            # Fallback for synthetic data
            self.num_examples = num_examples or 10000
            self.data = None

    def __len__(self):
        return len(self.data) if self.data is not None else self.num_examples

    def __getitem__(self, idx):
        if self.data is not None:
            row = self.data[idx]
            label = int(row[0])
            dense = [float(x) if x else 0.0 for x in row[1:14]]
            sparse = [x for x in row[14:40]]
        else:
            # Synthetic row
            label = np.random.randint(0, 2)
            dense = np.random.rand(13).tolist()
            sparse = [str(np.random.randint(0, 1000)) for _ in range(26)]
        
        return self._preprocess(dense, sparse, label)

    def _preprocess(self, dense, sparse, label):
        dense = np.array(dense, dtype=np.float32)
        dense = np.where(np.isnan(dense), 0.0, dense)
        dense = np.where(dense < 0.0, 0.0, dense)
        dense = np.log1p(dense)
        
        # Stable hashing for sparse features using mmh3
        sparse_hashed = []
        for val in sparse:
            hashed = mmh3.hash(val, signed=False) % (2**20)
            sparse_hashed.append(hashed)
        
        return dense, np.array(sparse_hashed, dtype=np.int32), np.array(label, dtype=np.int32)


def load_dataset(batch_size, shuffle_seed):
    train_path = "/home/aobrien/kaggle_datasets/criteo_small_train.txt"
    test_path = "/home/aobrien/kaggle_datasets/criteo_small_test.txt"

    if not os.path.exists(train_path):
        print(f"Dataset not found at {train_path}. Using synthetic data with Grain.")
        ds_train = CriteoDataset(train_path, num_examples=10000)
        ds_test = CriteoDataset(test_path, num_examples=2000)
    else:
        ds_train = CriteoDataset(train_path)
        ds_test = CriteoDataset(test_path)

    # Grain transformations for shuffling and batching
    sampler_train = grain.IndexSampler(
        num_records=len(ds_train),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=int(shuffle_seed),
    )
    
    loader_train = grain.DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        worker_count=0, # Use main process for simplicity
        operations=[grain.Batch(batch_size=batch_size)],
    )

    sampler_test = grain.IndexSampler(
        num_records=len(ds_test),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=False,
    )
    
    loader_test = grain.DataLoader(
        data_source=ds_test,
        sampler=sampler_test,
        worker_count=0,
        operations=[grain.Batch(batch_size=batch_size)],
    )

    return loader_train, loader_test
