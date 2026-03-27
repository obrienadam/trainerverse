import grain.python as grain
import numpy as np
import os
import mmh3

class CriteoWukongDataset(grain.MapDataset):
    def __init__(self, filename="criteo_sample.txt", split="train", test_ratio=0.2):
        self.filename = filename
        self.split = split
        self.data = []
        if os.path.exists(filename):
            all_data = []
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 40:
                        label = int(parts[0])
                        dense = []
                        for i in range(1, 14):
                            try:
                                val = float(parts[i]) if parts[i] != "" else 0.0
                                dense.append(np.log1p(max(0, val)))
                            except ValueError:
                                dense.append(0.0)
                        
                        sparse = parts[14:]
                        sparse_hashed = [mmh3.hash(s, signed=False) % (2**16) for s in sparse]
                        
                        all_data.append((np.array(dense, dtype=np.float32), np.array(sparse_hashed, dtype=np.uint32), label))
            
            # Deterministic split
            num_test = int(len(all_data) * test_ratio)
            if split == "test":
                self.data = all_data[:num_test]
            else:
                self.data = all_data[num_test:]
        
        if not self.data:
            # Fallback
            for _ in range(100):
                label = np.random.randint(0, 2)
                dense = np.random.rand(13).astype(np.float32)
                sparse = [str(np.random.randint(0, 10000)) for _ in range(26)]
                sparse_hashed = [mmh3.hash(s, signed=False) % (2**16) for s in sparse]
                self.data.append((dense, np.array(sparse_hashed, dtype=np.uint32), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dense, sparse, label = self.data[idx]
        return dense, sparse, np.array(label, dtype=np.int32)


def get_dataloaders(batch_size=2048, shuffle_seed=5342):
    train_ds = CriteoWukongDataset(split="train")
    test_ds = CriteoWukongDataset(split="test")
    
    def create_loader(ds, shuffle):
        sampler = grain.IndexSampler(
            num_records=len(ds),
            num_epochs=1,
            shard_options=grain.NoSharding(),
            shuffle=shuffle,
            seed=shuffle_seed,
        )
        return grain.DataLoader(
            data_source=ds,
            sampler=sampler,
            worker_count=0,
            operations=[grain.Batch(batch_size=batch_size)],
        )
    
    return create_loader(train_ds, True), create_loader(test_ds, False)
