import grain.python as grain
from enum import Enum


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


def load_datasets(dataset_type: DatasetType, batch_size: int, shuffle_seed: int = 42):
    if dataset_type == DatasetType.CIFAR10:
        name = "cifar10"
    elif dataset_type == DatasetType.CIFAR100:
        name = "cifar100"
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    train_ds = grain.TfdsDataSource(name=name, split="train", shuffle_files=True)
    test_ds = grain.TfdsDataSource(name=name, split="test", shuffle_files=False)

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
        operations=[
            grain.MapTransform(lambda x: (x["image"], x["label"])),
            grain.Batch(batch_size=batch_size)
        ],
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
        operations=[
            grain.MapTransform(lambda x: (x["image"], x["label"])),
            grain.Batch(batch_size=batch_size)
        ],
    )

    return loader_train, loader_test
