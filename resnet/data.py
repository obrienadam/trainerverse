import grain.python as grain
from enum import Enum
import numpy as np
from PIL import Image


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TF_FLOWERS = "tf_flowers"


def resize_and_crop(item, size=(224, 224)):
    img = item["image"]
    label = item["label"]
    
    # Using PIL for pure python resizing (Grain style)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
    
    return np.array(pil_img, dtype=np.float32), label


def load_datasets(dataset_type: DatasetType, batch_size: int, shuffle_seed: int = 42):
    if dataset_type == DatasetType.CIFAR10:
        name = "cifar10"
    elif dataset_type == DatasetType.CIFAR100:
        name = "cifar100"
    elif dataset_type == DatasetType.TF_FLOWERS:
        name = "tf_flowers"
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    train_ds = grain.TfdsDataSource(name=name, split="train" if name != "tf_flowers" else "train[:80%]", shuffle_files=True)
    test_ds = grain.TfdsDataSource(name=name, split="test" if name != "tf_flowers" else "train[80%:]", shuffle_files=False)

    def get_loader(ds, shuffle):
        sampler = grain.IndexSampler(
            num_records=len(ds),
            num_epochs=1,
            shard_options=grain.NoSharding(),
            shuffle=shuffle,
            seed=shuffle_seed,
        )
        
        ops = []
        if dataset_type == DatasetType.TF_FLOWERS:
            ops.append(grain.MapTransform(resize_and_crop))
        else:
            ops.append(grain.MapTransform(lambda x: (x["image"].astype(np.float32), x["label"])))
            
        ops.append(grain.Batch(batch_size=batch_size))
        
        return grain.DataLoader(
            data_source=ds,
            sampler=sampler,
            worker_count=0,
            operations=ops,
        )

    return get_loader(train_ds, True), get_loader(test_ds, False)
