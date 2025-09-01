# DLRM Trainer

Currently this trains on the `criteo-small` dataset. This dataset has about 1 million examples. The 80% of the examples are used for training and 20% for validation.

There seems to be some ongoing issues with splitting the dataset using `tfds`, in the future it's probably best to have these split on disk.`
