from absl import app, flags
from train import TrainConfig, train
from model import DLRMConfig

FLAGS = flags.FLAGS

EMBEDDING_DIM = flags.DEFINE_integer("embedding_dim", 64, "Dimension of embeddings.")
BOTTOM_MLP_DIMS = flags.DEFINE_list(
    "bottom_mlp_dims",
    [512, 256, 64],
    "Dimensions of bottom MLP layers.",
)
TOP_MLP_DIMS = flags.DEFINE_list(
    "top_mlp_dims",
    [1024, 512, 256],
    "Dimensions of top MLP layers.",
)
DENSE_LR = flags.DEFINE_float("dense_lr", 0.001, "Learning rate for dense parameters.")
SPARSE_LR = flags.DEFINE_float(
    "sparse_lr", 0.01, "Learning rate for sparse parameters."
)
NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 1, "Number of training epochs.")
BATCH_SIZE = flags.DEFINE_integer("batch_size", 1024, "Batch size.")
SEED = flags.DEFINE_integer("seed", 4753, "Random seed.")

_VOCAB_SIZES = [
    1261,
    531,
    321439,
    120965,
    267,
    16,
    10863,
    563,
    3,
    30792,
    4731,
    268488,
    3068,
    26,
    8934,
    205924,
    10,
    3881,
    1855,
    4,
    240748,
    16,
    15,
    41283,
    70,
    30956,
]


def main(argv):
    model_config = DLRMConfig(
        num_embeddings=_VOCAB_SIZES,
        embedding_dim=FLAGS.embedding_dim,
        bottom_mlp_dims=FLAGS.bottom_mlp_dims,
        top_mlp_dims=FLAGS.top_mlp_dims,
    )

    config = TrainConfig(
        model_config=model_config,
        dense_learning_rate=FLAGS.dense_lr,
        sparse_learning_rate=FLAGS.sparse_lr,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        seed=FLAGS.seed,
    )

    train(config)


if __name__ == "__main__":
    app.run(main)
