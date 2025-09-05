import absl.flags
import train
import model
import absl

DENSE_LR = absl.flags.DEFINE_float(
    "dense_lr", 0.001, "Learning rate for dense parameters."
)
SPARSE_LR = absl.flags.DEFINE_float(
    "sparse_lr", 0.01, "Learning rate for sparse parameters."
)
NUM_EPOCHS = absl.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs.")
BATCH_SIZE = absl.flags.DEFINE_integer("batch_size", 1024, "Batch size.")
SEED = absl.flags.DEFINE_integer("seed", 4753, "Random seed.")
EMBEDDING_DIM = absl.flags.DEFINE_integer(
    "embedding_dim", 16, "Dimension of embedding vectors."
)


def main(argv):
    model_config = model.DLRMConfig(
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
    absl.app.run(main)
