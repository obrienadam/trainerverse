from absl import app
from absl import flags
from train import TrainConfig, train

FLAGS = flags.FLAGS

LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 200, "Number of epochs.")
BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Batch size.")
SEED = flags.DEFINE_integer("seed", 1425, "Random seed.")


def main(argv):
    del argv  # Unused.
    train_config = TrainConfig(
        learning_rate=LEARNING_RATE.value,
        num_epochs=NUM_EPOCHS.value,
        batch_size=BATCH_SIZE.value,
        seed=SEED.value,
    )
    train(train_config)


if __name__ == "__main__":
    app.run(main)
