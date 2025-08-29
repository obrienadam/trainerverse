import train
from absl import flags
from absl import app

NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Batch size.")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")


def main(argv):
    del argv  # Unused.
    train.train(
        train.HyperParams(
            learning_rate=LEARNING_RATE.value,
            batch_size=BATCH_SIZE.value,
            num_epochs=NUM_EPOCHS.value,
        )
    )


if __name__ == "__main__":
    app.run(main)
