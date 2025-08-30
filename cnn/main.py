from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 0.001, "Learning rate for the optimizer."
)
BATCH_SIZE = flags.DEFINE_integer("batch_size", 128, "Batch size for training.")
NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train.")
SEED = flags.DEFINE_integer("seed", 9234578, "Random seed for reproducibility.")


def main(argv):
    # Delete argv[0] which is the name of the program
    del argv

    # Your main program logic goes here
    logging.info("Starting application...")


if __name__ == "__main__":
    app.run(main)
