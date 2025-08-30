from absl import app
from absl import flags
from absl import logging

from model import ModelParams
from train import HyperParams, train

FLAGS = flags.FLAGS

LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 0.001, "Learning rate for the optimizer."
)
BATCH_SIZE = flags.DEFINE_integer("batch_size", 128, "Batch size for training.")
NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train.")
SEED = flags.DEFINE_integer("seed", 9234578, "Random seed for reproducibility.")

FEATURES = flags.DEFINE_list(
    "features",
    ["32", "64", "128"],
    "Comma-separated list of feature sizes for each Conv layer.",
)
KERNEL_SIZE = flags.DEFINE_list(
    "kernel_size",
    ["3", "3"],
    "Kernel size for Conv layers as two comma-separated integers.",
)
MLP_DIMS = flags.DEFINE_list(
    "mlp_dims",
    ["256", "128"],
    "Comma-separated list of hidden layer sizes for the MLP.",
)
WINDOW_SHAPE = flags.DEFINE_list(
    "window_shape",
    ["3", "3"],
    "Window shape for max pooling as two comma-separated integers.",
)


def main(argv):
    # Delete argv[0] which is the name of the program
    del argv

    # Your main program logic goes here
    logging.info("Starting application...")

    params = ModelParams(
        features=[int(f) for f in FEATURES.value],
        kernel_size=(int(KERNEL_SIZE.value[0]), int(KERNEL_SIZE.value[1])),
        mlp_dims=[int(d) for d in MLP_DIMS.value],
        window_shape=(int(WINDOW_SHAPE.value[0]), int(WINDOW_SHAPE.value[1])),
    )
    hyperparams = HyperParams(
        learning_rate=LEARNING_RATE.value,
        batch_size=BATCH_SIZE.value,
        num_epochs=NUM_EPOCHS.value,
        seed=SEED.value,
    )
    train(params, hyperparams)


if __name__ == "__main__":
    app.run(main)
