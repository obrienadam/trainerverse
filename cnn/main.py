from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


def main(argv):
    # Delete argv[0] which is the name of the program
    del argv

    # Your main program logic goes here
    logging.info("Starting application...")


if __name__ == "__main__":
    app.run(main)
