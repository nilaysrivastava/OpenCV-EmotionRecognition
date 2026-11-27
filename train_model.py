"""Backwards-compatible training entry point. See training.py for the CNN."""

import argparse

from training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train this project’s CNN from random initialization."
    )
    parser.add_argument("--output", default="models/training-run")
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=65)
    args = parser.parse_args()
    train(args.output, args.data, args.epochs)
