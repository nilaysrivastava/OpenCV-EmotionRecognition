"""Inspect preprocessed examples without assuming a fixed test-set size."""

import argparse
from pathlib import Path

import numpy as np

from training import LABELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/preprocessed_data.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test-images.png"))
    args = parser.parse_args()
    if not args.data.exists():
        parser.error("Preprocessed data not found. Run preprocess_data.py first.")
    import matplotlib.pyplot as plt

    data = np.load(args.data)
    n = min(7, len(data["X_test"]))
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2), squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.imshow(data["X_test"][i].squeeze(), cmap="gray")
        ax.set_title(LABELS[int(data["y_test"][i])])
        ax.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(args.output)
