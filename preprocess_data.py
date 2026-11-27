"""Export the exact stratified 48x48 splits used for training and evaluation."""

import argparse
from pathlib import Path

import numpy as np

from training import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="data/preprocessed_data.npz")
    args = parser.parse_args()
    xtrain, ytrain, xval, yval, xtest, ytest, digest = prepare_dataset(args.data)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_train=xtrain,
        y_train=ytrain,
        X_val=xval,
        y_val=yval,
        X_test=xtest,
        y_test=ytest,
        dataset_sha256=digest,
    )
    print(
        f"Saved {len(xtrain)} training, {len(xval)} validation, and {len(xtest)} untouched test images to {out}"
    )
