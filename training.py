"""Train this project's expression CNN from random initialization on FER-2013."""

import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def prepare_dataset(work):
    from PIL import Image
    from sklearn.model_selection import train_test_split

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "fer2013.zip"
    if not archive.exists():
        print("Downloading FER-2013 from Kaggle (msambare/fer2013)…", flush=True)
        urllib.request.urlretrieve(
            "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013", archive
        )
    images, labels = {"train": [], "test": []}, {"train": [], "test": []}
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if (
                len(parts) != 3
                or parts[0] not in images
                or not name.lower().endswith((".png", ".jpg", ".jpeg"))
            ):
                continue
            label = parts[1].capitalize()
            if label not in LABELS:
                continue
            with z.open(name) as f, Image.open(f) as img:
                images[parts[0]].append(
                    np.asarray(img.convert("L").resize((48, 48)), dtype=np.uint8)
                )
            labels[parts[0]].append(LABELS.index(label))
    x, y = np.asarray(images["train"]), np.asarray(labels["train"])
    if len(x) < 20000 or len(images["test"]) < 5000:
        raise ValueError(
            "Dataset is incomplete; refusing to train on a partial archive."
        )
    xtrain, xval, ytrain, yval = train_test_split(
        x, y, test_size=0.15, random_state=42, stratify=y
    )
    return (
        xtrain[..., None],
        ytrain,
        xval[..., None],
        yval,
        np.asarray(images["test"])[..., None],
        np.asarray(labels["test"]),
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


def build_model():
    import tensorflow as tf

    L = tf.keras.layers
    inputs = L.Input((48, 48, 1), name="face")
    x = L.Rescaling(1 / 255.0, name="normalization")(inputs)
    x = L.RandomFlip("horizontal")(x)
    x = L.RandomRotation(0.03, fill_mode="reflect")(x)
    x = L.RandomTranslation(0.07, 0.07, fill_mode="reflect")(x)
    for width, drop in [(32, 0.15), (64, 0.2), (128, 0.25), (256, 0.3)]:
        for _ in range(2):
            x = L.Conv2D(
                width, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
            )(x)
            x = L.BatchNormalization()(x)
            x = L.Activation("relu")(x)
        x = L.MaxPooling2D()(x)
        x = L.Dropout(drop)(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(x)
    x = L.Dropout(0.4)(x)
    output = L.Dense(7, activation="softmax", dtype="float32", name="expression")(x)
    model = tf.keras.Model(inputs, output, name="nilay_expression_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(run_dir, work="/content/fer2013", epochs=65):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    data = prepare_dataset(work)
    xtrain, ytrain, xval, yval, xtest, ytest, digest = data
    print(
        "Split sizes (train / validation / untouched test):",
        len(xtrain),
        len(xval),
        len(xtest),
        flush=True,
    )
    model = build_model()
    model.summary()
    (run / "model_config.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "input_size": [48, 48],
                "channels": 1,
                "normalization": "embedded_0_255",
                "owner": "Nilay Srivastava",
                "architecture": "nilay_expression_cnn",
                "trained_from_scratch": True,
            },
            indent=2,
        )
    )
    started = time.time()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(run / "best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(run / "history.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            restore_best_weights=True,
            start_from_epoch=15,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6
        ),
    ]
    hist = model.fit(
        xtrain,
        ytrain,
        validation_data=(xval, yval),
        batch_size=128,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model = tf.keras.models.load_model(run / "best.keras", compile=False)
    scores = model.predict(xtest, batch_size=128, verbose=0)
    pred = scores.argmax(axis=1)
    report = classification_report(
        ytest, pred, target_names=LABELS, output_dict=True, zero_division=0
    )
    metrics = {
        "test_accuracy": float(np.mean(pred == ytest)),
        "macro_f1": report["macro avg"]["f1-score"],
        "train_examples": len(xtrain),
        "validation_examples": len(xval),
        "test_examples": len(xtest),
        "epochs_completed": len(hist.history["loss"]),
        "training_seconds": round(time.time() - started, 2),
        "dataset_sha256": digest,
        "trained_from_scratch": True,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(ytest, pred, labels=range(7)).tolist(),
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save(run / "emotion_recognition_model.keras")
    model.save(run / "emotion_recognition_model.h5")
    # A deterministic test batch supports deployment numerical parity checks.
    np.savez_compressed(
        run / "parity.npz", images=xtest[:12], probabilities=scores[:12]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ["accuracy", "val_accuracy"]:
        axes[0].plot(hist.history[name], label=name)
    for name in ["loss", "val_loss"]:
        axes[1].plot(hist.history[name], label=name)
    for ax in axes:
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "learning-curves.png", dpi=160)
    plt.close(fig)
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(7), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(7), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run / "confusion-matrix.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k not in ["classification_report", "confusion_matrix"]
            },
            indent=2,
        ),
        flush=True,
    )
    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/training-run")
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=65)
    args = parser.parse_args()
    train(args.output, args.data, args.epochs)
