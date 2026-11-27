"""FERPlus label-distribution study with a scratch-trained residual SE network."""

import csv
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
FERPLUS_COMMIT = "ae2128abf776409c93e50cf7c9d87180673314e6"


def prepare_ferplus(work="data/ferplus"):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "fer2013-original.zip"
    annotation = work / "fer2013new.csv"
    if not archive.exists():
        print("Downloading original-order FER2013 CSV…", flush=True)
        urllib.request.urlretrieve(
            "https://www.kaggle.com/api/v1/datasets/download/deadskull7/fer2013",
            archive,
        )
    if not annotation.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/microsoft/FERPlus/{FERPLUS_COMMIT}/fer2013new.csv",
            annotation,
        )
    with zipfile.ZipFile(archive) as z:
        source = next(n for n in z.namelist() if n.endswith("fer2013.csv"))
        with z.open(source) as raw:
            original = list(csv.DictReader(io.TextIOWrapper(raw)))
    new = list(csv.DictReader(annotation.open()))
    assert len(original) == len(new) == 35887, "Annotation row count mismatch."
    # FERPlus guarantees row-wise correspondence with the original CSV.
    # Export filenames are not reliable row indices (NF entries can be blank).
    assert all(a["Usage"] == b["Usage"] for a, b in zip(original, new)), (
        "Source/annotation split order mismatch."
    )

    images = np.stack(
        [
            np.fromstring(row["pixels"], dtype=np.uint8, sep=" ").reshape(48, 48, 1)
            for row in original
        ]
    )
    usage = np.asarray([r["Usage"] for r in original])
    hard = np.asarray([int(r["emotion"]) for r in original])
    columns = [
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "neutral",
    ]
    votes = np.asarray([[float(r[c]) for c in columns] for r in new], dtype=np.float32)
    excluded = np.asarray(
        [sum(float(r[c]) for c in ["contempt", "unknown", "NF"]) for r in new]
    )
    # Keep the original seven-class task. Votes outside it are never relabelled.
    valid = (votes.sum(axis=1) >= 6) & (excluded <= 4)
    soft = votes / np.maximum(votes.sum(axis=1, keepdims=True), 1)
    hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in images]
    test_idx = np.where(usage == "PrivateTest")[0]
    test_hashes = {hashes[i] for i in test_idx}
    val_candidates = np.where((usage == "PublicTest") & valid)[0]
    val_idx = []
    val_hashes = set()
    val_overlap = 0
    for i in val_candidates:
        if hashes[i] in test_hashes:
            val_overlap += 1
            continue
        if hashes[i] not in val_hashes:
            val_idx.append(i)
            val_hashes.add(hashes[i])
    protected = test_hashes | {hashes[i] for i in np.where(usage == "PublicTest")[0]}
    groups = {}
    train_overlap = 0
    duplicate_train = 0
    for i in np.where((usage == "Training") & valid)[0]:
        h = hashes[i]
        if h in protected:
            train_overlap += 1
            continue
        if h in groups:
            groups[h][1] += votes[i]
            duplicate_train += 1
        else:
            groups[h] = [i, votes[i].copy()]
    train_idx = np.asarray([r[0] for r in groups.values()])
    train_votes = np.stack([r[1] for r in groups.values()])
    train_soft = train_votes / train_votes.sum(axis=1, keepdims=True)
    val_idx = np.asarray(val_idx)
    assert len(train_idx) > 18000 and len(val_idx) > 2500 and len(test_idx) == 3589
    assert not (
        {hashes[i] for i in train_idx} & ({hashes[i] for i in val_idx} | test_hashes)
    )
    assert not ({hashes[i] for i in val_idx} & test_hashes)
    audit = {
        "source_rows": len(original),
        "annotation_commit": FERPLUS_COMMIT,
        "original_csv_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "training_images": len(train_idx),
        "validation_images": len(val_idx),
        "test_images": len(test_idx),
        "training_overlap_removed": train_overlap,
        "training_duplicates_merged": duplicate_train,
        "validation_test_overlap_removed": val_overlap,
        "invalid_training_annotations_removed": int(
            np.sum((usage == "Training") & ~valid)
        ),
        "split_protocol": "Original Training / PublicTest validation / PrivateTest held-out test",
        "annotation_policy": "Seven supported classes with at least six valid votes; excluded-class votes are not reassigned. All remaining vote distributions are retained.",
        "test_crowd_coverage": float(valid[test_idx].mean()),
        "test_consensus_coverage": float(
            (valid[test_idx] & (votes[test_idx].max(axis=1) >= 6)).mean()
        ),
        "exact_duplicate_leakage_after_cleaning": 0,
    }
    print("Data audit:", json.dumps(audit, indent=2), flush=True)
    return (
        images[train_idx],
        train_soft,
        images[val_idx],
        soft[val_idx],
        hard[val_idx],
        images[test_idx],
        hard[test_idx],
        soft[test_idx],
        valid[test_idx],
        votes[test_idx].max(axis=1) >= 6,
        audit,
    )


def build_residual_se():
    import tensorflow as tf

    L = tf.keras.layers
    inputs = L.Input((48, 48, 1), name="face")
    x = L.Rescaling(1 / 255.0, name="normalization")(inputs)
    x = L.Conv2D(32, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(
        x
    )
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    def block(x, channels, stride=1):
        skip = x
        for j in range(2):
            x = L.Conv2D(
                channels,
                3,
                strides=stride if j == 0 else 1,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(x)
            x = L.BatchNormalization()(x)
            if j == 0:
                x = L.Activation("relu")(x)
        gate = L.GlobalAveragePooling2D(keepdims=True)(x)
        gate = L.Dense(max(channels // 8, 4), activation="relu")(gate)
        gate = L.Dense(channels, activation="sigmoid")(gate)
        x = L.Multiply()([x, gate])
        if stride != 1 or skip.shape[-1] != channels:
            skip = L.Conv2D(channels, 1, strides=stride, use_bias=False)(skip)
            skip = L.BatchNormalization()(skip)
        return L.Activation("relu")(L.Add()([x, skip]))

    for stage, channels in enumerate([32, 64, 128, 256]):
        x = block(x, channels, 1 if stage == 0 else 2)
        x = block(x, channels)
    x = L.Concatenate()([L.GlobalAveragePooling2D()(x), L.GlobalMaxPooling2D()(x)])
    x = L.Dense(
        256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = L.Dropout(0.35)(x)
    outputs = L.Dense(7, activation="softmax", dtype="float32", name="expression")(x)
    return tf.keras.Model(inputs, outputs, name="nilay_ferplus_residual_se")


def float32_clone(model):
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("float32")

    def clone(layer):
        cfg = layer.get_config()
        cfg["dtype"] = "float32"
        return layer.__class__.from_config(cfg)

    out = tf.keras.models.clone_model(model, clone_function=clone)
    out.set_weights(model.get_weights())
    return out


def summarize(y, p):
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    pred = p.argmax(axis=1)
    report = classification_report(
        y, pred, labels=range(7), target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": report["macro avg"]["f1-score"],
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "samples": len(y),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, pred, labels=range(7)).tolist(),
    }


def train_improved(run_dir, baseline_dir, epochs=100, work="data/ferplus"):
    import matplotlib.pyplot as plt
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(2026)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    data = prepare_ferplus(work)
    (
        xtrain,
        ytrain,
        xval,
        yval,
        _yvalraw,
        xtest,
        ytestraw,
        ytestsoft,
        valid_test,
        consensus_test,
        audit,
    ) = data
    (run / "data-audit.json").write_text(json.dumps(audit, indent=2))
    class_mass = ytrain.sum(axis=0)
    class_weight = np.clip((class_mass.mean() / class_mass) ** 0.25, 0.8, 1.8)
    weights = (ytrain * class_weight).sum(axis=1)
    weights /= weights.mean()
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.025, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="reflect"),
        ]
    )
    ds = (
        tf.data.Dataset.from_tensor_slices((xtrain, ytrain, weights.astype(np.float32)))
        .shuffle(len(xtrain), seed=2026)
        .batch(128)
    )
    ds = ds.map(
        lambda x, y, w: (augmentation(tf.cast(x, tf.float32), training=True), y, w),
        num_parallel_calls=2,
    ).prefetch(2)
    model = build_residual_se()
    steps = int(np.ceil(len(xtrain) / 128))
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=(epochs - 4) * steps,
        alpha=0.01,
        warmup_target=0.001,
        warmup_steps=4 * steps,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    print(
        "Parameters:",
        model.count_params(),
        "Class balance weights:",
        class_weight,
        flush=True,
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(run / "best.keras"), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(run / "history.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=16,
            restore_best_weights=True,
            start_from_epoch=25,
        ),
    ]
    start = time.time()
    history = model.fit(
        ds, validation_data=(xval, yval), epochs=epochs, callbacks=callbacks, verbose=2
    )
    model = float32_clone(tf.keras.models.load_model(run / "best.keras", compile=False))

    def predict(x):
        return model.predict(x.astype(np.float32), batch_size=128, verbose=0)

    val_single = predict(xval)
    val_flip = predict(xval[:, :, ::-1, :])
    val_tta = (val_single + val_flip) / 2
    va = summarize(yval.argmax(axis=1), val_single)
    vt = summarize(yval.argmax(axis=1), val_tta)
    # Decide whether to enable flip TTA using validation data only.
    use_tta = (vt["accuracy"] + vt["macro_f1"]) > (va["accuracy"] + va["macro_f1"])
    test_single = predict(xtest)
    test_flip = predict(xtest[:, :, ::-1, :])
    test_prob = (test_single + test_flip) / 2 if use_tta else test_single
    baseline = tf.keras.models.load_model(
        Path(baseline_dir) / "emotion_recognition_model.keras", compile=False
    )
    base_prob = baseline.predict(xtest.astype(np.float32), batch_size=128, verbose=0)
    comparable_baseline = summarize(ytestraw, base_prob)
    raw_metrics = summarize(ytestraw, test_prob)
    crowd_metrics = summarize(
        ytestsoft[valid_test].argmax(axis=1), test_prob[valid_test]
    )
    clean = valid_test & consensus_test
    consensus_metrics = summarize(ytestsoft[clean].argmax(axis=1), test_prob[clean])
    comparison = {
        "protocol": audit["split_protocol"],
        "baseline_original_labels": comparable_baseline,
        "improved_original_labels": raw_metrics,
        "baseline_crowd_labels": summarize(
            ytestsoft[valid_test].argmax(axis=1), base_prob[valid_test]
        ),
        "improved_crowd_labels": crowd_metrics,
        "consensus_subset": consensus_metrics,
        "crowd_coverage": audit["test_crowd_coverage"],
        "consensus_coverage": audit["test_consensus_coverage"],
        "validation_single": va,
        "validation_flip_tta": vt,
        "tta_selected_on_validation": bool(use_tta),
        "raw_label_disagreement_with_crowd": float(
            np.mean(ytestraw[valid_test] != ytestsoft[valid_test].argmax(axis=1))
        ),
    }
    (run / "comparison.json").write_text(json.dumps(comparison, indent=2))
    metrics = {
        "test_accuracy": crowd_metrics["accuracy"],
        "macro_f1": crowd_metrics["macro_f1"],
        "test_examples": crowd_metrics["samples"],
        "test_protocol": "FERPlus seven-class labels on original PrivateTest; all eligible annotations, not only high-consensus examples",
        "test_coverage": audit["test_crowd_coverage"],
        "original_fer2013_accuracy": raw_metrics["accuracy"],
        "original_fer2013_macro_f1": raw_metrics["macro_f1"],
        "original_fer2013_test_examples": len(xtest),
        "epochs_completed": len(history.history["loss"]),
        "training_seconds": round(time.time() - start, 2),
        "trained_from_scratch": True,
        "architecture": "Residual CNN with squeeze-and-excitation attention",
        "parameters": model.count_params(),
        "classification_report": crowd_metrics["classification_report"],
        "confusion_matrix": crowd_metrics["confusion_matrix"],
        "test_time_augmentation": bool(use_tta),
        "balanced_accuracy": crowd_metrics["balanced_accuracy"],
        "brier_against_vote_distribution": float(
            np.mean(
                np.sum((test_prob[valid_test] - ytestsoft[valid_test]) ** 2, axis=1)
            )
        ),
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    config = {
        "labels": LABELS,
        "input_size": [48, 48],
        "channels": 1,
        "normalization": "embedded_0_255",
        "owner": "Nilay Srivastava",
        "architecture": "nilay_ferplus_residual_se",
        "trained_from_scratch": True,
        "test_time_augmentation": bool(use_tta),
        "dataset": "FERPlus",
        "annotation_commit": FERPLUS_COMMIT,
    }
    (run / "model_config.json").write_text(json.dumps(config, indent=2))
    model.save(run / "emotion_recognition_model.keras")
    model.save(run / "emotion_recognition_model.h5")
    np.savez_compressed(
        run / "parity.npz",
        images=xtest[:16],
        probabilities=test_single[:16],
        pipeline_probabilities=test_prob[:16],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ["accuracy", "val_accuracy"]:
        axes[0].plot(history.history[name], label=name)
    for name in ["loss", "val_loss"]:
        axes[1].plot(history.history[name], label=name)
    for ax in axes:
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "learning-curves.png", dpi=160)
    plt.close(fig)
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(7), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(7), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Crowd-labelled reference")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run / "confusion-matrix.png", dpi=160)
    plt.close(fig)
    print(
        "Study complete:",
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

    parser = argparse.ArgumentParser(
        description="Train the FERPlus residual study from scratch."
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output", default="models/ferplus-training-run")
    parser.add_argument("--data", default="data/ferplus")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train_improved(args.output, args.baseline_dir, args.epochs, args.data)
