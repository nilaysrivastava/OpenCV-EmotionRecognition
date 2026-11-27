"""Export the trained project CNN, then verify parity against its saved test batch."""

import hashlib
import json
from pathlib import Path

import numpy as np


def export(run_dir):
    import onnxruntime as ort
    import tensorflow as tf

    run = Path(run_dir)
    model = tf.keras.models.load_model(
        run / "emotion_recognition_model.keras", compile=False
    )

    # Clone the learned functional graph while making training-only operations
    # explicit identities. This preserves residual/attention connections.
    def clone(layer):
        training_only = (
            tf.keras.layers.RandomFlip,
            tf.keras.layers.RandomRotation,
            tf.keras.layers.RandomTranslation,
            tf.keras.layers.Dropout,
        )
        if isinstance(layer, training_only):
            return tf.keras.layers.Activation(
                "linear", name=layer.name, dtype="float32"
            )
        cfg = layer.get_config()
        cfg["dtype"] = "float32"
        return layer.__class__.from_config(cfg)

    inference = tf.keras.models.clone_model(model, clone_function=clone)
    inference.set_weights(model.get_weights())
    signature = [tf.TensorSpec((None, 48, 48, 1), tf.float32, name="face")]

    @tf.function(input_signature=signature)
    def serving(face):
        return inference(face, training=False)

    import tf2onnx

    target = run / "emotion_recognition_model.onnx"
    tf2onnx.convert.from_function(
        serving, input_signature=signature, opset=17, output_path=str(target)
    )
    batch = np.load(run / "parity.npz")
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    converted = session.run(
        None, {session.get_inputs()[0].name: batch["images"].astype(np.float32)}
    )[0]
    error = float(np.max(np.abs(converted - batch["probabilities"])))
    assert error < 1e-4, f"Conversion parity failed: {error}"
    report = {
        "onnx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "max_absolute_error": error,
        "parity_passed": True,
        "samples": len(converted),
        "same_top_class": bool(
            np.all(converted.argmax(axis=1) == batch["probabilities"].argmax(axis=1))
        ),
    }
    assert report["same_top_class"]
    config = json.loads((run / "model_config.json").read_text())
    if "pipeline_probabilities" in batch:
        flip = session.run(
            None,
            {
                session.get_inputs()[0].name: batch["images"][:, :, ::-1, :].astype(
                    np.float32
                )
            },
        )[0]
        pipeline = (
            (converted + flip) / 2
            if config.get("test_time_augmentation")
            else converted
        )
        report["pipeline_max_absolute_error"] = float(
            np.max(np.abs(pipeline - batch["pipeline_probabilities"]))
        )
        assert report["pipeline_max_absolute_error"] < 1e-4

    (run / "export-verification.json").write_text(json.dumps(report, indent=2))
    print("Export verified:", json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    export(p.parse_args().run_dir)
