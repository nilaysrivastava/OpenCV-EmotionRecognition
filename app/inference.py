"""Shared inference for the web app and CLI. Only project-trained weights are loaded."""

import hashlib
import io
import json
import os
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
cv2.setNumThreads(2)
Image.MAX_IMAGE_PIXELS = 16_000_000


class InputError(ValueError):
    pass


class ModelService:
    def __init__(self, directory=None):
        self.directory = Path(
            directory or os.environ.get("MODEL_DIR", str(ROOT / "models"))
        )
        self.lock = Lock()
        self.session = None
        self.model = None
        self.config = {}
        self.error = None
        self.metrics = self._read_json("metrics.json")
        self.comparison = self._read_json("comparison.json")
        self.audit = self._read_json("data-audit.json")
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.detector.empty():
            raise RuntimeError("OpenCV face detector is unavailable.")
        self._load()

    def _read_json(self, name):
        path = self.directory / name
        return json.loads(path.read_text()) if path.exists() else {}

    def _load(self):
        try:
            config = self.directory / "model_config.json"
            if config.exists():
                self.config = json.loads(config.read_text())
            if not self.config.get("trained_from_scratch"):
                self.error = "The project-trained CNN has not been installed yet."
                return
            labels = self.config.get("labels", LABELS)
            if len(labels) != 7 or len(set(labels)) != 7:
                raise ValueError("Model must define seven unique class labels.")
            if (self.directory / "emotion_recognition_model.onnx").exists():
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(
                    str(self.directory / "emotion_recognition_model.onnx"),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.model_file = self.directory / "emotion_recognition_model.onnx"
            else:
                model = next(
                    (
                        self.directory / n
                        for n in [
                            "emotion_recognition_model.keras",
                            "emotion_recognition_model.h5",
                        ]
                        if (self.directory / n).exists()
                    ),
                    None,
                )
                if model is None:
                    self.error = (
                        "The project-trained CNN weights have not been installed yet."
                    )
                    return
                from tensorflow.keras.models import load_model

                self.model = load_model(model, compile=False, safe_mode=True)
                self.model_file = model
                shape = self.model.input_shape
                self.config["input_size"] = [int(shape[1]), int(shape[2])]
                self.config["channels"] = int(shape[3])
            self.labels = labels
            self.digest = hashlib.sha256(self.model_file.read_bytes()).hexdigest()
        except Exception as exc:
            self.error = "The project model could not be loaded. Check the model configuration and runtime dependencies."
            import logging

            logging.getLogger(__name__).exception(
                "Model loading failed: %s", type(exc).__name__
            )

    def status(self):
        ready = self.session is not None or self.model is not None
        metrics = self.metrics
        return {
            "ready": ready,
            "name": "Residual-SE CNN"
            if "ferplus" in self.config.get("architecture", "")
            else "Baseline CNN",
            "dataset": self.config.get("dataset", "FER-2013"),
            "test_protocol": metrics.get("test_protocol", "Held-out FER-2013 images"),
            "test_coverage": metrics.get("test_coverage", 1),
            "owner": "Nilay Srivastava",
            "trained_from_scratch": True,
            "labels": self.config.get("labels", LABELS),
            "input_size": self.config.get("input_size", [48, 48]),
            "message": self.error,
            "sha256": getattr(self, "digest", None),
            "test_accuracy": metrics.get("test_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "epochs_completed": metrics.get("epochs_completed"),
            "test_examples": metrics.get("test_examples"),
            "detector": "OpenCV Haar Cascade",
            "stores_images": False,
        }

    def predict(self, face):
        height, width = self.config.get("input_size", [48, 48])
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )[None, ..., None]
        if self.config.get("channels", 1) == 3:
            x = np.repeat(x, 3, axis=-1)
        if self.config.get("normalization") == "divide_by_255":
            x /= 255.0
        tta = self.config.get("test_time_augmentation", False)
        inputs = np.concatenate([x, x[:, :, ::-1, :]], axis=0) if tta else x
        if self.session is not None:
            spec = self.session.get_inputs()[0]
            if (
                len(spec.shape) == 4
                and spec.shape[1] in [1, 3]
                and spec.shape[-1] not in [1, 3]
            ):
                inputs = np.transpose(inputs, (0, 3, 1, 2))
            values = np.asarray(self.session.run(None, {spec.name: inputs})[0]).mean(
                axis=0
            )
        else:
            values = np.asarray(self.model(inputs, training=False)).mean(axis=0)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("Model returned invalid scores.")
        if np.any(values < 0) or not np.isclose(values.sum(), 1, atol=0.01):
            values = np.exp(values - values.max())
            values /= values.sum()
        scores = {
            label: round(float(score), 6) for label, score in zip(self.labels, values)
        }
        top = self.labels[int(np.argmax(values))]
        return {"label": top, "score": scores[top], "scores": scores}

    def analyze(self, data):
        start = time.perf_counter()
        try:
            with Image.open(io.BytesIO(data)) as raw:
                if raw.width * raw.height > 16_000_000:
                    raise InputError("Choose an image under 16 megapixels.")
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((1280, 1280))
                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InputError("This file is not a supported image.") from exc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self.lock:
            boxes = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            boxes = sorted(boxes, key=lambda b: int(b[2]) * int(b[3]), reverse=True)[:8]
            faces = []
            for i, (x, y, w, h) in enumerate(boxes):
                face = {
                    "id": i + 1,
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                    },
                }
                roi = frame[y : y + h, x : x + w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                quality = []
                if min(w, h) < 64:
                    quality.append("small_face")
                if float(roi_gray.mean()) < 55:
                    quality.append("low_light")
                if (
                    float(
                        cv2.Laplacian(cv2.resize(roi_gray, (96, 96)), cv2.CV_64F).var()
                    )
                    < 30
                ):
                    quality.append("soft_focus")
                face["quality_flags"] = quality
                if self.session is not None or self.model is not None:
                    face.update(self.predict(roi))
                faces.append(face)
        return {
            "width": frame.shape[1],
            "height": frame.shape[0],
            "faces": faces,
            "inference_ms": round((time.perf_counter() - start) * 1000, 1),
            "model_ready": self.status()["ready"],
            "labels": self.config.get("labels", LABELS),
            "message": (
                "No face detected. Try a front-facing, well-lit portrait."
                if not faces
                else self.error
            ),
        }
