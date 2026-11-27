import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.inference import ROOT, ModelService
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def image_bytes(size=(96, 96)):
    b = io.BytesIO()
    Image.new("RGB", size, "white").save(b, format="PNG")
    return b.getvalue()


def test_health_is_honest(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model"]["owner"] == "Nilay Srivastava"
    assert r.json()["model"]["stores_images"] is False


def test_portrait_detection_without_replacement_model():
    service = ModelService(ROOT / "tests" / "no-model")
    result = service.analyze((ROOT / "static/assets/sample-portrait.jpg").read_bytes())
    assert result["model_ready"] is False
    assert len(result["faces"]) >= 1
    assert all("label" not in face and "scores" not in face for face in result["faces"])


def test_blank_image_is_not_given_an_expression(client):
    r = client.post(
        "/api/analyze", content=image_bytes(), headers={"Content-Type": "image/png"}
    )
    assert r.status_code == 200
    assert r.json()["faces"] == []
    assert "No face detected" in r.json()["message"]


def test_invalid_image_rejected(client):
    assert (
        client.post(
            "/api/analyze",
            content=b"not an image",
            headers={"Content-Type": "image/png"},
        ).status_code
        == 422
    )


def test_svg_and_empty_rejected(client):
    assert (
        client.post(
            "/api/analyze", content=b"<svg/>", headers={"Content-Type": "image/svg+xml"}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/analyze", content=b"", headers={"Content-Type": "image/jpeg"}
        ).status_code
        == 400
    )


def test_payload_limit(client):
    assert (
        client.post(
            "/api/analyze",
            content=b"x" * (8 * 1024 * 1024 + 1),
            headers={"Content-Type": "image/jpeg"},
        ).status_code
        == 413
    )


def test_large_pixel_image_rejected(client):
    r = client.post(
        "/api/analyze",
        content=image_bytes((4100, 4000)),
        headers={"Content-Type": "image/png"},
    )
    assert r.status_code == 422


def test_no_storage_and_security_headers(client):
    r = client.post(
        "/api/analyze", content=image_bytes(), headers={"Content-Type": "image/png"}
    )
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_runtime_dimensions_and_class_mapping(tmp_path):
    service = ModelService(tmp_path)
    service.config = {"input_size": [128, 128], "normalization": "divide_by_255"}
    service.labels = ["Sad", "Angry", "Fear", "Happy", "Neutral", "Surprise", "Disgust"]

    class FixtureModel:
        def __call__(self, x, training):
            assert x.shape == (1, 128, 128, 1) and x.max() <= 1
            return np.asarray([[0.8, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03]])

    service.model = FixtureModel()
    result = service.predict(np.full((80, 80, 3), 255, dtype=np.uint8))
    assert result["label"] == "Sad" and result["score"] == 0.8


def test_pages_and_assets(client):
    for path in [
        "/",
        "/static/app.js",
        "/static/style.css",
        "/static/assets/body-plex.ttf",
        "/static/assets/display-condensed.ttf",
    ]:
        assert client.get(path).status_code == 200


def test_shipped_model_and_evaluation_agree(client):
    health = client.get("/api/health").json()["model"]
    report = client.get("/api/evaluation").json()
    assert health["ready"] is True
    assert health["trained_from_scratch"] is True
    assert health["test_accuracy"] == report["metrics"]["test_accuracy"]
    assert (
        sum(map(sum, report["metrics"]["confusion_matrix"])) == health["test_examples"]
    )
    result = client.post(
        "/api/analyze",
        content=(ROOT / "static/assets/sample-portrait.jpg").read_bytes(),
        headers={"Content-Type": "image/jpeg"},
    ).json()
    face = result["faces"][0]
    assert set(face["scores"]) == set(health["labels"])
    assert abs(sum(face["scores"].values()) - 1) < 1e-4
    assert face["label"] == max(face["scores"], key=face["scores"].get)
    assert isinstance(face["quality_flags"], list)
