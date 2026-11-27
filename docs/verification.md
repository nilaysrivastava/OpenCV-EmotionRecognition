# Verification — 27 November 2026

## Model

- Own residual-SE classifier trained from scratch, 75 epochs on a free Colab T4.
- FERPlus test accuracy 83.60%, macro F1 0.701, 3,530 eligible PrivateTest images, 98.36% coverage.
- Both models scored on identical original and crowd references; the full comparison and known weaknesses are in the model card.
- ONNX export SHA-256 `f5773fae1cc6679d2dc0b1d2a7ea01130ce51fb2dcc4598b646edaba5df1764b` verified locally.
- Full application preprocessing and flip-averaged predictions matched the saved 16-image training parity batch: largest absolute difference 0.00000148, identical top classes.
- Shared CLI successfully analyzed the public-domain NASA portrait and wrote a JSON observation.

## App

- 11 pytest tests passed, including actual shipped model inference and evaluation consistency, missing-model behavior, no-face image, invalid/empty uploads, format and size limits, class order/input dimensions and HTTP security headers.
- Ruff checks, JavaScript syntax check and dependency checks passed.
- In-app browser: public sample analyzed with real scores; multi-face upload and selection worked; model-notes dialog worked; evaluation matrix, per-class report and comparison rendered from actual saved metrics.
- Local webcam permission, capture and Stop behavior were tested with user authorization; no webcam image was saved. Camera was not tested against a hosted service.
- 390-pixel iframe viewport: no horizontal document overflow (clientWidth and scrollWidth both 390); sample inference and new model/evaluation data rendered. The temporary local preview helper's iframe header override is outside the repository and is not part of the app.
- Export button prepared the JSON blob without a frontend error. Automated browser download capture was unavailable, so a saved browser download was not independently verified. CLI JSON saving passed.

## Publication

Source, runtime weights, notebooks and reports are committed to the original GitHub repository. A free Render web service is prepared with Python 3.11.15, `pip install -r requirements.txt`, and `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, in My Workspace/Singapore. Automatic approval review requires explicit authorization for this project's public deployment. No service was created, and no live URL or hosted test success is claimed yet.
