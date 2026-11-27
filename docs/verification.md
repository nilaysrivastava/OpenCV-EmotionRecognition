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

Live demo: https://expression-atlas-nilay.onrender.com

Render Free, My Workspace, Singapore. Python 3.11.15; build `pip install -r requirements.txt`; start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. The service has no database, persisted image gallery or external model dependency. Auto-deploy is disabled; updates require a manual deploy in Render.

The initial deployment reached `live` on 27 November 2026. Hosted checks verified the exact model SHA-256, evaluation data, public NASA sample prediction, blank-image behavior, invalid/unsupported/empty input handling, and security headers. The public sample returned Happy with score 0.991897 and a measured 711 ms server analysis time in this check. This is one observed request, not a throughput or latency guarantee. See `live-api-checks.json` for the recorded checks. Browser webcam use was tested locally only.

Render Free services sleep after inactivity and share the workspace's free usage limits. No paid compute plan was selected. Source, weights, notebooks, reports and the live URL are committed to the original GitHub repository.
