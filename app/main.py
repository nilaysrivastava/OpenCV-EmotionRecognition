import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .inference import ROOT, InputError, ModelService


@asynccontextmanager
async def lifespan(app):
    app.state.model = ModelService()
    app.state.slots = asyncio.Semaphore(2)
    yield


app = FastAPI(title="Expression Atlas", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; script-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.model.status()}


@app.get("/api/evaluation")
def evaluation(request: Request):
    model = request.app.state.model
    return {
        "model": model.status(),
        "metrics": model.metrics,
        "comparison": model.comparison,
        "data_audit": model.audit,
    }


@app.post("/api/analyze")
async def analyze(request: Request):
    if request.headers.get("content-type", "").split(";")[0] not in [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]:
        raise HTTPException(415, "Choose a JPEG, PNG or WebP image.")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(413, "Choose an image smaller than 8 MB.")
    if not data:
        raise HTTPException(400, "Choose an image first.")
    async with request.app.state.slots:
        try:
            return await run_in_threadpool(request.app.state.model.analyze, bytes(data))
        except InputError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.get("/")
def home():
    return FileResponse(ROOT / "static/index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
