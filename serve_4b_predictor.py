"""4B chart-image movement predictor — local FastAPI server (REAL model only).

Same HTTP contract as `colab_4b_stock_movement.ipynb` cells 16-18:

  GET  /healthz              liveness, no auth
  GET  /readyz               readiness, no auth (503 until checkpoint is loaded)
  GET  /v1/models            auth required
  POST /v1/predict           auth + multipart (file=<chart image>, text=<optional>)
  POST /v1/chat/completions  auth + OpenAI-compatible JSON (needs image_url data URI)

REAL-ONLY rules (per operator request — no demo/heuristic predictions):
  - torch + torchvision must be installed.
  - A trained checkpoint from the Colab notebook must exist. Search order:
      1. $MODEL_CHECKPOINT env var
      2. ./4b_stock_movement_outputs/best_4b_chart_model.pt
      3. ./best_4b_chart_model.pt
  - Until both hold, /readyz returns 503 and every prediction endpoint
    returns 503 with the exact reason (no fake probabilities ever).
  - The movement probability comes from CHART PIXELS ONLY (ResNet18 trained
    exclusively on the 4B chart-image collection). The `text` field is an
    instruction/response control, not a learned market feature.

Get the checkpoint: run colab_4b_stock_movement.ipynb on a GPU runtime with
MyDrive/4b_chart_images/, then copy
  MyDrive/4b_stock_movement_outputs/best_4b_chart_model.pt
to local
  ./4b_stock_movement_outputs/best_4b_chart_model.pt
and restart this server.

Auth:  Authorization: Bearer <PREDICTION_API_KEY>
  Key source: $PREDICTION_API_KEY env var, else the PREDICTION_API_KEY=
  line in .env.4b (created by start_4b_tunnel.py). Keep it in your website
  backend / secret store — never in frontend JavaScript.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import secrets
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

MODEL_ID = "4b-stock-movement-resnet18"
CLASS_NAMES = ("down", "flat", "up")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_TEXT_CHARS = 1_000
PORT = int(os.environ.get("PORT_4B", "8000"))

HERE = Path(__file__).resolve().parent
CHECKPOINT_CANDIDATES = []
_env_ckpt = os.environ.get("MODEL_CHECKPOINT", "").strip()
if _env_ckpt:
    CHECKPOINT_CANDIDATES.append(Path(_env_ckpt).expanduser())
CHECKPOINT_CANDIDATES += [
    HERE / "4b_stock_movement_outputs" / "best_4b_chart_model.pt",
    HERE / "best_4b_chart_model.pt",
]


def _read_dotenv_key(dotenv: Path, name: str) -> str:
    try:
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def get_api_keys() -> list:
    """All accepted bearer keys. Comma-separated to support one key per
    website (e.g. PREDICTION_API_KEY=mainkey,tsxsitekey) so a site's key
    can be revoked without touching the others."""
    raw = os.environ.get("PREDICTION_API_KEY", "").strip()
    if not raw:
        raw = _read_dotenv_key(HERE / ".env.4b", "PREDICTION_API_KEY")
    if not raw:
        raw = _read_dotenv_key(HERE / ".4b_api_key", "PREDICTION_API_KEY")
    if not raw:
        try:
            single = (HERE / ".4b_api_key").read_text().strip()
            if single and "=" not in single and "\n" not in single:
                raw = single
        except OSError:
            pass
    return [k.strip() for k in raw.split(",") if k.strip()]


PREDICTION_API_KEYS = get_api_keys()
PREDICTION_API_KEY = PREDICTION_API_KEYS[0] if PREDICTION_API_KEYS else ""

PREDICTION_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("PREDICTION_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]


# ------------------------------------------------------------- real model ---
_real_model = None
_eval_transform = None
_torch_device = "cpu"
CHECKPOINT_PATH: Optional[Path] = None
NOT_READY_REASON = "predictor not initialized"

for _candidate in CHECKPOINT_CANDIDATES:
    if _candidate is not None and _candidate.is_file():
        CHECKPOINT_PATH = _candidate
        break

if CHECKPOINT_PATH is None:
    searched = ", ".join(str(c) for c in CHECKPOINT_CANDIDATES)
    NOT_READY_REASON = (
        "Trained checkpoint not found. Run colab_4b_stock_movement.ipynb "
        "(GPU, MyDrive/4b_chart_images/) then copy "
        "MyDrive/4b_stock_movement_outputs/best_4b_chart_model.pt to "
        "./4b_stock_movement_outputs/best_4b_chart_model.pt and restart. "
        f"Searched: {searched}"
    )
    print(f"[4b] {NOT_READY_REASON}", flush=True)
else:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torchvision import models, transforms as T  # type: ignore

        _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        _ckpt = torch.load(str(CHECKPOINT_PATH), map_location=_torch_device)
        _names = tuple(_ckpt.get("class_names", CLASS_NAMES))
        if tuple(_names) != tuple(CLASS_NAMES):
            raise ValueError(f"Checkpoint classes {_names} != {CLASS_NAMES}")
        _mean = tuple(_ckpt.get("mean", (0.485, 0.456, 0.406)))
        _std = tuple(_ckpt.get("std", (0.229, 0.224, 0.225)))
        _size = int(_ckpt.get("image_size", 224))
        _backbone = models.resnet18(weights=None)
        _backbone.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(_backbone.fc.in_features, len(CLASS_NAMES)),
        )
        _backbone.load_state_dict(_ckpt["state_dict"])
        _backbone.to(_torch_device).eval()
        _real_model = _backbone
        _eval_transform = T.Compose(
            [
                T.Resize((_size, _size)),
                T.ToTensor(),
                T.Normalize(_mean, _std),
            ]
        )
        NOT_READY_REASON = ""
        print(f"[4b] REAL model loaded: {CHECKPOINT_PATH} (device={_torch_device})", flush=True)
    except ImportError as exc:
        NOT_READY_REASON = (
            f"torch/torchvision is not installed ({exc}). "
            "Install the CPU wheels: python -m pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        print(f"[4b] {NOT_READY_REASON}", flush=True)
    except Exception as exc:
        NOT_READY_REASON = f"Checkpoint {CHECKPOINT_PATH} could not be loaded: {exc}"
        print(f"[4b] {NOT_READY_REASON}", flush=True)

MODEL_READY = _real_model is not None and _eval_transform is not None


def _text_response(text_input: str, probabilities: Dict[str, float]) -> str:
    request = (text_input or "").strip() or "Predict the next movement."
    top = max(probabilities, key=probabilities.get)
    lines = [
        f"Request: {request}",
        f"Image-only movement estimate: {top.upper()}",
        f"Image-only model confidence: {probabilities[top]:.2%}",
        "This is a statistical chart-image estimate, not financial advice or a guarantee.",
    ]
    low = request.lower()
    if "probabilit" in low or "confidence" in low or "detail" in low:
        lines.append(
            "Probabilities: " + ", ".join(f"{k}={v:.2%}" for k, v in probabilities.items())
        )
    return "\n".join(lines)


def predict_chart_bytes(image_bytes: bytes, text_input: str = "") -> dict:
    """REAL inference from chart pixels only. Never fakes a result."""
    if not MODEL_READY or _real_model is None or _eval_transform is None:
        raise HTTPException(status_code=503, detail=NOT_READY_REASON or "Model is not ready")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB limit")
    try:
        with Image.open(io.BytesIO(image_bytes)) as raw:
            w, h = raw.size
            if w * h > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image has too many pixels")
            raw.load()
            image = raw.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=415, detail="Uploaded file is not a readable image") from exc

    import torch  # type: ignore

    tensor = _eval_transform(image).unsqueeze(0).to(_torch_device)
    with torch.inference_mode():
        probs = torch.softmax(_real_model(tensor), dim=1)[0].cpu().numpy()
    order = sorted(range(len(CLASS_NAMES)), key=lambda j: -probs[j])
    ordered = {CLASS_NAMES[i]: float(probs[i]) for i in order}

    return {
        "model": MODEL_ID,
        "predicted_movement": max(ordered, key=ordered.get),
        "probabilities": ordered,
        "text_request": (text_input or "").strip(),
        "text_response": _text_response(text_input, ordered),
        "disclaimer": "Statistical chart-image estimate only; not financial advice or a guarantee.",
    }


# ------------------------------------------------------------------ app ---
class ContentLengthLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "POST" and request.url.path == "/v1/predict":
            try:
                raw = request.headers.get("content-length")
                if raw and int(raw) > MAX_IMAGE_BYTES + 2 * 1024 * 1024:
                    return JSONResponse({"detail": "Request body is too large"}, status_code=413)
            except ValueError:
                pass
        return await call_next(request)


app = FastAPI(title="4B Stock Movement API", version="1.0.0")
app.add_middleware(ContentLengthLimitMiddleware)
if PREDICTION_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=PREDICTION_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

_gate = asyncio.Semaphore(1)


def require_api_key(authorization: Optional[str]) -> None:
    if not PREDICTION_API_KEYS or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    presented = authorization[len("Bearer "):]
    if not any(secrets.compare_digest(presented, k) for k in PREDICTION_API_KEYS):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


async def run_prediction(image_bytes: bytes, text_input: str) -> dict:
    try:
        await asyncio.wait_for(_gate.acquire(), timeout=0.1)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429, detail="Predictor is busy; retry shortly", headers={"Retry-After": "1"}
        ) from exc
    try:
        return await asyncio.to_thread(predict_chart_bytes, image_bytes, text_input)
    finally:
        _gate.release()


def extract_openai_image_and_text(body: dict):
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            raise HTTPException(
                status_code=400,
                detail="chat completions requires an image_url data URI in user content",
            )
        if not isinstance(content, list):
            raise HTTPException(status_code=400, detail="user content must be text or content parts")
        texts, image_bytes = [], None
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                iu = part.get("image_url") or {}
                url = iu.get("url", "") if isinstance(iu, dict) else ""
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise HTTPException(status_code=400, detail="Only base64 image data URLs are supported")
                try:
                    image_bytes = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc
        if image_bytes is None:
            raise HTTPException(status_code=400, detail="No image was supplied in user content")
        return image_bytes, "\n".join(texts)[:MAX_TEXT_CHARS]
    raise HTTPException(status_code=400, detail="At least one user message is required")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "4b-stock-movement-api"}


@app.get("/readyz")
def readyz():
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail=NOT_READY_REASON or "Model checkpoint is not loaded")
    return {"status": "ready", "model": MODEL_ID, "checkpoint": str(CHECKPOINT_PATH), "device": _torch_device}


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    require_api_key(authorization)
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail=NOT_READY_REASON or "Model is not ready")
    return {"data": [{"id": MODEL_ID, "object": "model", "owned_by": "4b-chart-project"}]}


@app.post("/v1/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    text: str = Form(default="Predict the next movement and show confidence."),
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(authorization)
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="Text request is too long")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="file must be an image")
    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    return await run_prediction(image_bytes, text)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None)):
    require_api_key(authorization)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
    image_bytes, text = extract_openai_image_and_text(body)
    result = await run_prediction(image_bytes, text)
    rid = f"4b-chat-{int(time.time() * 1000)}"
    if body.get("stream", False):
        def stream_response():
            chunk = {
                "id": rid,
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": result["text_response"]},
                        "finish_reason": None,
                    }
                ],
            }
            yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text_response"]},
                "finish_reason": "stop",
            }
        ],
        "prediction": result,
    }


if __name__ == "__main__":
    if not PREDICTION_API_KEY:
        raise SystemExit(
            "PREDICTION_API_KEY is missing.\n"
            "  $env:PREDICTION_API_KEY = 'a-long-random-secret'\n"
            "or run:  python start_4b_tunnel.py   (it creates .env.4b for you)\n"
            "then:    python serve_4b_predictor.py"
        )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
