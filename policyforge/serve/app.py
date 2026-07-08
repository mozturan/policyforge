"""PolicyForge inference server.

Exposes a trained SmolVLA policy as a REST API.

Start with:
    make serve CKPT=outputs/train/smolvla_libero_lora
    # or directly:
    CHECKPOINT_PATH=outputs/train/smolvla_libero_lora \\
        uvicorn policyforge.serve.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /         → redirects to /docs (Swagger UI)
    GET  /health   → server status and model info
    POST /predict  → image + instruction → predicted actions
"""

from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image

from policyforge.serve.loader import _cache, clear_cache, get_checkpoint_path, get_policy
from policyforge.serve.schemas import ActionRequest, ActionResponse, HealthResponse

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the policy at startup so the first request isn't slow."""
    checkpoint = get_checkpoint_path()
    if checkpoint:
        print(f"[serve] Loading policy at startup: {checkpoint}")
        try:
            get_policy(checkpoint)
            print("[serve] Policy ready. Listening for requests.")
        except Exception as exc:
            # Don't crash the server — /health will report model_loaded=False
            print(f"[serve] Warning: could not load policy at startup: {exc}")
    else:
        print("[serve] No CHECKPOINT_PATH set. Policy will not be loaded.")
        print("[serve] Set CHECKPOINT_PATH and restart, or the /predict endpoint will return 503.")
    yield
    # Clean up on shutdown
    clear_cache()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PolicyForge Inference API",
    description=(
        "Serve trained VLA robot policies as a REST API.\n\n"
        "Send a robot observation image and a task instruction, "
        "receive predicted robot actions.\n\n"
        "Configure with the `CHECKPOINT_PATH` environment variable."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health():
    """Return server status and whether a policy is loaded in memory."""
    return HealthResponse(
        status="ok",
        model_loaded=bool(_cache),
        checkpoint=get_checkpoint_path(),
        device="cuda" if _cuda_available() else "cpu",
    )


@app.post("/predict", response_model=ActionResponse, tags=["inference"])
async def predict(request: ActionRequest):
    """Run policy inference and return a predicted action chunk.

    The policy is loaded lazily on the first call and cached for all
    subsequent requests. On an RTX 3060, typical inference latency is
    30–60 ms per call.

    Args (request body):
        image_base64: Base64-encoded RGB image (JPEG or PNG).
        instruction:  Natural language task description.
        state:        Optional robot proprioception vector.

    Returns:
        actions:      Predicted action sequence (horizon × action_dim).
        inference_ms: Server-side time for this request.
    """
    checkpoint = get_checkpoint_path()
    if not checkpoint:
        raise HTTPException(
            status_code=503,
            detail=(
                "No policy loaded. "
                "Set the CHECKPOINT_PATH environment variable and restart the server."
            ),
        )

    t0 = time.perf_counter()

    # ── Decode and preprocess image ───────────────────────────────────────────
    image_np = _decode_image(request.image_base64)   # (224, 224, 3) uint8

    # ── Build observation tensors ─────────────────────────────────────────────
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="torch is not installed on the server."
        ) from e

    use_cuda = _cuda_available()

    image_t = (
        torch.from_numpy(image_np)
        .float()
        .permute(2, 0, 1)     # (H,W,C) → (C,H,W)
        .unsqueeze(0)          # → (1, C, H, W)
        .div(255.0)
    )

    state_vals = request.state if request.state is not None else [0.0] * 9
    state_t = torch.tensor([state_vals], dtype=torch.float32)

    if use_cuda:
        image_t = image_t.cuda()
        state_t = state_t.cuda()

    obs_dict = {
        "observation.images.top": image_t,
        "observation.state":      state_t,
        "task":                   [request.instruction],
    }

    # ── Run inference ─────────────────────────────────────────────────────────
    policy = get_policy(checkpoint)

    # Reset action chunk buffer so each REST call is stateless.
    # SmolVLA maintains an internal queue; resetting ensures we always
    # get a fresh chunk rather than continuing a previous episode's sequence.
    policy.reset()

    with torch.no_grad():
        action = policy.select_action(obs_dict)

    # ── Format response ───────────────────────────────────────────────────────
    action_np = action.squeeze().cpu().numpy()

    # Normalise to 2-D (horizon, action_dim) regardless of what select_action returns
    if action_np.ndim == 1:
        actions_2d = action_np.reshape(1, -1)
    else:
        actions_2d = action_np

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ActionResponse(
        actions=actions_2d.tolist(),
        action_dim=int(actions_2d.shape[1]),
        horizon=int(actions_2d.shape[0]),
        inference_ms=round(elapsed_ms, 2),
        checkpoint=checkpoint,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(b64_string: str, target_size: int = 224) -> np.ndarray:
    """Decode a base64 image string to a (target_size, target_size, 3) uint8 array."""
    try:
        raw_bytes = base64.b64decode(b64_string)
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        img = img.resize((target_size, target_size), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image: {exc}. "
                   "Provide a valid base64-encoded JPEG or PNG."
        ) from exc


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-untyped]
        return torch.cuda.is_available()
    except ImportError:
        return False