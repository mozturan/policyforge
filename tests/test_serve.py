"""Unit tests for policyforge/serve/.

Tests the FastAPI endpoints using TestClient.
No GPU, lerobot, or real policy needed — the loader is monkeypatched
where inference is tested.

Run with: pytest tests/test_serve.py -v
"""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_loader_cache():
    """Wipe the policy cache before each test so tests are isolated."""
    from policyforge.serve.loader import clear_cache
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def client(monkeypatch):
    """TestClient with no CHECKPOINT_PATH set (clean slate)."""
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    from policyforge.serve.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_checkpoint(monkeypatch):
    """TestClient with CHECKPOINT_PATH set to a fake path.

    Also injects a mock policy into the cache so /predict doesn't
    actually try to load a model from disk.
    """
    fake_ckpt = "fake/checkpoint"
    monkeypatch.setenv("CHECKPOINT_PATH", fake_ckpt)

    from policyforge.serve.app import app
    with TestClient(app) as c:
        yield c, fake_ckpt


def make_image_b64(size: int = 64) -> str:
    """Return a base64-encoded JPEG of a random image."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_ok(self, client):
        data = response = client.get("/health").json()
        assert data["status"] == "ok"

    def test_model_not_loaded_when_no_checkpoint(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is False

    def test_checkpoint_null_when_not_set(self, client):
        data = client.get("/health").json()
        assert data["checkpoint"] is None

    def test_device_field_present(self, client):
        data = client.get("/health").json()
        assert data["device"] in ("cuda", "cpu")

    def test_model_loaded_true_when_cached(self, monkeypatch):
        """Manually inject a mock into the cache and check health reflects it."""
        monkeypatch.setenv("CHECKPOINT_PATH", "fake/path")
        import policyforge.serve.loader as loader_mod
        loader_mod._cache["fake/path"] = MagicMock()

        from policyforge.serve.app import app
        with TestClient(app) as c:
            data = c.get("/health").json()
        assert data["model_loaded"] is True


# ── / (root redirect) ─────────────────────────────────────────────────────────

class TestRootRedirect:
    def test_redirects_to_docs(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (301, 302, 307, 308)
        assert "/docs" in response.headers.get("location", "")


# ── /predict — request validation ────────────────────────────────────────────

class TestPredictValidation:
    def test_missing_instruction_returns_422(self, client):
        response = client.post("/predict", json={"image_base64": make_image_b64()})
        assert response.status_code == 422

    def test_missing_image_returns_422(self, client):
        response = client.post("/predict", json={"instruction": "pick up block"})
        assert response.status_code == 422

    def test_invalid_base64_returns_422(self, client_with_checkpoint, monkeypatch):
        client, fake_ckpt = client_with_checkpoint

        # Inject a mock policy so the loader doesn't try to hit disk
        import policyforge.serve.loader as loader_mod
        loader_mod._cache[fake_ckpt] = MagicMock()

        response = client.post("/predict", json={
            "image_base64": "NOT_VALID_BASE64!!!",
            "instruction": "pick up block",
        })
        assert response.status_code == 422

    def test_no_checkpoint_returns_503(self, client):
        response = client.post("/predict", json={
            "image_base64": make_image_b64(),
            "instruction": "pick up block",
        })
        assert response.status_code == 503


# ── /predict — inference (requires torch) ────────────────────────────────────

class TestPredictInference:
    @pytest.fixture(autouse=True)
    def require_torch(self):
        pytest.importorskip("torch", reason="torch not installed — skipping inference tests")

    def test_predict_returns_200(self, client_with_checkpoint, monkeypatch):
        import torch
        client, fake_ckpt = client_with_checkpoint

        mock_policy = MagicMock()
        mock_policy.select_action.return_value = torch.zeros(7)  # 7-dim action

        import policyforge.serve.loader as loader_mod
        loader_mod._cache[fake_ckpt] = mock_policy

        response = client.post("/predict", json={
            "image_base64": make_image_b64(),
            "instruction": "pick up the red block",
        })
        assert response.status_code == 200

    def test_predict_response_shape(self, client_with_checkpoint, monkeypatch):
        import torch
        client, fake_ckpt = client_with_checkpoint

        mock_policy = MagicMock()
        mock_policy.select_action.return_value = torch.zeros(7)

        import policyforge.serve.loader as loader_mod
        loader_mod._cache[fake_ckpt] = mock_policy

        data = client.post("/predict", json={
            "image_base64": make_image_b64(),
            "instruction": "pick up block",
        }).json()

        assert "actions" in data
        assert data["action_dim"] == 7
        assert data["horizon"] == 1        # single action reshaped to (1, 7)
        assert len(data["actions"]) == 1
        assert len(data["actions"][0]) == 7

    def test_predict_multi_step_action(self, client_with_checkpoint, monkeypatch):
        """Policy returning (horizon, action_dim) should be passed through."""
        import torch
        client, fake_ckpt = client_with_checkpoint

        horizon, action_dim = 10, 7
        mock_policy = MagicMock()
        mock_policy.select_action.return_value = torch.zeros(horizon, action_dim)

        import policyforge.serve.loader as loader_mod
        loader_mod._cache[fake_ckpt] = mock_policy

        data = client.post("/predict", json={
            "image_base64": make_image_b64(),
            "instruction": "pick up block",
        }).json()

        assert data["horizon"] == horizon
        assert data["action_dim"] == action_dim

    def test_predict_inference_ms_positive(self, client_with_checkpoint, monkeypatch):
        import torch
        client, fake_ckpt = client_with_checkpoint

        mock_policy = MagicMock()
        mock_policy.select_action.return_value = torch.zeros(7)

        import policyforge.serve.loader as loader_mod
        loader_mod._cache[fake_ckpt] = mock_policy

        data = client.post("/predict", json={
            "image_base64": make_image_b64(),
            "instruction": "pick up block",
        }).json()

        assert data["inference_ms"] > 0

    def test_predict_with_optional_state(self, client_with_checkpoint, monkeypatch):
        import torch
        client, fake_ckpt = client_with_checkpoint

        mock_policy = MagicMock()
        mock_policy.select_action.return_value = torch.zeros(7)

        import policyforge.serve.loader as loader_mod
        loader_mod._cache[fake_ckpt] = mock_policy

        response = client.post("/predict", json={
            "image_base64": make_image_b64(),
            "instruction": "pick up block",
            "state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        })
        assert response.status_code == 200

    def test_policy_reset_called_per_request(self, client_with_checkpoint, monkeypatch):
        """Each /predict call must reset the policy's action chunk buffer."""
        import torch
        client, fake_ckpt = client_with_checkpoint

        mock_policy = MagicMock()
        mock_policy.select_action.return_value = torch.zeros(7)

        import policyforge.serve.loader as loader_mod
        loader_mod._cache[fake_ckpt] = mock_policy

        payload = {"image_base64": make_image_b64(), "instruction": "pick up block"}
        client.post("/predict", json=payload)
        client.post("/predict", json=payload)

        assert mock_policy.reset.call_count == 2


# ── schemas ───────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_action_request_requires_image(self):
        from pydantic import ValidationError
        from policyforge.serve.schemas import ActionRequest
        with pytest.raises(ValidationError):
            ActionRequest(instruction="pick up block")

    def test_action_request_requires_instruction(self):
        from pydantic import ValidationError
        from policyforge.serve.schemas import ActionRequest
        with pytest.raises(ValidationError):
            ActionRequest(image_base64="abc")

    def test_action_request_state_optional(self):
        from policyforge.serve.schemas import ActionRequest
        req = ActionRequest(image_base64="abc", instruction="test")
        assert req.state is None

    def test_health_response_fields(self):
        from policyforge.serve.schemas import HealthResponse
        h = HealthResponse(status="ok", model_loaded=False, device="cpu")
        assert h.status == "ok"
        assert h.checkpoint is None