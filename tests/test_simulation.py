"""Unit tests for policyforge/simulation/.

These tests run without LIBERO, MuJoCo, or a GPU.
They cover: rendering setup, observation formatting, episode result
structure, and video recording utilities.

Run with: pytest tests/test_simulation.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── env.py tests ─────────────────────────────────────────────────────────────

class TestSetupRendering:
    def test_headless_sets_egl(self):
        # Clear any existing values so setdefault works
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)

        from policyforge.simulation.env import setup_rendering
        setup_rendering(headless=True)

        assert os.environ.get("MUJOCO_GL") == "egl"
        assert os.environ.get("PYOPENGL_PLATFORM") == "egl"

    def test_display_sets_glfw(self):
        os.environ.pop("MUJOCO_GL", None)

        from policyforge.simulation.env import setup_rendering
        setup_rendering(headless=False)

        assert os.environ.get("MUJOCO_GL") == "glfw"

    def test_headless_does_not_overwrite_existing_value(self):
        os.environ["MUJOCO_GL"] = "osmesa"

        from policyforge.simulation.env import setup_rendering
        setup_rendering(headless=True)          # setdefault — should not overwrite

        assert os.environ["MUJOCO_GL"] == "osmesa"
        os.environ.pop("MUJOCO_GL")             # cleanup

    def test_display_always_overwrites(self):
        os.environ["MUJOCO_GL"] = "egl"        # was headless before

        from policyforge.simulation.env import setup_rendering
        setup_rendering(headless=False)         # should overwrite to glfw

        assert os.environ["MUJOCO_GL"] == "glfw"
        os.environ.pop("MUJOCO_GL")


# ── rollout.py tests ──────────────────────────────────────────────────────────

class TestEpisodeResult:
    def test_episode_result_fields(self):
        from policyforge.simulation.rollout import EpisodeResult

        result = EpisodeResult(
            task_name="libero_spatial",
            task_language="pick up the red block",
            episode_idx=0,
            success=True,
            steps_taken=42,
            duration_seconds=3.7,
            frames=[],
        )
        assert result.success is True
        assert result.steps_taken == 42
        assert result.task_language == "pick up the red block"

    def test_episode_result_default_frames_is_empty_list(self):
        from policyforge.simulation.rollout import EpisodeResult

        r = EpisodeResult("t", "lang", 0, False, 10, 1.0)
        assert r.frames == []


class TestFormatObs:
    """Test format_obs without GPU — patches torch.Tensor.cuda to be a no-op.

    These tests are skipped automatically when torch is not installed.
    On your dev machine (with torch installed via make setup) they run fully.
    """

    @pytest.fixture(autouse=True)
    def patch_cuda(self, monkeypatch):
        torch = pytest.importorskip("torch", reason="torch not installed — skipping")
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    def test_output_keys(self):
        from policyforge.simulation.rollout import format_obs

        obs = {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": np.zeros(2),
        }
        result = format_obs(obs, task_language="pick up the block")

        assert "observation.images.top" in result
        assert "observation.state" in result
        assert "task" in result

    def test_image_tensor_shape(self):
        from policyforge.simulation.rollout import format_obs

        obs = {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        }
        result = format_obs(obs, task_language="test")
        image = result["observation.images.top"]

        assert image.shape == (1, 3, 128, 128), f"Expected (1,3,128,128), got {image.shape}"

    def test_image_normalized_to_0_1(self):
        import torch
        from policyforge.simulation.rollout import format_obs

        obs = {
            "agentview_image": np.full((64, 64, 3), fill_value=255, dtype=np.uint8),
        }
        result = format_obs(obs, task_language="test")
        image = result["observation.images.top"]

        assert torch.allclose(image, torch.ones_like(image)), "255 should normalize to 1.0"

    def test_task_language_in_output(self):
        from policyforge.simulation.rollout import format_obs

        obs = {"agentview_image": np.zeros((64, 64, 3), dtype=np.uint8)}
        result = format_obs(obs, task_language="stack the cubes")

        assert result["task"] == ["stack the cubes"]

    def test_missing_state_keys_falls_back_to_zeros(self):
        from policyforge.simulation.rollout import format_obs

        # Only provide image, no robot state keys
        obs = {"agentview_image": np.zeros((64, 64, 3), dtype=np.uint8)}
        result = format_obs(obs, task_language="test")

        state = result["observation.state"]
        assert state.shape[0] == 1               # batch dim
        assert (state.numpy() == 0).all()        # all zeros fallback


# ── recorder.py tests ────────────────────────────────────────────────────────

class TestRecorder:
    @pytest.fixture
    def dummy_frames(self):
        """10 frames of 64×64 random RGB."""
        rng = np.random.default_rng(42)
        return [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]

    def test_save_mp4_creates_file(self, dummy_frames, tmp_path):
        from policyforge.simulation.recorder import save_mp4

        out = tmp_path / "test.mp4"
        result = save_mp4(dummy_frames, out, fps=10)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_gif_creates_file(self, dummy_frames, tmp_path):
        from policyforge.simulation.recorder import save_gif

        out = tmp_path / "test.gif"
        result = save_gif(dummy_frames, out, fps=5)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_mp4_creates_parent_dirs(self, dummy_frames, tmp_path):
        from policyforge.simulation.recorder import save_mp4

        deep_path = tmp_path / "a" / "b" / "c" / "out.mp4"
        save_mp4(dummy_frames, deep_path, fps=10)

        assert deep_path.exists()

    def test_save_comparison_mp4_equal_length(self, dummy_frames, tmp_path):
        from policyforge.simulation.recorder import save_comparison_mp4

        out = tmp_path / "comparison.mp4"
        save_comparison_mp4(dummy_frames, dummy_frames, out, fps=10)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_comparison_pads_shorter_sequence(self, tmp_path):
        from policyforge.simulation.recorder import save_comparison_mp4

        rng = np.random.default_rng(0)
        short = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        long_ = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(8)]

        out = tmp_path / "padded.mp4"
        save_comparison_mp4(short, long_, out, fps=5)  # should not raise

        assert out.exists()

    def test_comparison_raises_on_empty_frames(self, tmp_path):
        from policyforge.simulation.recorder import save_comparison_mp4

        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

        with pytest.raises(ValueError, match="non-empty"):
            save_comparison_mp4([], frames, tmp_path / "out.mp4")

    def test_frames_to_gif_bytes_returns_bytes(self, dummy_frames):
        from policyforge.simulation.recorder import frames_to_gif_bytes

        result = frames_to_gif_bytes(dummy_frames, fps=5)

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:6] == b"GIF89a"         # GIF magic bytes