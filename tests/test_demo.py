"""Unit tests for demo/app.py helper functions.

Tests the pure helper functions (plot_actions, format_actions_table,
get_mode, _action_labels) without starting a Gradio server or needing
a model loaded.

Run with: pytest tests/test_demo.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))


# ── Import helpers without launching the Gradio server ────────────────────────
# The demo module creates the Gradio Blocks at import time (which is fine —
# it doesn't start a server until demo.launch() is called).

from app import (  # noqa: E402
    _action_labels,
    format_actions_table,
    get_mode,
    plot_actions,
    predict,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def single_step_actions():
    """(1, 7) action array — single timestep, 7-DOF."""
    return np.array([[0.1, -0.2, 0.3, 0.01, -0.01, 0.0, 0.5]])


@pytest.fixture
def multi_step_actions():
    """(10, 7) action array — 10 timestep chunk."""
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, (10, 7)).astype(np.float32)


@pytest.fixture
def high_dim_actions():
    """(5, 12) action array — more dims than standard labels."""
    rng = np.random.default_rng(1)
    return rng.uniform(-1, 1, (5, 12)).astype(np.float32)


# ── get_mode ──────────────────────────────────────────────────────────────────

class TestGetMode:
    def test_api_mode_when_api_url_set(self, monkeypatch):
        monkeypatch.setenv("POLICYFORGE_API_URL", "http://localhost:8000")
        monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
        monkeypatch.delenv("CHECKPOINT_REPO", raising=False)
        assert get_mode() == "api"

    def test_direct_mode_when_checkpoint_path_set(self, monkeypatch):
        monkeypatch.delenv("POLICYFORGE_API_URL", raising=False)
        monkeypatch.setenv("CHECKPOINT_PATH", "outputs/train/test")
        monkeypatch.delenv("CHECKPOINT_REPO", raising=False)
        assert get_mode() == "direct"

    def test_direct_mode_when_checkpoint_repo_set(self, monkeypatch):
        monkeypatch.delenv("POLICYFORGE_API_URL", raising=False)
        monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
        monkeypatch.setenv("CHECKPOINT_REPO", "user/model")
        assert get_mode() == "direct"

    def test_unconfigured_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("POLICYFORGE_API_URL", raising=False)
        monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
        monkeypatch.delenv("CHECKPOINT_REPO", raising=False)
        assert get_mode() == "unconfigured"

    def test_api_takes_priority_over_checkpoint(self, monkeypatch):
        monkeypatch.setenv("POLICYFORGE_API_URL", "http://localhost:8000")
        monkeypatch.setenv("CHECKPOINT_PATH", "outputs/train/test")
        assert get_mode() == "api"


# ── _action_labels ────────────────────────────────────────────────────────────

class TestActionLabels:
    def test_7_dim_uses_standard_labels(self):
        labels = _action_labels(7)
        assert labels == ["x", "y", "z", "rx", "ry", "rz", "gripper"]

    def test_3_dim_returns_first_three(self):
        labels = _action_labels(3)
        assert labels == ["x", "y", "z"]

    def test_extra_dims_get_numbered(self):
        labels = _action_labels(10)
        assert labels[7] == "dim_7"
        assert labels[9] == "dim_9"
        assert len(labels) == 10

    def test_zero_dim_returns_empty(self):
        assert _action_labels(0) == []


# ── format_actions_table ──────────────────────────────────────────────────────

class TestFormatActionsTable:
    def test_returns_list_of_lists(self, single_step_actions):
        table = format_actions_table(single_step_actions)
        assert isinstance(table, list)
        assert all(isinstance(row, list) for row in table)

    def test_first_row_is_header(self, single_step_actions):
        table = format_actions_table(single_step_actions)
        header = table[0]
        assert header[0] == "step"
        assert "x" in header
        assert "gripper" in header

    def test_data_rows_count(self, multi_step_actions):
        table = format_actions_table(multi_step_actions)
        # +1 for header row
        assert len(table) == multi_step_actions.shape[0] + 1

    def test_step_labels(self, multi_step_actions):
        table = format_actions_table(multi_step_actions)
        assert table[1][0] == "t=0"
        assert table[2][0] == "t=1"
        assert table[-1][0] == f"t={multi_step_actions.shape[0] - 1}"

    def test_values_are_rounded(self, single_step_actions):
        table = format_actions_table(single_step_actions)
        row = table[1]  # first data row
        for val in row[1:]:
            assert isinstance(val, float)
            # Check not more than 4 decimal places
            assert round(val, 4) == val

    def test_extra_dims_in_header(self, high_dim_actions):
        table = format_actions_table(high_dim_actions)
        header = table[0]
        assert "dim_7" in header
        assert len(header) == 1 + high_dim_actions.shape[1]  # "step" + action_dim


# ── plot_actions ──────────────────────────────────────────────────────────────

class TestPlotActions:
    def test_returns_figure(self, single_step_actions):
        import matplotlib.pyplot as plt
        fig = plot_actions(single_step_actions)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_lines(self, multi_step_actions):
        import matplotlib.pyplot as plt
        fig = plot_actions(multi_step_actions)
        ax = fig.axes[0]
        assert len(ax.lines) == multi_step_actions.shape[1]
        plt.close(fig)

    def test_figure_has_legend(self, single_step_actions):
        import matplotlib.pyplot as plt
        fig = plot_actions(single_step_actions)
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_single_timestep_renders(self, single_step_actions):
        import matplotlib.pyplot as plt
        fig = plot_actions(single_step_actions)
        assert fig is not None
        plt.close(fig)

    def test_high_dim_renders(self, high_dim_actions):
        import matplotlib.pyplot as plt
        fig = plot_actions(high_dim_actions)
        ax = fig.axes[0]
        assert len(ax.lines) == high_dim_actions.shape[1]
        plt.close(fig)


# ── predict — validation paths ────────────────────────────────────────────────

class TestPredictValidation:
    """Test the input validation logic in predict() without needing a model."""

    def test_no_image_returns_message(self, monkeypatch):
        monkeypatch.delenv("POLICYFORGE_API_URL", raising=False)
        fig, table, msg, raw = predict(None, "pick up block", "")
        assert fig is None
        assert "image" in msg.lower() or "upload" in msg.lower()

    def test_empty_instruction_returns_message(self, monkeypatch):
        monkeypatch.delenv("POLICYFORGE_API_URL", raising=False)
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        fig, table, msg, raw = predict(dummy_image, "   ", "")
        assert fig is None
        assert "instruction" in msg.lower() or "enter" in msg.lower()

    def test_invalid_state_returns_message(self, monkeypatch):
        monkeypatch.delenv("POLICYFORGE_API_URL", raising=False)
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        fig, table, msg, raw = predict(dummy_image, "pick up block", "not_a_float")
        assert fig is None
        assert "invalid" in msg.lower() or "state" in msg.lower()

    def test_unconfigured_returns_configure_message(self, monkeypatch):
        monkeypatch.delenv("POLICYFORGE_API_URL", raising=False)
        monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
        monkeypatch.delenv("CHECKPOINT_REPO", raising=False)
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        fig, table, msg, raw = predict(dummy_image, "pick up block", "")
        assert fig is None
        assert msg is not None
        # Should mention configuration
        assert "configured" in msg.lower() or "checkpoint" in msg.lower()