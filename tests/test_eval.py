"""Unit tests for policyforge/eval/.

Covers metrics computation and report generation entirely without
LIBERO, torch, or a GPU. Uses dummy EpisodeResult objects built
from scratch — no simulation needed.

Run with: pytest tests/test_eval.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from policyforge.eval.metrics import (
    TaskMetrics,
    SuiteMetrics,
    compute_task_metrics,
    compute_suite_metrics,
)
from policyforge.eval.reporter import (
    compare_reports,
    generate_json_report,
    generate_markdown_report,
    save_reports,
)
from policyforge.simulation.rollout import EpisodeResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_result(success: bool, steps: int = 100, duration: float = 2.0) -> EpisodeResult:
    return EpisodeResult(
        task_name="libero_spatial",
        task_language="pick up the red block and place it in the bowl",
        episode_idx=0,
        success=success,
        steps_taken=steps,
        duration_seconds=duration,
        frames=[],
    )


@pytest.fixture
def three_success_two_fail():
    return [
        make_result(True,  steps=80,  duration=1.5),
        make_result(True,  steps=100, duration=2.0),
        make_result(False, steps=400, duration=8.0),
        make_result(True,  steps=90,  duration=1.8),
        make_result(False, steps=400, duration=8.0),
    ]


@pytest.fixture
def all_success():
    return [make_result(True, steps=i * 10 + 50) for i in range(4)]


@pytest.fixture
def all_fail():
    return [make_result(False, steps=400) for _ in range(3)]


@pytest.fixture
def suite_metrics(three_success_two_fail, all_success):
    task_results = {
        "pick up the red block": three_success_two_fail,
        "stack the wooden blocks": all_success,
    }
    return compute_suite_metrics("libero_spatial", "outputs/train/test", task_results)


# ── compute_task_metrics ──────────────────────────────────────────────────────

class TestComputeTaskMetrics:
    def test_success_rate(self, three_success_two_fail):
        tm = compute_task_metrics("task", three_success_two_fail)
        assert tm.success_rate == pytest.approx(3 / 5)

    def test_n_episodes(self, three_success_two_fail):
        tm = compute_task_metrics("task", three_success_two_fail)
        assert tm.n_episodes == 5

    def test_n_success(self, three_success_two_fail):
        tm = compute_task_metrics("task", three_success_two_fail)
        assert tm.n_success == 3

    def test_avg_steps(self, three_success_two_fail):
        tm = compute_task_metrics("task", three_success_two_fail)
        expected_avg = (80 + 100 + 400 + 90 + 400) / 5
        assert tm.avg_steps == pytest.approx(expected_avg)

    def test_all_success(self, all_success):
        tm = compute_task_metrics("task", all_success)
        assert tm.success_rate == 1.0
        assert tm.n_success == 4

    def test_all_fail(self, all_fail):
        tm = compute_task_metrics("task", all_fail)
        assert tm.success_rate == 0.0
        assert tm.n_success == 0

    def test_empty_results(self):
        tm = compute_task_metrics("task", [])
        assert tm.success_rate == 0.0
        assert tm.n_episodes == 0

    def test_success_pct_property(self, three_success_two_fail):
        tm = compute_task_metrics("task", three_success_two_fail)
        assert tm.success_pct == "60%"


# ── compute_suite_metrics ─────────────────────────────────────────────────────

class TestComputeSuiteMetrics:
    def test_overall_success_rate(self, suite_metrics):
        # task1: 60%, task2: 100% → overall mean = 80%
        assert suite_metrics.overall_success_rate == pytest.approx(0.8)

    def test_total_episodes(self, suite_metrics):
        assert suite_metrics.total_episodes == 9   # 5 + 4

    def test_total_successes(self, suite_metrics):
        assert suite_metrics.total_successes == 7  # 3 + 4

    def test_task_keys_match_input(self, suite_metrics):
        assert "pick up the red block" in suite_metrics.task_metrics
        assert "stack the wooden blocks" in suite_metrics.task_metrics

    def test_empty_suite(self):
        m = compute_suite_metrics("libero_spatial", "ckpt", {})
        assert m.overall_success_rate == 0.0
        assert m.total_episodes == 0

    def test_suite_name_stored(self, suite_metrics):
        assert suite_metrics.suite_name == "libero_spatial"

    def test_checkpoint_path_stored(self, suite_metrics):
        assert suite_metrics.checkpoint_path == "outputs/train/test"

    def test_timestamp_is_set(self, suite_metrics):
        assert suite_metrics.timestamp  # non-empty string


# ── generate_markdown_report ──────────────────────────────────────────────────

class TestMarkdownReport:
    def test_contains_header(self, suite_metrics):
        md = generate_markdown_report(suite_metrics)
        assert "## Evaluation report" in md

    def test_contains_suite_name(self, suite_metrics):
        md = generate_markdown_report(suite_metrics)
        assert "libero_spatial" in md

    def test_contains_checkpoint_path(self, suite_metrics):
        md = generate_markdown_report(suite_metrics)
        assert "outputs/train/test" in md

    def test_contains_table_header(self, suite_metrics):
        md = generate_markdown_report(suite_metrics)
        assert "| Task |" in md
        assert "| Success rate |" in md

    def test_contains_overall_row(self, suite_metrics):
        md = generate_markdown_report(suite_metrics)
        assert "**Overall**" in md

    def test_overall_success_rate_shown(self, suite_metrics):
        md = generate_markdown_report(suite_metrics)
        assert "80%" in md

    def test_task_rows_present(self, suite_metrics):
        md = generate_markdown_report(suite_metrics)
        assert "pick up the red block" in md
        assert "stack the wooden blocks" in md

    def test_long_task_name_truncated(self):
        long_name = "a" * 100
        task_results = {long_name: [make_result(True)]}
        m = compute_suite_metrics("libero_spatial", "ckpt", task_results)
        md = generate_markdown_report(m)
        assert "…" in md                         # ellipsis from truncation

    def test_empty_suite_still_renders(self):
        m = compute_suite_metrics("libero_spatial", "ckpt", {})
        md = generate_markdown_report(m)
        assert "## Evaluation report" in md     # should not raise


# ── generate_json_report ──────────────────────────────────────────────────────

class TestJsonReport:
    def test_json_is_valid(self, suite_metrics):
        report = generate_json_report(suite_metrics)
        # Re-serialise and parse — should not raise
        reparsed = json.loads(json.dumps(report))
        assert reparsed["suite"] == "libero_spatial"

    def test_overall_success_rate(self, suite_metrics):
        report = generate_json_report(suite_metrics)
        assert report["overall_success_rate"] == pytest.approx(0.8, abs=0.01)

    def test_tasks_keys_present(self, suite_metrics):
        report = generate_json_report(suite_metrics)
        assert "pick up the red block" in report["tasks"]

    def test_task_success_rate(self, suite_metrics):
        report = generate_json_report(suite_metrics)
        task = report["tasks"]["pick up the red block"]
        assert task["success_rate"] == pytest.approx(0.6, abs=0.01)

    def test_checkpoint_stored(self, suite_metrics):
        report = generate_json_report(suite_metrics)
        assert report["checkpoint"] == "outputs/train/test"


# ── save_reports ──────────────────────────────────────────────────────────────

class TestSaveReports:
    def test_saves_markdown_file(self, suite_metrics, tmp_path):
        paths = save_reports(suite_metrics, tmp_path)
        assert paths["markdown"].exists()
        assert paths["markdown"].suffix == ".md"

    def test_saves_json_file(self, suite_metrics, tmp_path):
        paths = save_reports(suite_metrics, tmp_path)
        assert paths["json"].exists()
        content = json.loads(paths["json"].read_text())
        assert "overall_success_rate" in content

    def test_creates_output_dir(self, suite_metrics, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        save_reports(suite_metrics, deep)
        assert deep.exists()


# ── compare_reports ───────────────────────────────────────────────────────────

class TestCompareReports:
    @pytest.fixture
    def base_report(self, suite_metrics):
        return generate_json_report(suite_metrics)

    @pytest.fixture
    def better_report(self, suite_metrics):
        # Manually bump overall rate for comparison
        report = generate_json_report(suite_metrics)
        report["overall_success_rate"] = 0.95
        return report

    def test_comparison_contains_delta(self, base_report, better_report):
        md = compare_reports(base_report, better_report)
        assert "Delta" in md

    def test_positive_improvement_shown(self, base_report, better_report):
        md = compare_reports(base_report, better_report)
        assert "+" in md

    def test_comparison_has_table(self, base_report, better_report):
        md = compare_reports(base_report, better_report)
        assert "|" in md