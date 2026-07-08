"""Evaluation report generation.

Converts SuiteMetrics into human-readable Markdown (for README / PR comments)
and machine-readable JSON (for CI comparisons and dashboards).

Pure I/O — no ML, no simulation. Fully testable without any ML dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

from policyforge.eval.metrics import SuiteMetrics, TaskMetrics


def generate_markdown_report(metrics: SuiteMetrics) -> str:
    """Generate a Markdown evaluation report ready to paste into a README.

    Output format:
        # Evaluation report
        **Checkpoint:** ...
        **Suite:** ...
        | Task | Success | Avg steps | Episodes |
        | ...  | ...     | ...       | ...      |
        | Overall | 72% | 145 | 40 |

    Args:
        metrics: SuiteMetrics from compute_suite_metrics().

    Returns:
        Markdown string. No trailing newline.
    """
    lines: list[str] = [
        "## Evaluation report",
        "",
        f"**Checkpoint:** `{metrics.checkpoint_path}`  ",
        f"**Suite:** {metrics.suite_name}  ",
        f"**Date:** {metrics.timestamp}  ",
        f"**Overall success rate:** {metrics.overall_success_rate:.0%}"
        f" ({metrics.total_successes}/{metrics.total_episodes})",
        "",
        "### Per-task results",
        "",
        "| Task | Success rate | Avg steps | Avg duration | Episodes |",
        "|------|-------------|-----------|--------------|----------|",
    ]

    for task_m in metrics.task_metrics.values():
        # Truncate long task descriptions for table readability
        label = _truncate(task_m.task_language, max_chars=55)
        lines.append(
            f"| {label} "
            f"| {task_m.success_pct} "
            f"| {task_m.avg_steps:.0f} "
            f"| {task_m.avg_duration_seconds:.1f}s "
            f"| {task_m.n_episodes} |"
        )

    # Summary row
    if metrics.task_metrics:
        avg_steps_overall = (
            sum(t.avg_steps for t in metrics.task_metrics.values())
            / len(metrics.task_metrics)
        )
        avg_dur_overall = (
            sum(t.avg_duration_seconds for t in metrics.task_metrics.values())
            / len(metrics.task_metrics)
        )
        lines += [
            f"| **Overall** "
            f"| **{metrics.overall_success_rate:.0%}** "
            f"| {avg_steps_overall:.0f} "
            f"| {avg_dur_overall:.1f}s "
            f"| {metrics.total_episodes} |",
        ]

    return "\n".join(lines)


def generate_json_report(metrics: SuiteMetrics) -> dict:
    """Generate a machine-readable report dict from SuiteMetrics.

    Suitable for saving as JSON, posting to a CI system, or
    comparing across runs programmatically.

    Args:
        metrics: SuiteMetrics from compute_suite_metrics().

    Returns:
        Plain dict (JSON-serialisable).
    """
    return {
        "checkpoint": metrics.checkpoint_path,
        "suite": metrics.suite_name,
        "timestamp": metrics.timestamp,
        "overall_success_rate": round(metrics.overall_success_rate, 4),
        "total_episodes": metrics.total_episodes,
        "total_successes": metrics.total_successes,
        "tasks": {
            lang: _task_to_dict(tm)
            for lang, tm in metrics.task_metrics.items()
        },
    }


def save_reports(metrics: SuiteMetrics, output_dir: str | Path) -> dict[str, Path]:
    """Save both Markdown and JSON reports to disk.

    Args:
        metrics:    SuiteMetrics to serialise.
        output_dir: Directory to write files into (created if missing).

    Returns:
        Dict with keys "markdown" and "json" pointing to saved paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path   = out / "eval_report.md"
    json_path = out / "eval_report.json"

    md_path.write_text(generate_markdown_report(metrics))
    json_path.write_text(json.dumps(generate_json_report(metrics), indent=2))

    return {"markdown": md_path, "json": json_path}


def compare_reports(baseline: dict, candidate: dict) -> str:
    """Generate a Markdown diff table between two JSON reports.

    Useful for CI — post this as a PR comment to show whether
    the new checkpoint improved over the previous best.

    Args:
        baseline:  JSON report dict for the baseline (e.g. base model).
        candidate: JSON report dict for the candidate (e.g. fine-tuned).

    Returns:
        Markdown comparison table.
    """
    base_rate  = baseline.get("overall_success_rate", 0)
    cand_rate  = candidate.get("overall_success_rate", 0)
    delta      = cand_rate - base_rate
    direction  = "+" if delta >= 0 else ""

    lines: list[str] = [
        "### Checkpoint comparison",
        "",
        f"| | Baseline | Candidate | Delta |",
        f"|---|---|---|---|",
        f"| Overall success rate "
        f"| {base_rate:.0%} "
        f"| {cand_rate:.0%} "
        f"| {direction}{delta:.0%} |",
        "",
    ]

    # Per-task comparison
    base_tasks = baseline.get("tasks", {})
    cand_tasks = candidate.get("tasks", {})
    all_tasks  = sorted(set(base_tasks) | set(cand_tasks))

    if all_tasks:
        lines += [
            "| Task | Baseline | Candidate | Delta |",
            "|------|---------|-----------|-------|",
        ]
        for task in all_tasks:
            b = base_tasks.get(task, {}).get("success_rate", 0)
            c = cand_tasks.get(task, {}).get("success_rate", 0)
            d = c - b
            sign = "+" if d >= 0 else ""
            label = _truncate(task, 45)
            lines.append(f"| {label} | {b:.0%} | {c:.0%} | {sign}{d:.0%} |")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _task_to_dict(tm: TaskMetrics) -> dict:
    return {
        "success_rate":          round(tm.success_rate, 4),
        "n_episodes":            tm.n_episodes,
        "n_success":             tm.n_success,
        "avg_steps":             round(tm.avg_steps, 1),
        "avg_duration_seconds":  round(tm.avg_duration_seconds, 2),
    }


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"