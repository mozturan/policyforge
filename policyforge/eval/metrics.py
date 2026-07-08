"""Evaluation metrics for LIBERO simulation rollouts.

Pure computation — no I/O, no side effects. Every function takes
EpisodeResult objects and returns typed dataclasses. Fully testable
without LIBERO, torch, or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from policyforge.simulation.rollout import EpisodeResult


@dataclass
class TaskMetrics:
    """Metrics for all episodes of a single task."""
    task_language: str
    n_episodes: int
    n_success: int
    success_rate: float          # 0.0 – 1.0
    avg_steps: float
    avg_duration_seconds: float

    @property
    def success_pct(self) -> str:
        """Formatted string for reports, e.g. '75%'."""
        return f"{self.success_rate:.0%}"


@dataclass
class SuiteMetrics:
    """Aggregated metrics across all tasks in one evaluation run."""
    suite_name: str
    checkpoint_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def overall_success_rate(self) -> float:
        """Mean success rate across all tasks."""
        if not self.task_metrics:
            return 0.0
        return sum(t.success_rate for t in self.task_metrics.values()) / len(self.task_metrics)

    @property
    def total_episodes(self) -> int:
        return sum(t.n_episodes for t in self.task_metrics.values())

    @property
    def total_successes(self) -> int:
        return sum(t.n_success for t in self.task_metrics.values())


def compute_task_metrics(
    task_language: str,
    results: list["EpisodeResult"],
) -> TaskMetrics:
    """Compute metrics for a single task from its episode results.

    Args:
        task_language: Natural language description of the task.
        results:       List of EpisodeResult from run_episode() calls.

    Returns:
        TaskMetrics with success rate and step/duration averages.
    """
    if not results:
        return TaskMetrics(
            task_language=task_language,
            n_episodes=0,
            n_success=0,
            success_rate=0.0,
            avg_steps=0.0,
            avg_duration_seconds=0.0,
        )

    n_success = sum(1 for r in results if r.success)
    return TaskMetrics(
        task_language=task_language,
        n_episodes=len(results),
        n_success=n_success,
        success_rate=n_success / len(results),
        avg_steps=sum(r.steps_taken for r in results) / len(results),
        avg_duration_seconds=sum(r.duration_seconds for r in results) / len(results),
    )


def compute_suite_metrics(
    suite_name: str,
    checkpoint_path: str,
    task_results: dict[str, list["EpisodeResult"]],
) -> SuiteMetrics:
    """Compute suite-level metrics from per-task episode results.

    Args:
        suite_name:      LIBERO suite name, e.g. "libero_spatial".
        checkpoint_path: Path to the evaluated checkpoint.
        task_results:    Dict mapping task_language → list of EpisodeResult.

    Returns:
        SuiteMetrics with per-task and overall aggregated metrics.
    """
    suite = SuiteMetrics(suite_name=suite_name, checkpoint_path=checkpoint_path)

    for task_language, results in task_results.items():
        suite.task_metrics[task_language] = compute_task_metrics(task_language, results)

    return suite