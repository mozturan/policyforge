"""Abstract tracker interface.

Switch backends by setting `tracking.backend` in your experiment YAML:

    tracking:
      backend: wandb    # or: mlflow | none
      project: policyforge
      run_name: my_experiment

This controls tracking for PolicyForge's own work (eval results, rollout
metrics, experiment metadata). Training metrics are handled separately by
lerobot-train's built-in WandB integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tracker(ABC):
    """Minimal interface every tracking backend must implement."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics at an optional step."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters or config values (logged once per run)."""

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        """Upload a local file as an artifact (report, video, config YAML)."""

    @abstractmethod
    def finish(self) -> None:
        """Close and flush the run. Always call this when done."""

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *_) -> None:
        self.finish()


class NoOpTracker(Tracker):
    """Silent no-op tracker. Use backend='none' for local runs or testing."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


def get_tracker(
    backend: str,
    project: str = "policyforge",
    run_name: str | None = None,
    **kwargs: Any,
) -> Tracker:
    """Factory: return the right Tracker for the given backend name.

    Args:
        backend:  "wandb" | "mlflow" | "none"
        project:  Project / experiment name (meaning varies per backend)
        run_name: Optional name for this specific run
        **kwargs: Passed through to the backend constructor

    Returns:
        A Tracker instance ready to use.

    Example:
        tracker = get_tracker("wandb", project="policyforge", run_name="eval_run_1")
        tracker = get_tracker("mlflow", tracking_uri="http://localhost:5000")
        tracker = get_tracker("none")
    """
    backend = backend.strip().lower()

    if backend == "wandb":
        from policyforge.tracking.wandb_tracker import WandBTracker
        return WandBTracker(project=project, run_name=run_name, **kwargs)

    if backend == "mlflow":
        from policyforge.tracking.mlflow_tracker import MLflowTracker
        return MLflowTracker(experiment_name=project, run_name=run_name, **kwargs)

    if backend in ("none", "null", "noop", ""):
        return NoOpTracker()

    raise ValueError(
        f"Unknown tracking backend: {backend!r}. "
        "Valid options: 'wandb', 'mlflow', 'none'"
    )