"""MLflow tracker implementation.

Start a local MLflow server with:
    mlflow server --host 0.0.0.0 --port 5000

Then set in .env:
    MLFLOW_TRACKING_URI=http://localhost:5000
"""

from __future__ import annotations

from typing import Any

from policyforge.tracking.base import Tracker


class MLflowTracker(Tracker):
    """Logs to a self-hosted MLflow tracking server."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "policyforge",
        run_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        import mlflow  # lazy import

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run_ctx = mlflow.start_run(run_name=run_name)
        self._run_ctx.__enter__()
        print(f"[tracker/mlflow] Server: {tracking_uri}  Experiment: {experiment_name}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow requires string values, max 500 chars per value
        safe = {str(k): str(v)[:500] for k, v in params.items()}
        self._mlflow.log_params(safe)

    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        self._mlflow.log_artifact(local_path)

    def finish(self) -> None:
        self._run_ctx.__exit__(None, None, None)
        print("[tracker/mlflow] Run finished.")