"""WandB tracker implementation."""

from __future__ import annotations

from typing import Any

from policyforge.tracking.base import Tracker


class WandBTracker(Tracker):
    """Logs to Weights & Biases.

    Requires WANDB_API_KEY in your environment (set in .env).
    Creates a new run named "<run_name>" inside project "<project>".
    """

    def __init__(
        self,
        project: str = "policyforge",
        run_name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        import wandb  # lazy import — only needed when this backend is selected

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name or None,
            config=config or {},
            tags=tags or [],
            **kwargs,
        )
        print(f"[tracker/wandb] Run: {self._run.url}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        self._wandb.config.update(params, allow_val_change=True)

    def log_artifact(self, local_path: str, artifact_name: str | None = None) -> None:
        name = artifact_name or local_path.split("/")[-1]
        artifact = self._wandb.Artifact(name=name, type="file")
        artifact.add_file(local_path)
        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        self._wandb.finish()
        print("[tracker/wandb] Run finished.")