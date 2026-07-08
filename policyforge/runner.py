"""Runner — the entire training contribution of PolicyForge.

Reads an experiment YAML, builds the correct lerobot-train CLI command,
and runs it as a subprocess. Does NOT reimplement training, LoRA, or any
part of lerobot. It is a thin, typed config-to-CLI translation layer.

Usage:
    from policyforge.runner import run_experiment
    run_experiment("configs/experiments/smolvla_libero_lora.yaml")
    run_experiment("configs/experiments/smolvla_libero_lora.yaml", dry_run=True)
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ── Config dataclasses ────────────────────────────────────────────────────────
# Each dataclass maps to one section of the experiment YAML.
# All are optional except the fields on ExperimentConfig marked as required.

@dataclass
class PeftConfig:
    """LoRA / PEFT settings. Remove the `peft:` block entirely for full fine-tuning."""
    method_type: str = "LORA"
    r: int = 64


@dataclass
class EnvConfig:
    """LIBERO simulation environment used by lerobot-train's own eval loop."""
    type: str = "libero"
    task: str = "libero_spatial"


@dataclass
class HubConfig:
    """HuggingFace Hub push settings. Leave repo_id empty to skip."""
    repo_id: str = ""


@dataclass
class WandBConfig:
    """WandB settings passed to lerobot-train for training metric logging."""
    enable: bool = True
    project: str = "policyforge"


@dataclass
class TrackingConfig:
    """PolicyForge's own tracking: eval results, rollout metrics, metadata.
    Separate from lerobot-train's WandB tracking above.

    backend: "wandb" | "mlflow" | "none"
    """
    backend: str = "wandb"
    project: str = "policyforge"
    run_name: str = ""


@dataclass
class SimulationConfig:
    """Controls both scripts/simulate.py and policyforge/eval/ rollouts.

    headless: true  → EGL offscreen rendering, saves video to disk (any machine)
    headless: false → GLFW windowed rendering, opens a live window (needs display)
    """
    task_suite: str = "libero_spatial"
    n_episodes: int = 20
    max_steps: int = 400
    render_size: int = 256
    fps: int = 15
    record_video: bool = True
    headless: bool = True          # flip to false to watch live in a window
    output_dir: str = ""           # defaults to <checkpoint>/rollouts/ if empty


@dataclass
class ExperimentConfig:
    """Full experiment config. Parsed from a YAML file via ExperimentConfig.from_yaml()."""

    # ── Required ──────────────────────────────────────────────────────────────
    policy_path: str               # e.g. "lerobot/smolvla_base"
    dataset_repo_id: str           # e.g. "HuggingFaceVLA/libero"
    output_dir: str                # e.g. "outputs/train/smolvla_libero_lora"

    # ── Training ──────────────────────────────────────────────────────────────
    job_name: str = "policyforge_run"
    steps: int = 100_000
    batch_size: int = 32
    policy_optimizer_lr: float = 1e-3
    policy_scheduler_decay_lr: float = 1e-4

    # ── Optional blocks ───────────────────────────────────────────────────────
    peft: Optional[PeftConfig] = None          # None = full fine-tuning
    env: Optional[EnvConfig] = None            # None = no env eval during training
    hub: HubConfig = field(default_factory=HubConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an experiment YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config not found: {path}")

        raw = yaml.safe_load(path.read_text())

        # Validate required fields are present
        for required in ("policy_path", "dataset_repo_id", "output_dir"):
            if required not in raw:
                raise ValueError(f"Missing required field '{required}' in {path}")

        cfg = cls(
            policy_path=raw["policy_path"],
            dataset_repo_id=raw["dataset_repo_id"],
            output_dir=raw["output_dir"],
            job_name=raw.get("job_name", "policyforge_run"),
            steps=raw.get("steps", 100_000),
            batch_size=raw.get("batch_size", 32),
            policy_optimizer_lr=raw.get("policy_optimizer_lr", 1e-3),
            policy_scheduler_decay_lr=raw.get("policy_scheduler_decay_lr", 1e-4),
        )

        if raw.get("peft"):
            cfg.peft = PeftConfig(**raw["peft"])

        if raw.get("env"):
            cfg.env = EnvConfig(**raw["env"])

        if raw.get("hub"):
            cfg.hub = HubConfig(**raw["hub"])

        if raw.get("wandb"):
            cfg.wandb = WandBConfig(**raw["wandb"])

        if raw.get("tracking"):
            cfg.tracking = TrackingConfig(**raw["tracking"])

        if raw.get("simulation"):
            cfg.simulation = SimulationConfig(**raw["simulation"])

        return cfg

    def resolve_tracking_run_name(self) -> str:
        """Return the run name, auto-generating from job_name if not set."""
        return self.tracking.run_name or f"eval_{self.job_name}"

    def resolve_simulation_output_dir(self) -> Path:
        """Return the simulation output dir, defaulting to <output_dir>/rollouts/."""
        if self.simulation.output_dir:
            return Path(self.simulation.output_dir)
        return Path(self.output_dir) / "rollouts"


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(cfg: ExperimentConfig) -> list[str]:
    """Translate an ExperimentConfig into a lerobot-train CLI command.

    This is a pure function — easy to test independently.
    Returns a list suitable for subprocess.run().
    """
    cmd: list[str] = [
        "lerobot-train",
        f"--policy.path={cfg.policy_path}",
        f"--dataset.repo_id={cfg.dataset_repo_id}",
        f"--output_dir={cfg.output_dir}",
        f"--job_name={cfg.job_name}",
        f"--steps={cfg.steps}",
        f"--batch_size={cfg.batch_size}",
        f"--policy.optimizer_lr={cfg.policy_optimizer_lr}",
        f"--policy.scheduler_decay_lr={cfg.policy_scheduler_decay_lr}",
        "--policy.device=cuda",
    ]

    # LoRA / PEFT — omit this block entirely in the YAML for full fine-tuning
    if cfg.peft:
        cmd += [
            f"--peft.method_type={cfg.peft.method_type}",
            f"--peft.r={cfg.peft.r}",
        ]

    # LIBERO simulation env — used by lerobot-train's own per-step eval
    if cfg.env:
        cmd += [
            f"--env.type={cfg.env.type}",
            f"--env.task={cfg.env.task}",
        ]

    # HuggingFace Hub auto-push after training
    if cfg.hub.repo_id:
        cmd.append(f"--policy.repo_id={cfg.hub.repo_id}")

    # WandB (lerobot-train's native training metric logging)
    if cfg.wandb.enable:
        cmd += [
            "--wandb.enable=true",
            f"--wandb.project={cfg.wandb.project}",
        ]

    return cmd


# ── Runner ────────────────────────────────────────────────────────────────────

def run_experiment(config_path: str | Path, dry_run: bool = False) -> None:
    """Load a YAML experiment config, print a summary, and launch lerobot-train.

    Args:
        config_path: Path to a YAML file under configs/experiments/
        dry_run:     If True, print the command but do not execute it.
    """
    cfg = ExperimentConfig.from_yaml(config_path)
    cmd = build_command(cfg)

    _print_summary(cfg, cmd)

    if dry_run:
        print("\n[dry-run] Command printed above. Not executing.\n")
        return

    # Create output dir so lerobot-train doesn't fail if it doesn't exist
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("\nLaunching lerobot-train...\n" + "─" * 60 + "\n")

    result = subprocess.run(cmd)

    print("\n" + "─" * 60)
    if result.returncode == 0:
        print(f"Training complete.")
        print(f"Checkpoint : {cfg.output_dir}")
        print(f"Next steps : make eval CKPT={cfg.output_dir}")
        print(f"           : make simulate CKPT={cfg.output_dir}")
    else:
        print(
            f"ERROR: lerobot-train exited with code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def _print_summary(cfg: ExperimentConfig, cmd: list[str]) -> None:
    """Print a human-readable summary of what is about to run."""
    sim = cfg.simulation
    sim_mode = "headless (EGL, saves video)" if sim.headless else "display (live window)"

    width = 62
    print("\n" + "=" * width)
    print(f"  PolicyForge — experiment summary")
    print("=" * width)
    print(f"  Job        : {cfg.job_name}")
    print(f"  Model      : {cfg.policy_path}")
    print(f"  Dataset    : {cfg.dataset_repo_id}")
    print(f"  Steps      : {cfg.steps:,}   batch {cfg.batch_size}")
    print(f"  LoRA       : {'rank ' + str(cfg.peft.r) + ' (' + cfg.peft.method_type + ')' if cfg.peft else 'disabled — full fine-tuning'}")
    print(f"  Env        : {cfg.env.task if cfg.env else 'none'}")
    print(f"  Tracking   : {cfg.tracking.backend}")
    print(f"  Simulation : {sim_mode}, {sim.n_episodes} episodes")
    print(f"  Output     : {cfg.output_dir}")
    print("=" * width)
    print("\nCommand:")
    print("  " + " \\\n    ".join(cmd))