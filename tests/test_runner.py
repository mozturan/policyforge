"""Unit tests for policyforge/runner.py.

These tests run entirely without GPU, lerobot, or WandB.
They verify that YAML configs are parsed correctly and that
the right CLI flags are generated.

Run with: pytest tests/test_runner.py -v
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from policyforge.runner import (
    ExperimentConfig,
    PeftConfig,
    SimulationConfig,
    build_command,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_CONFIG = {
    "policy_path": "lerobot/smolvla_base",
    "dataset_repo_id": "HuggingFaceVLA/libero",
    "output_dir": "outputs/train/test_run",
}

FULL_CONFIG = {
    **MINIMAL_CONFIG,
    "job_name": "test_lora_run",
    "steps": 50000,
    "batch_size": 16,
    "policy_optimizer_lr": 5e-4,
    "policy_scheduler_decay_lr": 5e-5,
    "peft": {"method_type": "LORA", "r": 32},
    "env": {"type": "libero", "task": "libero_goal"},
    "hub": {"repo_id": "testuser/smolvla-test"},
    "wandb": {"enable": True, "project": "test-project"},
    "tracking": {"backend": "none", "project": "test-project", "run_name": "my_run"},
    "simulation": {
        "task_suite": "libero_goal",
        "n_episodes": 5,
        "max_steps": 200,
        "render_size": 128,
        "fps": 10,
        "record_video": True,
        "headless": False,
        "output_dir": "outputs/test_rollouts",
    },
}


@pytest.fixture
def minimal_yaml(tmp_path):
    """Write minimal config to a temp YAML file."""
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml.dump(MINIMAL_CONFIG))
    return p


@pytest.fixture
def full_yaml(tmp_path):
    """Write full config to a temp YAML file."""
    p = tmp_path / "full.yaml"
    p.write_text(yaml.dump(FULL_CONFIG))
    return p


# ── Parsing tests ─────────────────────────────────────────────────────────────

def test_minimal_config_loads(minimal_yaml):
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    assert cfg.policy_path == "lerobot/smolvla_base"
    assert cfg.dataset_repo_id == "HuggingFaceVLA/libero"
    assert cfg.output_dir == "outputs/train/test_run"


def test_minimal_config_has_defaults(minimal_yaml):
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    assert cfg.peft is None            # no LoRA by default
    assert cfg.env is None             # no env by default
    assert cfg.steps == 100_000        # default steps
    assert cfg.batch_size == 32        # default batch


def test_full_config_loads(full_yaml):
    cfg = ExperimentConfig.from_yaml(full_yaml)
    assert cfg.peft is not None
    assert cfg.peft.r == 32
    assert cfg.peft.method_type == "LORA"
    assert cfg.env is not None
    assert cfg.env.task == "libero_goal"
    assert cfg.hub.repo_id == "testuser/smolvla-test"


def test_simulation_headless_false(full_yaml):
    """headless=false should load correctly — this enables live window mode."""
    cfg = ExperimentConfig.from_yaml(full_yaml)
    assert cfg.simulation.headless is False
    assert cfg.simulation.n_episodes == 5


def test_simulation_default_is_headless(minimal_yaml):
    """Default simulation config should be headless."""
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    assert cfg.simulation.headless is True


def test_missing_required_field_raises(tmp_path):
    bad = {"policy_path": "lerobot/smolvla_base"}  # missing dataset_repo_id, output_dir
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(bad))
    with pytest.raises(ValueError, match="Missing required field"):
        ExperimentConfig.from_yaml(p)


def test_config_not_found_raises():
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml("configs/experiments/does_not_exist.yaml")


# ── Command builder tests ─────────────────────────────────────────────────────

def test_command_starts_with_lerobot_train(minimal_yaml):
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    cmd = build_command(cfg)
    assert cmd[0] == "lerobot-train"


def test_command_contains_required_flags(minimal_yaml):
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    cmd = build_command(cfg)
    joined = " ".join(cmd)
    assert "--policy.path=lerobot/smolvla_base" in joined
    assert "--dataset.repo_id=HuggingFaceVLA/libero" in joined
    assert "--policy.device=cuda" in joined


def test_lora_flags_present_when_peft_configured(full_yaml):
    cfg = ExperimentConfig.from_yaml(full_yaml)
    cmd = build_command(cfg)
    joined = " ".join(cmd)
    assert "--peft.method_type=LORA" in joined
    assert "--peft.r=32" in joined


def test_lora_flags_absent_when_no_peft(minimal_yaml):
    """No peft block in YAML = no LoRA flags = full fine-tuning."""
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    assert cfg.peft is None
    cmd = build_command(cfg)
    joined = " ".join(cmd)
    assert "--peft" not in joined


def test_env_flags_present_when_configured(full_yaml):
    cfg = ExperimentConfig.from_yaml(full_yaml)
    cmd = build_command(cfg)
    joined = " ".join(cmd)
    assert "--env.type=libero" in joined
    assert "--env.task=libero_goal" in joined


def test_hub_flag_present_when_repo_id_set(full_yaml):
    cfg = ExperimentConfig.from_yaml(full_yaml)
    cmd = build_command(cfg)
    assert "--policy.repo_id=testuser/smolvla-test" in " ".join(cmd)


def test_hub_flag_absent_when_repo_id_empty(minimal_yaml):
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    assert cfg.hub.repo_id == ""
    cmd = build_command(cfg)
    assert "--policy.repo_id" not in " ".join(cmd)


def test_wandb_flags_present_when_enabled(full_yaml):
    cfg = ExperimentConfig.from_yaml(full_yaml)
    cmd = build_command(cfg)
    joined = " ".join(cmd)
    assert "--wandb.enable=true" in joined
    assert "--wandb.project=test-project" in joined


# ── Helper method tests ───────────────────────────────────────────────────────

def test_resolve_tracking_run_name_uses_job_name_when_empty(minimal_yaml):
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    assert cfg.resolve_tracking_run_name() == f"eval_{cfg.job_name}"


def test_resolve_tracking_run_name_uses_explicit_name(full_yaml):
    cfg = ExperimentConfig.from_yaml(full_yaml)
    assert cfg.resolve_tracking_run_name() == "my_run"


def test_resolve_simulation_output_dir_defaults_to_checkpoint(minimal_yaml):
    cfg = ExperimentConfig.from_yaml(minimal_yaml)
    expected = Path(cfg.output_dir) / "rollouts"
    assert cfg.resolve_simulation_output_dir() == expected


def test_resolve_simulation_output_dir_respects_explicit_dir(full_yaml):
    cfg = ExperimentConfig.from_yaml(full_yaml)
    assert cfg.resolve_simulation_output_dir() == Path("outputs/test_rollouts")