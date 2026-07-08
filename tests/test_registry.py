"""Unit tests for policyforge/registry/.

Tests model card generation entirely without HuggingFace credentials,
network access, or any ML dependencies.

Run with: pytest tests/test_registry.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from policyforge.registry.hub import ModelCardConfig, generate_model_card


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_card_cfg():
    return ModelCardConfig(repo_id="testuser/smolvla-test")


@pytest.fixture
def full_card_cfg():
    return ModelCardConfig(
        repo_id         = "testuser/smolvla-libero-lora",
        base_model      = "lerobot/smolvla_base",
        dataset_repo_id = "HuggingFaceVLA/libero",
        task_suite      = "libero_spatial",
        lora_rank       = 64,
        steps           = 100_000,
        batch_size      = 32,
        optimizer_lr    = 1e-3,
    )


@pytest.fixture
def eval_report():
    return {
        "checkpoint": "outputs/train/smolvla_libero_lora",
        "suite": "libero_spatial",
        "timestamp": "2024-01-15T14:32:00",
        "overall_success_rate": 0.725,
        "total_episodes": 40,
        "total_successes": 29,
        "tasks": {
            "pick up the red block and place it in the bowl": {
                "success_rate": 0.80,
                "n_episodes": 20,
                "n_success": 16,
                "avg_steps": 127.3,
                "avg_duration_seconds": 4.2,
            },
            "stack the wooden blocks in order": {
                "success_rate": 0.65,
                "n_episodes": 20,
                "n_success": 13,
                "avg_steps": 203.1,
                "avg_duration_seconds": 6.8,
            },
        },
    }


def parse_frontmatter(card_text: str) -> dict:
    """Extract and parse the YAML frontmatter from a model card string."""
    lines = card_text.split("\n")
    assert lines[0] == "---", "Card must start with ---"
    end = lines.index("---", 1)
    return yaml.safe_load("\n".join(lines[1:end]))


# ── ModelCardConfig ───────────────────────────────────────────────────────────

class TestModelCardConfig:
    def test_defaults(self, minimal_card_cfg):
        cfg = minimal_card_cfg
        assert cfg.repo_id == "testuser/smolvla-test"
        assert cfg.base_model == "lerobot/smolvla_base"
        assert cfg.lora_rank is None
        assert cfg.license == "mit"

    def test_tags_default_include_robotics(self, minimal_card_cfg):
        assert "robotics" in minimal_card_cfg.tags

    def test_from_experiment_config(self):
        from policyforge.runner import ExperimentConfig, PeftConfig, EnvConfig

        cfg = ExperimentConfig(
            policy_path     = "lerobot/smolvla_base",
            dataset_repo_id = "HuggingFaceVLA/libero",
            output_dir      = "outputs/train/test",
        )
        cfg.peft = PeftConfig(method_type="LORA", r=32)
        cfg.env  = EnvConfig(type="libero", task="libero_goal")

        card_cfg = ModelCardConfig.from_experiment_config("user/model", cfg)

        assert card_cfg.repo_id         == "user/model"
        assert card_cfg.lora_rank       == 32
        assert card_cfg.task_suite      == "libero_goal"
        assert card_cfg.base_model      == "lerobot/smolvla_base"
        assert card_cfg.dataset_repo_id == "HuggingFaceVLA/libero"

    def test_from_experiment_config_no_peft(self):
        from policyforge.runner import ExperimentConfig
        cfg = ExperimentConfig(
            policy_path="lerobot/smolvla_base",
            dataset_repo_id="HuggingFaceVLA/libero",
            output_dir="outputs",
        )
        card_cfg = ModelCardConfig.from_experiment_config("user/model", cfg)
        assert card_cfg.lora_rank is None   # full fine-tune


# ── Frontmatter ───────────────────────────────────────────────────────────────

class TestFrontmatter:
    def test_starts_with_triple_dash(self, minimal_card_cfg):
        card = generate_model_card(minimal_card_cfg)
        assert card.startswith("---")

    def test_frontmatter_is_valid_yaml(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        fm = parse_frontmatter(card)
        assert isinstance(fm, dict)

    def test_license_in_frontmatter(self, full_card_cfg):
        fm = parse_frontmatter(generate_model_card(full_card_cfg))
        assert fm["license"] == "mit"

    def test_base_model_in_frontmatter(self, full_card_cfg):
        fm = parse_frontmatter(generate_model_card(full_card_cfg))
        assert fm["base_model"] == "lerobot/smolvla_base"

    def test_tags_in_frontmatter(self, full_card_cfg):
        fm = parse_frontmatter(generate_model_card(full_card_cfg))
        assert "robotics" in fm["tags"]
        assert "smolvla" in fm["tags"]

    def test_lora_tag_added_when_lora_rank_set(self, full_card_cfg):
        fm = parse_frontmatter(generate_model_card(full_card_cfg))
        assert "lora" in fm["tags"]

    def test_lora_tag_absent_when_no_lora(self, minimal_card_cfg):
        fm = parse_frontmatter(generate_model_card(minimal_card_cfg))
        assert "lora" not in fm.get("tags", [])

    def test_dataset_in_frontmatter(self, full_card_cfg):
        fm = parse_frontmatter(generate_model_card(full_card_cfg))
        assert full_card_cfg.dataset_repo_id in fm["datasets"]

    def test_model_index_present_with_eval_report(self, full_card_cfg, eval_report):
        fm = parse_frontmatter(generate_model_card(full_card_cfg, eval_report=eval_report))
        assert "model-index" in fm

    def test_model_index_absent_without_eval_report(self, full_card_cfg):
        fm = parse_frontmatter(generate_model_card(full_card_cfg))
        assert "model-index" not in (fm or {})

    def test_eval_success_rate_in_model_index(self, full_card_cfg, eval_report):
        fm = parse_frontmatter(generate_model_card(full_card_cfg, eval_report=eval_report))
        metrics = fm["model-index"][0]["results"][0]["metrics"]
        values = [m["value"] for m in metrics]
        assert pytest.approx(0.725, abs=0.001) in values


# ── Card body ─────────────────────────────────────────────────────────────────

class TestCardBody:
    def test_title_uses_model_name(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert "# smolvla-libero-lora" in card

    def test_base_model_mentioned(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert "lerobot/smolvla_base" in card

    def test_lora_rank_mentioned(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert "LoRA" in card
        assert "64" in card

    def test_full_fine_tune_mentioned_when_no_lora(self, minimal_card_cfg):
        card = generate_model_card(minimal_card_cfg)
        assert "Full fine-tuning" in card

    def test_steps_in_training_table(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert "100,000" in card

    def test_usage_section_present(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert "## Usage" in card
        assert "from_pretrained" in card

    def test_citation_section_present(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert "## Citation" in card
        assert "bibtex" in card

    def test_eval_section_present_with_report(self, full_card_cfg, eval_report):
        card = generate_model_card(full_card_cfg, eval_report=eval_report)
        assert "## Evaluation results" in card

    def test_eval_section_absent_without_report(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert "## Evaluation results" not in card

    def test_eval_overall_rate_in_card(self, full_card_cfg, eval_report):
        card = generate_model_card(full_card_cfg, eval_report=eval_report)
        # f"{0.725:.0%}" uses banker's rounding → "72%" (72.5 rounds to even 72)
        assert "72%" in card

    def test_task_names_in_eval_table(self, full_card_cfg, eval_report):
        card = generate_model_card(full_card_cfg, eval_report=eval_report)
        assert "pick up the red block" in card

    def test_long_task_names_truncated(self, full_card_cfg):
        long_task = "a" * 100
        report = {
            "overall_success_rate": 0.5,
            "total_episodes": 10,
            "timestamp": "2024",
            "tasks": {long_task: {"success_rate": 0.5, "n_episodes": 10}},
        }
        card = generate_model_card(full_card_cfg, eval_report=report)
        assert "…" in card

    def test_repo_id_in_usage_example(self, full_card_cfg):
        card = generate_model_card(full_card_cfg)
        assert full_card_cfg.repo_id in card