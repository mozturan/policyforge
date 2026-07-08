"""HuggingFace Hub registry utilities.

Pushes a trained checkpoint to the Hub with an auto-generated model card
that includes training details and (optionally) evaluation results.

Usage:
    from policyforge.registry.hub import push_to_hub, ModelCardConfig
    from policyforge.runner import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/experiments/smolvla_libero_lora.yaml")

    push_to_hub(
        checkpoint_path = "outputs/train/smolvla_libero_lora",
        repo_id         = "your-username/smolvla-libero-lora",
        card_cfg        = ModelCardConfig.from_experiment_config("your-username/smolvla-libero-lora", cfg),
        eval_report     = "outputs/train/smolvla_libero_lora/eval/eval_report.json",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Model card config ─────────────────────────────────────────────────────────

@dataclass
class ModelCardConfig:
    """All the information needed to generate a HuggingFace model card."""

    repo_id: str                                    # "username/model-name"
    base_model: str       = "lerobot/smolvla_base"
    dataset_repo_id: str  = "HuggingFaceVLA/libero"
    task_suite: str       = "libero_spatial"
    lora_rank: int | None = None                    # None → full fine-tune
    steps: int | None     = None
    batch_size: int | None = None
    optimizer_lr: float | None = None
    license: str          = "mit"
    tags: list[str]       = field(default_factory=lambda: [
        "robotics", "vision-language-action", "lerobot", "smolvla", "policyforge",
    ])
    policyforge_url: str  = "https://github.com/your-username/policyforge"

    @classmethod
    def from_experiment_config(cls, repo_id: str, cfg: Any) -> "ModelCardConfig":
        """Build a ModelCardConfig from a loaded ExperimentConfig."""
        return cls(
            repo_id        = repo_id,
            base_model     = cfg.policy_path,
            dataset_repo_id = cfg.dataset_repo_id,
            task_suite     = cfg.env.task if cfg.env else "unknown",
            lora_rank      = cfg.peft.r if cfg.peft else None,
            steps          = cfg.steps,
            batch_size     = cfg.batch_size,
            optimizer_lr   = cfg.policy_optimizer_lr,
        )


# ── Model card generation ─────────────────────────────────────────────────────

def generate_model_card(
    card_cfg: ModelCardConfig,
    eval_report: dict | None = None,
) -> str:
    """Generate a HuggingFace model card as a Markdown string.

    Produces:
    - YAML frontmatter (tags, license, base_model, datasets)
    - Model description
    - Training details table
    - Evaluation results table (if eval_report is provided)
    - Usage example
    - Citation block

    Args:
        card_cfg:    Configuration for the model card content.
        eval_report: Parsed eval_report.json dict from run_eval(). Optional.

    Returns:
        Full model card as a Markdown string.
    """
    model_name = card_cfg.repo_id.split("/")[-1]
    fine_tune_method = (
        f"LoRA (rank {card_cfg.lora_rank})"
        if card_cfg.lora_rank
        else "Full fine-tuning"
    )

    tags = list(card_cfg.tags)
    if card_cfg.lora_rank:
        tags.append("lora")

    # ── YAML frontmatter ──────────────────────────────────────────────────────
    frontmatter_lines = [
        "---",
        "language: en",
        f"license: {card_cfg.license}",
        "library_name: lerobot",
        f"base_model: {card_cfg.base_model}",
        "tags:",
        *[f"  - {t}" for t in tags],
        "datasets:",
        f"  - {card_cfg.dataset_repo_id}",
    ]

    # Add model-index block if we have eval results
    if eval_report:
        overall = eval_report.get("overall_success_rate", 0)
        frontmatter_lines += [
            "model-index:",
            f"  - name: {model_name}",
            "    results:",
            "      - task:",
            "          type: robotics-manipulation",
            "        dataset:",
            f"          name: {card_cfg.task_suite}",
            f"          type: {card_cfg.dataset_repo_id}",
            "        metrics:",
            "          - type: success_rate",
            f"            value: {overall:.4f}",
            "            name: Task success rate",
        ]

    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    # ── Training details table ────────────────────────────────────────────────
    training_rows: list[tuple[str, str]] = [
        ("Base model",       card_cfg.base_model),
        ("Dataset",          card_cfg.dataset_repo_id),
        ("Task suite",       card_cfg.task_suite),
        ("Fine-tuning",      fine_tune_method),
    ]
    if card_cfg.steps:
        training_rows.append(("Steps", f"{card_cfg.steps:,}"))
    if card_cfg.batch_size:
        training_rows.append(("Batch size", str(card_cfg.batch_size)))
    if card_cfg.optimizer_lr:
        training_rows.append(("Learning rate", str(card_cfg.optimizer_lr)))

    training_table = _make_table(["Parameter", "Value"], training_rows)

    # ── Evaluation results table (optional) ───────────────────────────────────
    eval_section = ""
    if eval_report:
        overall_rate = eval_report.get("overall_success_rate", 0)
        total_eps    = eval_report.get("total_episodes", 0)
        eval_date    = eval_report.get("timestamp", "")

        task_rows = [
            (_truncate(lang, 60), f"{m['success_rate']:.0%}", str(m["n_episodes"]))
            for lang, m in eval_report.get("tasks", {}).items()
        ]
        task_rows.append(("**Overall**", f"**{overall_rate:.0%}**", str(total_eps)))

        eval_section = f"""
## Evaluation results

Evaluated on **{card_cfg.task_suite}** with {total_eps} total episodes.  
Date: {eval_date}

{_make_table(["Task", "Success rate", "Episodes"], task_rows)}
"""

    # ── Usage example ─────────────────────────────────────────────────────────
    usage_section = f"""
## Usage

```python
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("{card_cfg.repo_id}")
policy = policy.cuda().eval()

# Or via the PolicyForge inference API:
# CHECKPOINT_PATH={card_cfg.repo_id} uvicorn policyforge.serve.app:app
```
"""

    # ── Full card ─────────────────────────────────────────────────────────────
    body = f"""# {model_name}

Fine-tuned SmolVLA robot policy, trained with [PolicyForge]({card_cfg.policyforge_url}) —
a production MLOps platform for Vision-Language-Action model experimentation.

## Model description

- **Base model:** [{card_cfg.base_model}](https://huggingface.co/{card_cfg.base_model})
- **Fine-tuning:** {fine_tune_method}
- **Training data:** [{card_cfg.dataset_repo_id}](https://huggingface.co/datasets/{card_cfg.dataset_repo_id})

## Training details

{training_table}
{eval_section}{usage_section}
## Citation

```bibtex
@software{{policyforge,
  author  = {{Your Name}},
  title   = {{PolicyForge: Production MLOps for Robot Policy Learning}},
  url     = {{{card_cfg.policyforge_url}}},
}}
```
"""

    return frontmatter + "\n" + body


# ── Hub push ──────────────────────────────────────────────────────────────────

def push_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    card_cfg: ModelCardConfig | None = None,
    eval_report: dict | str | Path | None = None,
    commit_message: str = "Add PolicyForge checkpoint",
    private: bool = False,
) -> str:
    """Push a checkpoint to HuggingFace Hub with an auto-generated model card.

    Steps:
      1. Create the Hub repo if it doesn't exist.
      2. Generate and upload the model card (README.md).
      3. Upload the entire checkpoint directory.

    Args:
        checkpoint_path: lerobot-train output directory to push.
        repo_id:         HuggingFace repo, e.g. "username/smolvla-libero-lora".
        card_cfg:        Model card configuration. If None, uses defaults.
        eval_report:     Path to eval_report.json or already-parsed dict.
                         If None, checks <checkpoint>/eval/eval_report.json automatically.
        commit_message:  Git commit message for the upload.
        private:         Whether to make the repo private.

    Returns:
        URL of the uploaded model on HuggingFace Hub.
    """
    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface-hub"
        ) from e

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Resolve eval report ───────────────────────────────────────────────────
    report_dict: dict | None = None

    if eval_report is None:
        # Check the standard location automatically
        auto_path = checkpoint_path / "eval" / "eval_report.json"
        if auto_path.exists():
            report_dict = json.loads(auto_path.read_text())
            print(f"[registry] Found eval report: {auto_path}")
    elif isinstance(eval_report, (str, Path)):
        report_dict = json.loads(Path(eval_report).read_text())
    else:
        report_dict = eval_report

    # ── Build card config if not provided ─────────────────────────────────────
    if card_cfg is None:
        card_cfg = ModelCardConfig(repo_id=repo_id)

    # ── Generate model card ───────────────────────────────────────────────────
    card_text = generate_model_card(card_cfg, eval_report=report_dict)

    # ── Create repo ───────────────────────────────────────────────────────────
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    print(f"[registry] Repo ready: https://huggingface.co/{repo_id}")

    # ── Upload model card first (so it's visible immediately) ─────────────────
    api.upload_file(
        path_or_fileobj=card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=f"{commit_message} (model card)",
    )
    print("[registry] Model card uploaded.")

    # ── Upload checkpoint (skip README.md — we just uploaded a better one) ────
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        commit_message=commit_message,
        ignore_patterns=["README.md", "*.log", "wandb/", "__pycache__/"],
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[registry] Done. Model available at: {url}")
    return url

def pull_from_hub(repo_id: str, local_dir: str | Path) -> Path:
    """Download a checkpoint from HuggingFace Hub to a local directory.

    Args:
        repo_id:   HuggingFace model repo, e.g. "username/smolvla-libero-lora".
        local_dir: Where to save the downloaded files.

    Returns:
        Path to the downloaded checkpoint directory.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("huggingface_hub is not installed.") from e

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[registry] Downloading {repo_id} → {local_dir}")
    path = snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"[registry] Downloaded to: {path}")
    return Path(path)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_table(headers: list[str], rows: list[tuple]) -> str:
    """Build a simple Markdown table."""
    header_row    = "| " + " | ".join(headers) + " |"
    separator_row = "|" + "|".join(["---"] * len(headers)) + "|"
    data_rows     = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *data_rows])

def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"