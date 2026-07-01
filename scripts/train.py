import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    candidate_paths = [path]

    if not path.is_absolute():
        candidate_paths.append(REPO_ROOT / path)

        if path.parts[:2] == ("configs", "train"):
            candidate_paths.append(REPO_ROOT / "policyforge" / "train" / Path(*path.parts[2:]))

        candidate_paths.append(REPO_ROOT / "policyforge" / "train" / path.name)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find config file '{config_path}'. Tried: {', '.join(str(p) for p in candidate_paths)}"
    )


def main(config_path: str):
    import yaml

    from policyforge.train import TrainingConfig
    from policyforge.train import make_dataloaders
    from policyforge.train import apply_lora
    from policyforge.train import PolicyTrainer
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    resolved_config_path = resolve_config_path(config_path)
    cfg = TrainingConfig(**yaml.safe_load(resolved_config_path.read_text()))

    print(f"Loading model: {cfg.model_name}")
    model = SmolVLAPolicy.from_pretrained(cfg.model_name)
    model = apply_lora(model, cfg.lora)
    model = model.cuda()

    print(f"Loading dataset: {cfg.dataset_name}")
    train_loader, val_loader = make_dataloaders(cfg)

    trainer = PolicyTrainer(model, train_loader, val_loader, cfg)
    trainer.train()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/train.py <config_path>")
    main(sys.argv[1])

