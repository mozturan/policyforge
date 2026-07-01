import sys
import yaml
from pathlib import Path
from policyforge.train.config import TrainingConfig
from policyforge.train.dataset import make_dataloaders
from policyforge.train.lora_config import apply_lora
from policyforge.train.trainer import PolicyTrainer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main(config_path: str):
    cfg = TrainingConfig(**yaml.safe_load(Path(config_path).read_text()))
    
    print(f"Loading model: {cfg.model_name}")
    model = SmolVLAPolicy.from_pretrained(cfg.model_name)
    model = apply_lora(model, cfg.lora)
    model = model.cuda()
    
    print(f"Loading dataset: {cfg.dataset_name}")
    train_loader, val_loader = make_dataloaders(cfg)
    
    trainer = PolicyTrainer(model, train_loader, val_loader, cfg)
    trainer.train()

if __name__ == "__main__":
    main(sys.argv[1])

