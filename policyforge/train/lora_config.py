from typing import Literal, cast

from peft import LoraConfig, get_peft_model, TaskType
from .config import LoRAConfig

def apply_lora(model, lora_cfg: LoRAConfig):
    """Apply LoRA adapters to SmolVLA's language/vision backbone."""
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=cast(Literal["none", "all", "lora_only"], lora_cfg.bias),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Log this — it's satisfying and informative
    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "ratio": trainable / total}

