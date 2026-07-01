from pydantic import BaseModel, Field
from typing import Optional

class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = ["q_proj", "v_proj", "out_proj"]
    lora_dropout: float = 0.05
    bias: str = "none"

class TrainingConfig(BaseModel):
    # Model
    model_name: str = "lerobot/smolvla_base"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Data
    dataset_name: str = "lerobot/libero_spatial"
    train_split: float = 0.9

    # Training
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2

    # Infra
    output_dir: str = "checkpoints"
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 5
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "smolvla-libero"

    # Hardware
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4

class EvalConfig(BaseModel):
    checkpoint_path: str
    tasks: list[str] = ["libero_spatial", "libero_goal", "libero_object"]
    num_rollouts_per_task: int = 20
    max_steps_per_episode: int = 600
    render_video: bool = True
    output_dir: str = "eval_results"

