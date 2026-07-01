import mlflow
import torch
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from policyforge.train.config import TrainingConfig

class PolicyTrainer:
    def __init__(self, model, train_loader, val_loader, cfg: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.scaler = GradScaler() if cfg.mixed_precision in ("fp16", "bf16") else None
        
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_epochs
        )
    
    def train(self):
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)
        
        with mlflow.start_run():
            # Log all hyperparams — this is the point of config-driven training
            mlflow.log_params(self.cfg.model_dump())
            mlflow.log_params(count_parameters(self.model))
            mlflow.set_tag("model", self.cfg.model_name)
            mlflow.set_tag("dataset", self.cfg.dataset_name)
            
            best_val_loss = float("inf")
            global_step = 0
            
            for epoch in range(self.cfg.num_epochs):
                train_loss = self._train_epoch(epoch, global_step)
                val_loss = self._val_epoch()
                
                self.scheduler.step()
                
                mlflow.log_metrics({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                }, step=epoch)
                
                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best", epoch, val_loss)
                    mlflow.log_artifact(str(Path(self.cfg.output_dir) / "best"))
                
                # Periodic checkpoint
                if epoch % self.cfg.eval_every_n_epochs == 0:
                    self._save_checkpoint(f"epoch_{epoch:04d}", epoch, val_loss)
                
                print(f"[Epoch {epoch:03d}] train={train_loss:.4f} val={val_loss:.4f}")
    
    def _train_epoch(self, epoch, global_step):
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(self.train_loader):
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # choose autocast dtype only if using mixed precision
            # choose autocast context only if using mixed precision
            from contextlib import nullcontext
            use_autocast = self.scaler is not None

            dtype = None
            if self.cfg.mixed_precision == "bf16":
                dtype = torch.bfloat16
            elif self.cfg.mixed_precision == "fp16":
                dtype = torch.float16

            if use_autocast:
                # torch.cuda.amp.autocast expects device_type and (optionally) dtype
                if dtype is not None:
                    ctx = autocast(device_type="cuda", dtype=dtype)
                else:
                    ctx = autocast()
            else:
                ctx = nullcontext()

            with ctx:
                loss = self.model(**batch).loss
                loss = loss / self.cfg.gradient_accumulation_steps

            # backward + optimizer step with or without GradScaler
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                global_step += 1

                if global_step % self.cfg.log_every_n_steps == 0:
                    mlflow.log_metric("train/step_loss", loss.item(), step=global_step)

            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _val_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # use autocast only when mixed precision bf16 is enabled
                if self.scaler is not None and self.cfg.mixed_precision == "bf16":
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = self.model(**batch).loss
                else:
                    with nullcontext():
                        loss = self.model(**batch).loss
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    
    def _save_checkpoint(self, name, epoch, val_loss):
        path = Path(self.cfg.output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        # Save metadata alongside weights
        import json
        meta = {"epoch": epoch, "val_loss": val_loss, "config": self.cfg.model_dump()}
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))

