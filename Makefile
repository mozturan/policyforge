.PHONY: setup train train-dry eval simulate serve push demo test clean

# ── Setup ────────────────────────────────────────────────────────────────────
# Run once after cloning. Installs torch (CUDA 12.1), lerobot, then this package.
setup:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip install lerobot
	pip install -e ".[dev]"

# ── Training (wraps lerobot-train) ───────────────────────────────────────────
# Usage: make train EXP=smolvla_libero_lora
train:
	@test -n "$(EXP)" || (echo "ERROR: provide EXP=<experiment_name>" && exit 1)
	python scripts/train.py configs/experiments/$(EXP).yaml

# Dry run: prints the lerobot-train command without executing it
train-dry:
	@test -n "$(EXP)" || (echo "ERROR: provide EXP=<experiment_name>" && exit 1)
	python scripts/train.py configs/experiments/$(EXP).yaml --dry-run

# ── Post-training evaluation ─────────────────────────────────────────────────
# Usage: make eval CKPT=outputs/train/smolvla_libero_lora
eval:
	@test -n "$(CKPT)" || (echo "ERROR: provide CKPT=<checkpoint_path>" && exit 1)
	python scripts/evaluate.py --checkpoint $(CKPT)

# ── Simulation visualization ─────────────────────────────────────────────────
# Usage: make simulate CKPT=outputs/train/smolvla_libero_lora
# Usage: make simulate CKPT=outputs/train/smolvla_libero_lora TASK=libero_goal N=5
simulate:
	@test -n "$(CKPT)" || (echo "ERROR: provide CKPT=<checkpoint_path>" && exit 1)
	python scripts/simulate.py \
		--checkpoint $(CKPT) \
		$(if $(TASK),--task $(TASK)) \
		$(if $(N),--n-episodes $(N))

# ── Inference server ─────────────────────────────────────────────────────────
# Usage: make serve CKPT=outputs/train/smolvla_libero_lora
serve:
	@test -n "$(CKPT)" || (echo "ERROR: provide CKPT=<checkpoint_path>" && exit 1)
	CHECKPOINT_PATH=$(CKPT) uvicorn policyforge.serve.app:app \
		--host 0.0.0.0 --port 8000 --reload

# ── HuggingFace Hub push ─────────────────────────────────────────────────────
# Usage: make push CKPT=outputs/train/smolvla_libero_lora REPO=username/my-model
push:
	@test -n "$(CKPT)" || (echo "ERROR: provide CKPT=<checkpoint_path>" && exit 1)
	@test -n "$(REPO)" || (echo "ERROR: provide REPO=<hf-username/model-name>" && exit 1)
	python scripts/push_to_hub.py --checkpoint $(CKPT) --repo-id $(REPO)

# ── Local Gradio demo ────────────────────────────────────────────────────────
demo:
	python demo/app.py

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/ mlruns/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true