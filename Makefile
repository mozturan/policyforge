.PHONY: train eval serve test

train:
	python scripts/train.py config=configs/train/smolvla_libero_lora.yaml

eval:
	python scripts/evaluate.py config=configs/eval/libero_suite.yaml

serve:
	uvicorn policyforge.serve.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

push:
	python scripts/push_to_hub.py --checkpoint checkpoints/best
