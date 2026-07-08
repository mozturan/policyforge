"""scripts/train.py — launch a training experiment.

Usage:
    python scripts/train.py configs/experiments/smolvla_libero_lora.yaml
    python scripts/train.py configs/experiments/smolvla_libero_lora.yaml --dry-run

Via Makefile:
    make train EXP=smolvla_libero_lora
    make train-dry EXP=smolvla_libero_lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load .env file before anything else so all env vars are available
from dotenv import load_dotenv
load_dotenv()

# Add project root to path (handles running from any directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from policyforge.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PolicyForge training — wraps lerobot-train with a YAML config layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/train.py configs/experiments/smolvla_libero_lora.yaml
  python scripts/train.py configs/experiments/smolvla_libero_lora.yaml --dry-run
  make train EXP=smolvla_libero_lora
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML (e.g. configs/experiments/smolvla_libero_lora.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lerobot-train command without executing it. "
             "Use this to verify the config before a long training run.",
    )
    args = parser.parse_args()

    run_experiment(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()