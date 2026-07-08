"""scripts/evaluate.py — run post-training evaluation on a checkpoint.

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/train/smolvla_libero_lora \
        --config configs/experiments/smolvla_libero_lora.yaml

Via Makefile:
    make eval CKPT=outputs/train/smolvla_libero_lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training LIBERO evaluation for PolicyForge.",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to lerobot-train output directory.")
    parser.add_argument("--config", default=None,
                        help="Experiment YAML for simulation and tracking settings.")
    parser.add_argument("--task", default=None,
                        help="Override task suite (e.g. libero_spatial, libero_goal).")
    parser.add_argument("--n-episodes", type=int, default=None, metavar="N",
                        help="Episodes per task.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode before failure.")
    parser.add_argument("--output-dir", default=None,
                        help="Report output directory. Default: <checkpoint>/eval/")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording.")
    parser.add_argument("--no-tracker", action="store_true",
                        help="Disable experiment tracking.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", default=None)
    mode.add_argument("--display", dest="headless", action="store_false",
                      help="Show live window (requires display + opencv).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from policyforge.runner import ExperimentConfig, SimulationConfig, TrackingConfig

    if args.config:
        cfg     = ExperimentConfig.from_yaml(args.config)
        sim_cfg = cfg.simulation
        trk_cfg = cfg.tracking
    else:
        sim_cfg = SimulationConfig()
        trk_cfg = TrackingConfig()

    if args.task:       sim_cfg.task_suite   = args.task
    if args.n_episodes: sim_cfg.n_episodes   = args.n_episodes
    if args.max_steps:  sim_cfg.max_steps    = args.max_steps
    if args.no_video:   sim_cfg.record_video = False
    if args.headless is not None:
        sim_cfg.headless = args.headless

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.checkpoint) / "eval"
    )

    from policyforge.tracking import get_tracker

    if args.no_tracker:
        tracker = get_tracker("none")
    else:
        run_name = f"eval_{Path(args.checkpoint).name}"
        tracker  = get_tracker(
            trk_cfg.backend,
            project  = trk_cfg.project,
            run_name = trk_cfg.run_name or run_name,
        )

    from policyforge.eval.harness import run_eval
    from policyforge.eval.reporter import generate_markdown_report

    with tracker:
        metrics = run_eval(
            checkpoint_path = args.checkpoint,
            sim_cfg         = sim_cfg,
            tracker         = tracker,
            output_dir      = output_dir,
            verbose         = True,
        )

    print("\n" + generate_markdown_report(metrics) + "\n")


if __name__ == "__main__":
    main()