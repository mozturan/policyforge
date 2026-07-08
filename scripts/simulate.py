"""scripts/simulate.py — run LIBERO simulation rollouts with a trained policy.

Supports both headless mode (saves video to disk) and display mode
(opens a live window — requires a display and opencv-python installed).

Usage:
    # Headless — saves videos, no display needed
    python scripts/simulate.py --checkpoint outputs/train/my_run

    # Live window — watch the robot act in real time
    python scripts/simulate.py --checkpoint outputs/train/my_run --display

    # Override task and episode count
    python scripts/simulate.py --checkpoint outputs/train/my_run \\
        --task libero_goal --n-episodes 5

    # Side-by-side comparison: base model vs your fine-tuned model
    python scripts/simulate.py \\
        --checkpoint outputs/train/my_run \\
        --compare-with lerobot/smolvla_base

Via Makefile:
    make simulate CKPT=outputs/train/my_run
    make simulate CKPT=outputs/train/my_run TASK=libero_goal N=5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Argument parsing happens before any LIBERO/MuJoCo imports ────────────────
# setup_rendering() must be called before those imports, so we parse args first.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LIBERO simulation rollouts with a trained PolicyForge model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to lerobot-train output directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML to read simulation settings from. "
             "CLI flags below override YAML values.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="LIBERO task suite (e.g. libero_spatial, libero_goal, libero_object).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max environment steps per episode before declaring failure.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=None,
        help="Camera resolution in pixels (square, e.g. 256).",
    )

    # Headless vs display — mutually exclusive
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="Offscreen rendering — saves video to disk. Works without a display.",
    )
    mode_group.add_argument(
        "--display",
        dest="headless",
        action="store_false",
        help="Open a live window and show the rollout in real time. "
             'Requires: display + pip install -e ".[display]"',
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save videos. Defaults to <checkpoint>/rollouts/",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video saving (useful for quick success-rate checks).",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also save a GIF alongside each MP4 (for README/portfolio use).",
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        metavar="BASE_CHECKPOINT",
        help="Path to a second (base) checkpoint. Runs both and saves "
             "a side-by-side comparison video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load simulation config from YAML (CLI args override) ─────────────────
    from policyforge.runner import ExperimentConfig, SimulationConfig

    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
        sim = cfg.simulation
    else:
        sim = SimulationConfig()

    # Apply CLI overrides
    if args.task:        sim.task_suite   = args.task
    if args.n_episodes:  sim.n_episodes   = args.n_episodes
    if args.max_steps:   sim.max_steps    = args.max_steps
    if args.render_size: sim.render_size  = args.render_size
    if args.headless is not None:
        sim.headless = args.headless
    if args.no_video:
        sim.record_video = False

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.checkpoint) / "rollouts"
    )

    # ── Setup rendering BEFORE importing LIBERO ───────────────────────────────
    from policyforge.simulation.env import setup_rendering
    setup_rendering(headless=sim.headless)

    # ── Now safe to import LIBERO and the rest of the simulation module ───────
    from policyforge.simulation.env import get_libero_tasks, load_policy, make_libero_env
    from policyforge.simulation.recorder import save_comparison_mp4, save_gif, save_mp4
    from policyforge.simulation.rollout import run_episode

    # ── Print run summary ─────────────────────────────────────────────────────
    mode_str = "headless (EGL, saves video)" if sim.headless else "display (live window)"
    print("\n" + "=" * 58)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Task suite : {sim.task_suite}")
    print(f"  Episodes   : {sim.n_episodes}  max_steps {sim.max_steps}")
    print(f"  Mode       : {mode_str}")
    print(f"  Output     : {output_dir}")
    if args.compare_with:
        print(f"  Comparing  : {args.compare_with}")
    print("=" * 58 + "\n")

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = load_policy(args.checkpoint)
    base_policy = load_policy(args.compare_with) if args.compare_with else None

    # ── Get tasks ─────────────────────────────────────────────────────────────
    tasks = get_libero_tasks(sim.task_suite)
    # Run at most n_episodes across tasks (cycle through tasks if needed)
    task_sequence = [tasks[i % len(tasks)] for i in range(sim.n_episodes)]

    # ── Run episodes ──────────────────────────────────────────────────────────
    from policyforge.simulation.rollout import EpisodeResult
    results: list[EpisodeResult] = []
    base_results: list[EpisodeResult] = []

    for i, task in enumerate(task_sequence):
        env = make_libero_env(task.bddl_file, sim.render_size)

        print(f"Episode {i+1:02d}/{sim.n_episodes}  task: {task.language[:60]}")

        result = run_episode(
            policy=policy,
            env=env,
            task_name=sim.task_suite,
            task_language=task.language,
            episode_idx=i,
            max_steps=sim.max_steps,
            record=sim.record_video,
            headless=sim.headless,
            display_fps=sim.fps,
        )
        results.append(result)

        status = "SUCCESS" if result.success else "FAIL"
        print(f"  → {status}  ({result.steps_taken} steps, {result.duration_seconds:.1f}s)")

        # Save fine-tuned policy video
        if sim.record_video and result.frames:
            ep_path = output_dir / f"ep{i+1:03d}_{status.lower()}.mp4"
            save_mp4(result.frames, ep_path, fps=sim.fps)
            if args.gif:
                save_gif(result.frames, ep_path.with_suffix(".gif"), fps=sim.fps)
            print(f"  → Saved: {ep_path}")

        # Run base policy on same task (for comparison)
        if base_policy is not None:
            base_result = run_episode(
                policy=base_policy,
                env=env,
                task_name=sim.task_suite,
                task_language=task.language,
                episode_idx=i,
                max_steps=sim.max_steps,
                record=sim.record_video,
                headless=True,          # always headless for base run
                display_fps=sim.fps,
            )
            base_results.append(base_result)

            if sim.record_video and result.frames and base_result.frames:
                cmp_path = output_dir / f"ep{i+1:03d}_comparison.mp4"
                save_comparison_mp4(base_result.frames, result.frames, cmp_path, fps=sim.fps)
                print(f"  → Comparison: {cmp_path}")

        env.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    n_success = sum(r.success for r in results)
    success_rate = n_success / len(results) if results else 0.0

    print("\n" + "─" * 58)
    print(f"  Fine-tuned — success rate: {success_rate:.0%}  ({n_success}/{len(results)})")
    if base_results:
        n_base = sum(r.success for r in base_results)
        base_rate = n_base / len(base_results) if base_results else 0.0
        print(f"  Base model — success rate: {base_rate:.0%}  ({n_base}/{len(base_results)})")
        delta = success_rate - base_rate
        print(f"  Improvement: {delta:+.0%}")
    if sim.record_video:
        print(f"\n  Videos saved to: {output_dir}/")
    print("─" * 58 + "\n")


if __name__ == "__main__":
    main()