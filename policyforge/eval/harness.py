"""Evaluation harness — orchestrates the full post-training benchmark.

Ties together: simulation rollouts, metric computation, tracker logging,
and report generation. This is the main entry point for evaluate.py.

Usage:
    from policyforge.eval.harness import run_eval
    from policyforge.runner import ExperimentConfig
    from policyforge.tracking import get_tracker

    cfg     = ExperimentConfig.from_yaml("configs/experiments/smolvla_libero_lora.yaml")
    tracker = get_tracker(cfg.tracking.backend, project=cfg.tracking.project)
    metrics = run_eval(
        checkpoint_path = "outputs/train/smolvla_libero_lora",
        sim_cfg         = cfg.simulation,
        tracker         = tracker,
        output_dir      = "outputs/train/smolvla_libero_lora/eval",
    )
"""

from __future__ import annotations

from pathlib import Path

from policyforge.eval.metrics import SuiteMetrics, compute_suite_metrics
from policyforge.eval.reporter import save_reports
from policyforge.runner import SimulationConfig
from policyforge.tracking.base import Tracker, NoOpTracker


def run_eval(
    checkpoint_path: str | Path,
    sim_cfg: SimulationConfig,
    tracker: Tracker | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> SuiteMetrics:
    """Run a full post-training evaluation benchmark.

    Runs N episodes per LIBERO task in the configured suite, aggregates
    success rates, logs results to the tracker, and saves Markdown + JSON
    reports to disk.

    Args:
        checkpoint_path: lerobot-train output directory to evaluate.
        sim_cfg:         SimulationConfig from the experiment YAML.
        tracker:         Tracker instance (WandB/MLflow/NoOp). Defaults to NoOp.
        output_dir:      Where to write eval_report.md and eval_report.json.
                         Defaults to <checkpoint_path>/eval/.
        verbose:         Print per-episode progress.

    Returns:
        SuiteMetrics with per-task and overall success rates.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir      = Path(output_dir) if output_dir else checkpoint_path / "eval"
    tracker         = tracker or NoOpTracker()

    # ── 1. Setup rendering BEFORE importing LIBERO ────────────────────────────
    # MuJoCo reads MUJOCO_GL at import time — this call must come first.
    from policyforge.simulation.env import setup_rendering
    setup_rendering(headless=sim_cfg.headless)

    # ── 2. Now safe to import the simulation stack ────────────────────────────
    from policyforge.simulation.env import get_libero_tasks, load_policy, make_libero_env
    from policyforge.simulation.recorder import save_mp4
    from policyforge.simulation.rollout import run_episode

    _log(f"Evaluating checkpoint : {checkpoint_path}", verbose)
    _log(f"Suite                 : {sim_cfg.task_suite}", verbose)
    _log(f"Episodes per task     : {sim_cfg.n_episodes}", verbose)
    _log(f"Mode                  : {'headless' if sim_cfg.headless else 'display'}", verbose)
    _log(f"Output                : {output_dir}", verbose)

    # ── 3. Load policy once — reused across all tasks ─────────────────────────
    policy = load_policy(str(checkpoint_path))
    tracker.log_params({"checkpoint": str(checkpoint_path), "suite": sim_cfg.task_suite})

    # ── 4. Get all tasks in the suite ─────────────────────────────────────────
    tasks = get_libero_tasks(sim_cfg.task_suite)
    _log(f"\nFound {len(tasks)} tasks in {sim_cfg.task_suite}\n", verbose)

    # ── 5. Run episodes per task ──────────────────────────────────────────────
    task_results: dict[str, list] = {}

    for task_idx, task in enumerate(tasks):
        task_label = task.language
        _log(f"Task {task_idx + 1:02d}/{len(tasks)}: {task_label[:60]}", verbose)

        env = make_libero_env(task.bddl_file, sim_cfg.render_size)
        episode_results = []

        for ep_idx in range(sim_cfg.n_episodes):
            result = run_episode(
                policy        = policy,
                env           = env,
                task_name     = sim_cfg.task_suite,
                task_language = task_label,
                episode_idx   = ep_idx,
                max_steps     = sim_cfg.max_steps,
                record        = sim_cfg.record_video,
                headless      = sim_cfg.headless,
            )
            episode_results.append(result)

            status = "✓" if result.success else "✗"
            _log(
                f"  [{status}] ep {ep_idx + 1:02d}/{sim_cfg.n_episodes}"
                f"  steps={result.steps_taken}"
                f"  {result.duration_seconds:.1f}s",
                verbose,
            )

            # Save episode video
            if sim_cfg.record_video and result.frames:
                label   = "success" if result.success else "fail"
                vid_dir = output_dir / "videos" / f"task_{task_idx + 1:02d}"
                vid_path = vid_dir / f"ep{ep_idx + 1:03d}_{label}.mp4"
                save_mp4(result.frames, vid_path, fps=sim_cfg.fps)

        env.close()
        task_results[task_label] = episode_results

        # Log per-task metrics to tracker immediately
        n_ok   = sum(1 for r in episode_results if r.success)
        rate   = n_ok / sim_cfg.n_episodes
        safe_key = task_label[:40].replace(" ", "_")
        tracker.log_metrics({f"eval/{safe_key}/success_rate": rate})
        _log(f"  → success rate: {rate:.0%}  ({n_ok}/{sim_cfg.n_episodes})\n", verbose)

    # ── 6. Compute and log suite-level metrics ────────────────────────────────
    metrics = compute_suite_metrics(
        suite_name      = sim_cfg.task_suite,
        checkpoint_path = str(checkpoint_path),
        task_results    = task_results,
    )

    tracker.log_metrics({
        "eval/overall_success_rate": metrics.overall_success_rate,
        "eval/total_episodes":       float(metrics.total_episodes),
        "eval/total_successes":      float(metrics.total_successes),
    })

    # ── 7. Save reports ───────────────────────────────────────────────────────
    report_paths = save_reports(metrics, output_dir)

    _log("\n" + "─" * 56, verbose)
    _log(f"Overall success rate : {metrics.overall_success_rate:.0%}"
         f"  ({metrics.total_successes}/{metrics.total_episodes})", verbose)
    _log(f"Report (Markdown)    : {report_paths['markdown']}", verbose)
    _log(f"Report (JSON)        : {report_paths['json']}", verbose)
    if sim_cfg.record_video:
        _log(f"Videos               : {output_dir}/videos/", verbose)
    _log("─" * 56 + "\n", verbose)

    # Upload Markdown report as tracker artifact
    tracker.log_artifact(str(report_paths["markdown"]), "eval_report.md")

    return metrics


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)