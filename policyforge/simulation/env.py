"""Environment setup and loading utilities for LIBERO simulation.

IMPORTANT: call setup_rendering() BEFORE importing anything from lerobot,
libero, or mujoco. MuJoCo reads MUJOCO_GL at import time, not at runtime.
Calling setup_rendering() after those imports has no effect.

Correct order in any script:
    from policyforge.simulation.env import setup_rendering
    setup_rendering(headless=True)          # <-- first
    from libero.libero.envs import ...      # <-- then import LIBERO
"""

from __future__ import annotations

import os
from pathlib import Path


def setup_rendering(headless: bool) -> None:
    """Set MuJoCo rendering backend environment variables.

    Must be called before any import of mujoco, robosuite, or libero.

    headless=True  → EGL (offscreen, no display needed, works on servers/CI)
    headless=False → GLFW (opens a live window, requires X11/Wayland display)

    For display mode on Linux without a desktop, you can use a virtual display:
        Xvfb :1 -screen 0 1280x720x24 &
        export DISPLAY=:1
    """
    if headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    else:
        os.environ["MUJOCO_GL"] = "glfw"
        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = ":0"

    mode = "EGL (headless)" if headless else "GLFW (display)"
    print(f"[simulation] Rendering backend: {mode}")


def make_libero_env(bddl_file: str, render_size: int):
    """Create a LIBERO OffScreenRenderEnv for a given task.

    Always uses OffScreenRenderEnv — it captures frames as numpy arrays
    regardless of the rendering backend. In headless mode those frames are
    saved to disk. In display mode they are shown live via OpenCV AND saved.

    Args:
        bddl_file:   Path to the task's BDDL definition file.
        render_size: Camera resolution in pixels (square).

    Returns:
        A LIBERO OffScreenRenderEnv instance, not yet reset.
    """
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file=bddl_file,
        camera_heights=render_size,
        camera_widths=render_size,
        render_camera="agentview",
    )
    return env


def load_policy(checkpoint_path: str | Path):
    """Load a SmolVLA policy from a lerobot-train checkpoint.

    lerobot-train saves a complete checkpoint directory that SmolVLAPolicy
    can load directly via from_pretrained(). This handles both full
    fine-tuning and LoRA checkpoints transparently.

    Args:
        checkpoint_path: Directory saved by lerobot-train (output_dir).

    Returns:
        SmolVLAPolicy on GPU in eval mode, ready for select_action().
    """
    import torch
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[simulation] Loading policy from: {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
    policy = policy.cuda().eval()

    param_count = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"[simulation] Policy loaded ({param_count:.0f}M params) on GPU")
    return policy


def get_libero_tasks(task_suite: str) -> list:
    """Return all tasks in a LIBERO suite.

    Args:
        task_suite: Suite name, e.g. "libero_spatial", "libero_goal",
                    "libero_object", "libero_100"

    Returns:
        List of task objects, each with .bddl_file and .language attributes.
    """
    from libero.libero import benchmark

    bm_dict = benchmark.get_benchmark_dict()
    if task_suite not in bm_dict:
        available = list(bm_dict.keys())
        raise ValueError(
            f"Unknown task suite: {task_suite!r}. Available: {available}"
        )

    bm = bm_dict[task_suite]
    return [bm.get_task(i) for i in range(bm.n_tasks)]