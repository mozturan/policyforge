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
    """Create a LIBERO OffScreenRenderEnv for a given task."""
    try:
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "LIBERO is not installed.\n"
            "Install it with:\n"
            "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
            "  cd LIBERO && pip install -e ."
        ) from e

    env = OffScreenRenderEnv(
        bddl_file=bddl_file,
        camera_heights=render_size,
        camera_widths=render_size,
        render_camera="agentview",
    )
    return env


def load_policy(checkpoint_path: str | Path):
    """Load a SmolVLA policy from a lerobot-train checkpoint directory."""
    import torch  # type: ignore[import-untyped]

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Could not import SmolVLAPolicy. Make sure lerobot is installed:\n"
            "  pip install lerobot\n\n"
            "If lerobot is installed but the import path differs in your version, "
            "find the correct path with:\n"
            "  python -c \"import lerobot, pathlib; "
            "[print(p) for p in pathlib.Path(lerobot.__file__).parent.rglob('modeling_smolvla.py')]\"\n"
            "Then update load_policy() in policyforge/simulation/env.py accordingly."
        ) from e

    print(f"[simulation] Loading policy from: {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
    policy = policy.cuda().eval()

    param_count = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"[simulation] Policy loaded ({param_count:.0f}M params) on GPU")
    return policy


def get_libero_tasks(task_suite: str) -> list:
    """Return all tasks in a LIBERO suite."""
    try:
        from libero.libero import benchmark  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "LIBERO is not installed. See make_libero_env() for install instructions."
        ) from e

    bm_dict = benchmark.get_benchmark_dict()
    if task_suite not in bm_dict:
        available = list(bm_dict.keys())
        raise ValueError(
            f"Unknown task suite: {task_suite!r}. Available: {available}"
        )

    bm = bm_dict[task_suite]
    return [bm.get_task(i) for i in range(bm.n_tasks)]