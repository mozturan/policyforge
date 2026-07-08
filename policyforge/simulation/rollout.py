"""Episode runner for LIBERO simulation.

Runs policy rollouts inside a LIBERO environment, collects frames,
and supports both headless (save to disk) and display (live window) modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeResult:
    """Result from a single simulation episode."""
    task_name: str
    task_language: str          # natural language description of the task
    episode_idx: int
    success: bool
    steps_taken: int
    duration_seconds: float
    frames: list = field(default_factory=list, repr=False)  # list of (H,W,3) uint8 arrays


def format_obs(libero_obs: dict, task_language: str) -> dict:  # noqa: F401
    import torch  # lazy — only when actually running inference
    """Convert a LIBERO observation dict to SmolVLA's expected input format.

    LIBERO returns a raw dict with numpy arrays. SmolVLA expects a specific
    dict of tensors. This function bridges the two.

    The observation keys (observation.images.top, observation.state) must
    match what the policy was trained on. If your training used different
    camera names, adjust the image key here.

    Args:
        libero_obs:    Raw observation dict from LIBERO env.step() / env.reset().
        task_language: Natural language task description from task.language.

    Returns:
        Dict of tensors ready for policy.select_action().
    """
    # ── Image ──────────────────────────────────────────────────────────────
    image = libero_obs["agentview_image"]           # (H, W, 3) uint8

    image_tensor = (
        torch.from_numpy(image.copy())
        .float()
        .permute(2, 0, 1)                           # (H,W,C) → (C,H,W)
        .unsqueeze(0)                               # → (1, C, H, W)
        .div(255.0)
    ).cuda()

    # ── Robot state ─────────────────────────────────────────────────────────
    # Concatenate available proprioception fields into a flat vector.
    # Adjust which keys to include based on your dataset's state definition.
    state_keys = [
        "robot0_eef_pos",       # end-effector position     (3,)
        "robot0_eef_quat",      # end-effector orientation  (4,)
        "robot0_gripper_qpos",  # gripper joint positions   (2,)
    ]
    state_parts = [
        libero_obs[k].flatten()
        for k in state_keys
        if k in libero_obs
    ]
    state = np.concatenate(state_parts) if state_parts else np.zeros(9)
    state_tensor = torch.from_numpy(state.copy()).float().unsqueeze(0).cuda()

    return {
        "observation.images.top": image_tensor,    # (1, 3, H, W)
        "observation.state": state_tensor,          # (1, state_dim)
        "task": [task_language],                    # list[str] for VLA language input
    }


def run_episode(
    policy,
    env,
    task_name: str,
    task_language: str,
    episode_idx: int,
    max_steps: int,
    record: bool = True,
    headless: bool = True,
    display_fps: int = 15,
) -> EpisodeResult:
    """Run a single episode and return the result.

    Args:
        policy:        Loaded SmolVLA policy (GPU, eval mode).
        env:           LIBERO OffScreenRenderEnv (already created).
        task_name:     Suite name, e.g. "libero_spatial".
        task_language: Natural language task description.
        episode_idx:   Episode number (used for display window title).
        max_steps:     Maximum environment steps before declaring failure.
        record:        Whether to collect frames for video saving.
        headless:      If False, show frames live in an OpenCV window.
        display_fps:   Target FPS for the live display (approximate).

    Returns:
        EpisodeResult with success flag, step count, duration, and frames.
    """
    import torch  # lazy — only when actually running inference
    obs = env.reset()
    policy.reset()              # clear action chunk buffer — critical before each episode

    frames: list[np.ndarray] = []
    start_time = time.time()
    success = False
    display_delay = max(1, int(1000 / display_fps))

    # Live display setup
    if not headless:
        _init_display(f"PolicyForge — episode {episode_idx + 1}")

    for step in range(max_steps):
        # Capture frame
        frame = env.render(mode="rgb_array")        # (H, W, 3) uint8

        if record:
            frames.append(frame)

        if not headless:
            _show_frame(frame, delay_ms=display_delay)

        # Policy inference
        obs_dict = format_obs(obs, task_language)
        with torch.no_grad():
            # select_action() manages the action chunk internally:
            # runs the full VLM forward pass only when the buffer is empty,
            # then dispenses one action at a time until the buffer refills.
            action = policy.select_action(obs_dict)

        # Step the environment
        obs, _reward, done, info = env.step(action.squeeze(0).cpu().numpy())

        if done or info.get("success", False):
            success = True
            if record:
                frames.append(env.render(mode="rgb_array"))  # capture final state
            break

    duration = time.time() - start_time

    if not headless:
        _close_display()

    return EpisodeResult(
        task_name=task_name,
        task_language=task_language,
        episode_idx=episode_idx,
        success=success,
        steps_taken=step + 1,
        duration_seconds=duration,
        frames=frames,
    )


# ── Display helpers ───────────────────────────────────────────────────────────
# OpenCV is only imported when headless=False.
# Install with: pip install -e ".[display]"

def _init_display(window_title: str) -> None:
    try:
        import cv2
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 512, 512)
    except ImportError:
        raise ImportError(
            "Display mode requires OpenCV. "
            'Install it with: pip install -e ".[display]"'
        )


def _show_frame(frame: np.ndarray, delay_ms: int = 33) -> None:
    import cv2
    # OpenCV uses BGR, LIBERO renders RGB — convert
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(list(cv2.getWindowProperty.__doc__ or ["PolicyForge"])[0]
               if False else "PolicyForge", bgr)
    key = cv2.waitKey(delay_ms)
    if key == ord("q"):
        raise KeyboardInterrupt("User pressed 'q' to quit simulation.")


def _close_display() -> None:
    try:
        import cv2
        cv2.destroyAllWindows()
    except ImportError:
        pass