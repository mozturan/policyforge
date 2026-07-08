"""Video recording utilities for simulation rollouts.

All functions accept a list of numpy arrays (H, W, 3) uint8 and write
to disk. Output directory is created automatically if it doesn't exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 15) -> Path:
    """Save a list of frames as an MP4 video.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        path:   Output file path. Extension should be .mp4
        fps:    Frames per second.

    Returns:
        Resolved path of the saved file.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(str(path), fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    return path


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 10) -> Path:
    """Save a list of frames as a GIF.

    GIFs are lower quality than MP4 but display inline in GitHub READMEs.
    Useful for the portfolio README.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        path:   Output file path. Extension should be .gif
        fps:    Frames per second (GIFs are capped ~25 fps in practice).

    Returns:
        Resolved path of the saved file.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    imageio.mimsave(str(path), frames, duration=duration_ms, loop=0)
    return path


def save_comparison_mp4(
    base_frames: list[np.ndarray],
    ft_frames: list[np.ndarray],
    path: str | Path,
    fps: int = 15,
) -> Path:
    """Save two rollouts side-by-side as a single MP4.

    Useful for visually comparing the base model vs your fine-tuned model.
    The shorter rollout is padded by repeating its last frame.

    Layout: [base model | divider | fine-tuned]

    Args:
        base_frames: Frames from the base (non-fine-tuned) policy.
        ft_frames:   Frames from your fine-tuned policy.
        path:        Output file path.
        fps:         Frames per second.

    Returns:
        Resolved path of the saved file.
    """
    if not base_frames or not ft_frames:
        raise ValueError("Both base_frames and ft_frames must be non-empty.")

    # Pad the shorter rollout so both are the same length
    max_len = max(len(base_frames), len(ft_frames))
    base_frames = base_frames + [base_frames[-1]] * (max_len - len(base_frames))
    ft_frames   = ft_frames   + [ft_frames[-1]]   * (max_len - len(ft_frames))

    divider_width = 4
    combined: list[np.ndarray] = []

    for base_frame, ft_frame in zip(base_frames, ft_frames):
        h = base_frame.shape[0]
        divider = np.full((h, divider_width, 3), fill_value=100, dtype=np.uint8)
        combined.append(np.concatenate([base_frame, divider, ft_frame], axis=1))

    return save_mp4(combined, path, fps=fps)


def frames_to_gif_bytes(frames: list[np.ndarray], fps: int = 10) -> bytes:
    """Convert frames to GIF bytes (in-memory, no file written).

    Useful for serving the GIF directly from the FastAPI inference server
    or embedding it in a Gradio demo response.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        fps:    Frames per second.

    Returns:
        GIF file as raw bytes.
    """
    import io
    import imageio.v2 as imageio

    buf = io.BytesIO()
    duration_ms = int(1000 / fps)
    imageio.mimsave(buf, frames, format="gif", duration=duration_ms, loop=0)
    return buf.getvalue()