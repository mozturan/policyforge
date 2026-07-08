"""Policy loader with a module-level cache.

Loads a SmolVLA policy from a checkpoint directory exactly once,
then serves all subsequent requests from memory.

The cache is keyed by checkpoint path so switching checkpoints at
runtime (e.g. A/B testing two fine-tunes) is supported without restart.

Delegates the actual loading to policyforge.simulation.env.load_policy
so the loading logic lives in one place.
"""

from __future__ import annotations

import os
from typing import Any


# Module-level cache: checkpoint_path → loaded policy object
# Exposed so tests can monkeypatch it directly.
_cache: dict[str, Any] = {}


def get_checkpoint_path() -> str | None:
    """Return the checkpoint path from CHECKPOINT_PATH env var, or None."""
    return os.environ.get("CHECKPOINT_PATH") or None


def get_policy(checkpoint_path: str) -> Any:
    """Return a cached policy, loading it on first call.

    Args:
        checkpoint_path: Path to a lerobot-train output directory.

    Returns:
        SmolVLA policy on GPU in eval mode.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist on disk.
        ImportError: If lerobot is not installed.
    """
    if checkpoint_path not in _cache:
        # setup_rendering must be called before load_policy imports LIBERO/MuJoCo.
        # We call it here with headless=True since the serve process has no display.
        from policyforge.simulation.env import load_policy, setup_rendering
        setup_rendering(headless=True)
        _cache[checkpoint_path] = load_policy(checkpoint_path)

    return _cache[checkpoint_path]


def clear_cache() -> None:
    """Clear all cached policies (useful for testing or freeing GPU memory)."""
    _cache.clear()