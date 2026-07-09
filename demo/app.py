"""PolicyForge — Gradio demo for HuggingFace Spaces.

Supports two modes, controlled by environment variables:

  API mode (preferred for Spaces without GPU):
    Set POLICYFORGE_API_URL=https://your-server:8000
    The demo calls your running FastAPI inference server.

  Direct mode (for Spaces with GPU or local use):
    Set CHECKPOINT_PATH=outputs/train/smolvla_libero_lora
    OR CHECKPOINT_REPO=your-username/smolvla-libero-lora (downloads from Hub)
    The demo loads the model directly.

Deploy to HuggingFace Spaces:
    git clone https://huggingface.co/spaces/your-username/policyforge-demo
    cp demo/app.py demo/requirements.txt demo/README.md .
    git add . && git commit -m "Add PolicyForge demo" && git push
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server use
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import gradio as gr


# ── Mode detection ────────────────────────────────────────────────────────────

def get_mode() -> str:
    """Detect which inference mode to use based on environment variables."""
    if os.environ.get("POLICYFORGE_API_URL"):
        return "api"
    if os.environ.get("CHECKPOINT_PATH") or os.environ.get("CHECKPOINT_REPO"):
        return "direct"
    return "unconfigured"


# ── Backend: API mode ─────────────────────────────────────────────────────────

def call_api(
    image_np: np.ndarray,
    instruction: str,
    state: list[float] | None,
) -> dict:
    """Call the PolicyForge FastAPI inference server."""
    import requests

    api_url = os.environ["POLICYFORGE_API_URL"].rstrip("/")
    buf = BytesIO()
    Image.fromarray(image_np).save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload: dict = {"image_base64": img_b64, "instruction": instruction}
    if state is not None:
        payload["state"] = state

    resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Backend: direct mode ──────────────────────────────────────────────────────

_policy = None


def call_direct(
    image_np: np.ndarray,
    instruction: str,
    state: list[float] | None,
) -> dict:
    """Load the policy and run inference directly."""
    import time
    import torch  # type: ignore[import-untyped]

    global _policy
    if _policy is None:
        from policyforge.simulation.env import load_policy, setup_rendering
        setup_rendering(headless=True)
        checkpoint = os.environ.get("CHECKPOINT_PATH") or os.environ.get("CHECKPOINT_REPO")
        _policy = load_policy(checkpoint)

    use_cuda = torch.cuda.is_available()
    image_t = (
        torch.from_numpy(image_np.copy()).float()
        .permute(2, 0, 1).unsqueeze(0).div(255.0)
    )
    state_vals = state if state is not None else [0.0] * 9
    state_t = torch.tensor([state_vals], dtype=torch.float32)
    if use_cuda:
        image_t = image_t.cuda()
        state_t = state_t.cuda()

    obs = {
        "observation.images.top": image_t,
        "observation.state":      state_t,
        "task":                   [instruction],
    }

    t0 = time.perf_counter()
    _policy.reset()
    with torch.no_grad():
        action = _policy.select_action(obs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    action_np = action.squeeze().cpu().numpy()
    if action_np.ndim == 1:
        action_np = action_np.reshape(1, -1)

    checkpoint_label = (
        os.environ.get("CHECKPOINT_PATH")
        or os.environ.get("CHECKPOINT_REPO", "unknown")
    )
    return {
        "actions":      action_np.tolist(),
        "action_dim":   int(action_np.shape[1]),
        "horizon":      int(action_np.shape[0]),
        "inference_ms": round(elapsed_ms, 2),
        "checkpoint":   checkpoint_label,
    }


# ── Visualization helpers ─────────────────────────────────────────────────────

_ACTION_LABELS = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def _action_labels(action_dim: int) -> list[str]:
    base  = _ACTION_LABELS[:action_dim]
    extra = [f"dim_{i}" for i in range(len(base), action_dim)]
    return base + extra


def plot_actions(actions: np.ndarray) -> plt.Figure:
    """Plot predicted action trajectory — one line per action dimension."""
    horizon, action_dim = actions.shape
    labels = _action_labels(action_dim)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, label in enumerate(labels):
        ax.plot(range(horizon), actions[:, i], label=label, marker="o", markersize=3)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Action value")
    ax.set_title("Predicted action trajectory")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def format_actions_table(actions: np.ndarray) -> list[list]:
    """Format action array as a list-of-lists for gr.Dataframe."""
    horizon, action_dim = actions.shape
    labels = _action_labels(action_dim)
    header = ["step"] + labels
    rows   = [
        [f"t={i}"] + [round(float(v), 4) for v in row]
        for i, row in enumerate(actions)
    ]
    return [header] + rows


# ── Main predict function ─────────────────────────────────────────────────────

def predict(
    image: np.ndarray | None,
    instruction: str,
    state_text: str,
) -> tuple:
    """Gradio handler — runs inference and returns all UI outputs."""
    if image is None:
        return None, None, "Upload a robot observation image to get started.", None

    if not instruction.strip():
        return None, None, "Enter a task instruction.", None

    state: list[float] | None = None
    if state_text.strip():
        try:
            state = [float(x.strip()) for x in state_text.split(",")]
        except ValueError:
            return None, None, "Invalid state. Use comma-separated floats: 0.0, 0.0, ...", None

    mode = get_mode()
    try:
        if mode == "api":
            result = call_api(image, instruction, state)
        elif mode == "direct":
            result = call_direct(image, instruction, state)
        else:
            return (
                None, None,
                "No model configured.\n\n"
                "Set POLICYFORGE_API_URL (API mode) or "
                "CHECKPOINT_PATH / CHECKPOINT_REPO (direct mode).",
                None,
            )
    except Exception as exc:
        return None, None, f"Inference error: {exc}", None

    actions = np.array(result["actions"])
    return (
        plot_actions(actions),
        format_actions_table(actions),
        (
            f"Inference: {result['inference_ms']:.1f} ms  |  "
            f"Horizon: {result['horizon']} steps  |  "
            f"Action dim: {result['action_dim']}"
        ),
        json.dumps(result, indent=2),
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_mode = get_mode()
_status = {
    "api":          f"API mode — {os.environ.get('POLICYFORGE_API_URL', '')}",
    "direct":       f"Direct mode — {os.environ.get('CHECKPOINT_PATH') or os.environ.get('CHECKPOINT_REPO', '')}",
    "unconfigured": "Not configured. Set POLICYFORGE_API_URL or CHECKPOINT_PATH.",
}[_mode]

EXAMPLE_INSTRUCTIONS = [
    "Pick up the red block and place it in the bowl.",
    "Stack the wooden blocks from largest to smallest.",
    "Push the T-shaped block to the green target.",
    "Open the drawer and place the object inside.",
    "Pick up the mug and set it on the plate.",
]

with gr.Blocks(title="PolicyForge — VLA Robot Policy Demo") as demo:

    gr.Markdown(f"""
# PolicyForge — VLA Robot Policy Demo

Fine-tuned [SmolVLA](https://huggingface.co/lerobot/smolvla_base) policy for robot manipulation tasks.
Upload a robot observation image, enter a task instruction, and see the predicted action trajectory.

**Status:** {_status}
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Robot observation", type="numpy", height=280)
            instruction_input = gr.Textbox(
                label="Task instruction",
                placeholder="Pick up the red block and place it in the bowl.",
                lines=2,
            )
            state_input = gr.Textbox(
                label="Robot state (optional)",
                placeholder="0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0",
                info="Comma-separated: eef_x, eef_y, eef_z, rx, ry, rz, gripper, ...",
            )
            submit_btn = gr.Button("Predict actions", variant="primary", size="lg")

        with gr.Column(scale=2):
            plot_output   = gr.Plot(label="Predicted action trajectory")
            status_output = gr.Textbox(label="Inference info", interactive=False)
            table_output  = gr.Dataframe(
                label="Predicted actions  (time step x action dim)",
                interactive=False,
                wrap=True,
            )

    with gr.Accordion("Raw JSON response", open=False):
        json_output = gr.Code(language="json", label="Full API response")

    gr.Examples(
        examples=[[None, instr, ""] for instr in EXAMPLE_INSTRUCTIONS],
        inputs=[image_input, instruction_input, state_input],
        label="Example instructions — upload your own robot observation image above",
    )

    gr.Markdown("""
---
### How it works

1. A robot camera captures an **observation image** of the workspace.
2. The image + a **natural-language instruction** are sent to the fine-tuned SmolVLA policy.
3. The policy predicts a **chunk of future robot actions** (x, y, z, rotation, gripper) for each timestep.
4. The robot executes these actions sequentially.

Trained with **PolicyForge** on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
manipulation tasks via LoRA fine-tuning.
Robot observation images can be captured from simulation (`make simulate`) or a real camera.
""")

    submit_btn.click(
        fn=predict,
        inputs=[image_input, instruction_input, state_input],
        outputs=[plot_output, table_output, status_output, json_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),
        share=False,
    )