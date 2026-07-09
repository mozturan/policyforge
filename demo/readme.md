---
title: PolicyForge VLA Demo
emoji: 🤖
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

# PolicyForge — VLA Robot Policy Demo

Fine-tuned SmolVLA policy for robot manipulation, served via Gradio.

## Configuration

Set one of these environment variables in the Space settings:

| Variable | Mode | Description |
|---|---|---|
| `POLICYFORGE_API_URL` | API | URL of your running FastAPI server, e.g. `https://your-server:8000` |
| `CHECKPOINT_PATH` | Direct | Local path to a lerobot-train checkpoint |
| `CHECKPOINT_REPO` | Direct | HuggingFace model repo ID, e.g. `your-username/smolvla-libero-lora` |

API mode is recommended for Spaces without GPU. Direct mode requires a GPU Space.