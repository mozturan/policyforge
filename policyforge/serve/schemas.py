"""Pydantic schemas for the PolicyForge inference API.

These models define what the /predict endpoint accepts and returns.
They also auto-generate the OpenAPI documentation at /docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    """Observation sent to the policy for action prediction."""

    image_base64: str = Field(
        description="Base64-encoded RGB image (JPEG or PNG). "
                    "Resized server-side to the model's expected input resolution.",
        examples=["<base64 string>"],
    )
    instruction: str = Field(
        description="Natural language task instruction.",
        examples=["Pick up the red block and place it in the bowl."],
    )
    state: list[float] | None = Field(
        default=None,
        description="Optional robot proprioception: concatenated joint positions, "
                    "end-effector pose, gripper state. "
                    "Defaults to zeros if not provided.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "<base64-encoded JPEG>",
                "instruction": "Pick up the red block and place it in the bowl.",
                "state": None,
            }
        }
    }


class ActionResponse(BaseModel):
    """Predicted robot action chunk returned by the policy."""

    actions: list[list[float]] = Field(
        description="Predicted action sequence. "
                    "Shape: (horizon, action_dim). "
                    "Execute these actions sequentially on the robot."
    )
    action_dim: int = Field(description="Dimensionality of each action vector.")
    horizon: int = Field(description="Number of predicted future steps.")
    inference_ms: float = Field(description="Server-side inference time in milliseconds.")
    checkpoint: str = Field(description="Checkpoint path used for inference.")


class HealthResponse(BaseModel):
    """Server health and model status."""

    status: str = Field(description="'ok' if the server is running.")
    model_loaded: bool = Field(description="True if a policy is loaded in memory.")
    checkpoint: str | None = Field(
        default=None,
        description="Path of the loaded checkpoint, or null if no model is loaded.",
    )
    device: str = Field(description="'cuda' or 'cpu'.")