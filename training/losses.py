"""Losses for the Day 14 single-trajectory policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as functional

from training.model import PolicyOutput


@dataclass(frozen=True)
class PolicyLossWeights:
    waypoint: float = 1.0
    endpoint: float = 0.5
    stop: float = 0.2
    smoothness: float = 0.05

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, float] | None
    ) -> "PolicyLossWeights":
        if value is None:
            return cls()
        unknown = set(value) - {"waypoint", "endpoint", "stop", "smoothness"}
        if unknown:
            raise ValueError(f"unknown loss weight keys: {sorted(unknown)}")
        return cls(**{key: float(item) for key, item in value.items()})

    def __post_init__(self) -> None:
        weights = (self.waypoint, self.endpoint, self.stop, self.smoothness)
        if any(not torch.isfinite(torch.tensor(value)) for value in weights):
            raise ValueError("loss weights must be finite")
        if any(value < 0.0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one loss weight must be positive")


def _validate_finite(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def trajectory_policy_loss(
    output: PolicyOutput,
    target_trajectory: Tensor,
    target_stop: Tensor,
    *,
    sample_valid: Tensor | None = None,
    weights: PolicyLossWeights | None = None,
) -> dict[str, Tensor]:
    """Compute the frozen Day 14 objective over valid samples only."""

    if output.trajectory.ndim != 3:
        raise ValueError("policy trajectory must have shape [B, horizon, action]")
    batch_size, horizon, action_dim = output.trajectory.shape
    expected_trajectory = (batch_size, horizon, action_dim)
    if tuple(target_trajectory.shape) != expected_trajectory:
        raise ValueError(
            f"target trajectory shape {tuple(target_trajectory.shape)} "
            f"does not match {expected_trajectory}"
        )
    if tuple(output.stop_logit.shape) != (batch_size, 1):
        raise ValueError("stop_logit must have shape [B, 1]")
    if tuple(target_stop.shape) not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("target_stop must have shape [B] or [B, 1]")
    if tuple(output.valid_mask.shape) != (batch_size,):
        raise ValueError("output.valid_mask must have shape [B]")

    valid = output.valid_mask
    if sample_valid is not None:
        if tuple(sample_valid.shape) != (batch_size,):
            raise ValueError("sample_valid must have shape [B]")
        valid = valid & sample_valid.to(device=valid.device, dtype=torch.bool)
    if not torch.any(valid):
        raise ValueError("loss batch has no valid samples")

    prediction = output.trajectory[valid]
    trajectory_target = target_trajectory.to(
        device=prediction.device, dtype=prediction.dtype
    )[valid]
    stop_prediction = output.stop_logit[valid]
    stop_target = target_stop.reshape(batch_size, 1).to(
        device=stop_prediction.device,
        dtype=stop_prediction.dtype,
    )[valid]
    _validate_finite(prediction, "policy trajectory")
    _validate_finite(trajectory_target, "target trajectory")
    _validate_finite(stop_prediction, "stop_logit")
    _validate_finite(stop_target, "target_stop")
    if torch.any((stop_target < 0.0) | (stop_target > 1.0)):
        raise ValueError("target_stop values must be in [0, 1]")

    waypoint = functional.smooth_l1_loss(prediction, trajectory_target)
    endpoint = functional.smooth_l1_loss(
        prediction[:, -1], trajectory_target[:, -1]
    )
    stop = functional.binary_cross_entropy_with_logits(
        stop_prediction, stop_target
    )
    if horizon >= 3:
        increments = prediction[:, 1:] - prediction[:, :-1]
        smoothness = functional.smooth_l1_loss(
            increments[:, 1:], increments[:, :-1]
        )
    else:
        smoothness = prediction.sum() * 0.0

    loss_weights = weights or PolicyLossWeights()
    total = (
        loss_weights.waypoint * waypoint
        + loss_weights.endpoint * endpoint
        + loss_weights.stop * stop
        + loss_weights.smoothness * smoothness
    )
    _validate_finite(total, "total loss")
    return {
        "total": total,
        "waypoint": waypoint,
        "endpoint": endpoint,
        "stop": stop,
        "smoothness": smoothness,
        "valid_samples": torch.tensor(
            int(torch.count_nonzero(valid)),
            device=total.device,
            dtype=torch.int64,
        ),
    }
