"""Day 15 trajectory metrics and deterministic baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


MAXIMUM_STEP_M = 0.3
STOP_DRIFT_LIMIT_M = 0.10


def _as_trajectories(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (20, 2):
        raise ValueError(f"{name} must have shape [N,20,2], got {array.shape}")
    return array


def _binary_metrics(
    predicted: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _trajectory_error(
    prediction: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int]:
    if not np.any(indices):
        return {"sample_count": 0, "ade_m": 0.0, "fde_m": 0.0}
    error = np.linalg.norm(prediction[indices] - target[indices], axis=-1)
    return {
        "sample_count": int(np.count_nonzero(indices)),
        "ade_m": float(np.mean(error)),
        "fde_m": float(np.mean(error[:, -1])),
    }


def compute_policy_metrics(
    prediction: Any,
    target: Any,
    stop_logits: Any,
    target_stop: Any,
    task_labels: Iterable[str],
    *,
    maximum_step_m: float = MAXIMUM_STEP_M,
    stop_drift_limit_m: float = STOP_DRIFT_LIMIT_M,
) -> dict[str, Any]:
    predicted_trajectory = _as_trajectories(prediction, "prediction")
    target_trajectory = _as_trajectories(target, "target")
    if predicted_trajectory.shape != target_trajectory.shape:
        raise ValueError("prediction and target trajectory shapes differ")
    sample_count = predicted_trajectory.shape[0]
    logits = np.asarray(stop_logits, dtype=np.float32).reshape(-1)
    stop_target = np.asarray(target_stop, dtype=np.bool_).reshape(-1)
    labels = np.asarray([str(label) for label in task_labels], dtype=np.str_)
    if len(logits) != sample_count or len(stop_target) != sample_count:
        raise ValueError("stop arrays do not match trajectory sample count")
    if len(labels) != sample_count:
        raise ValueError("task labels do not match trajectory sample count")

    finite_rows = (
        np.all(np.isfinite(predicted_trajectory), axis=(1, 2))
        & np.isfinite(logits)
    )
    invalid_count = int(np.count_nonzero(~finite_rows))
    if invalid_count:
        raise ValueError(f"prediction contains {invalid_count} invalid samples")
    if not np.all(np.isfinite(target_trajectory)):
        raise ValueError("target trajectory contains NaN or Inf")

    all_rows = np.ones(sample_count, dtype=np.bool_)
    overall = _trajectory_error(
        predicted_trajectory, target_trajectory, all_rows
    )
    predicted_stop = logits >= 0.0
    classification = _binary_metrics(predicted_stop, stop_target)

    zero = np.zeros((sample_count, 1, 2), dtype=np.float32)
    increments = np.diff(
        np.concatenate((zero, predicted_trajectory), axis=1), axis=1
    )
    increment_norm = np.linalg.norm(increments, axis=-1)
    violation = increment_norm > maximum_step_m + 1.0e-6
    violation_count = int(np.count_nonzero(violation))

    stop_drift = (
        np.max(np.linalg.norm(predicted_trajectory[stop_target], axis=-1), axis=1)
        if np.any(stop_target)
        else np.zeros(0, dtype=np.float32)
    )
    stop_drift_metrics = {
        "sample_count": int(len(stop_drift)),
        "mean_m": float(np.mean(stop_drift)) if len(stop_drift) else 0.0,
        "p95_m": (
            float(np.percentile(stop_drift, 95)) if len(stop_drift) else 0.0
        ),
        "maximum_m": float(np.max(stop_drift)) if len(stop_drift) else 0.0,
        "within_0_10m_rate": (
            float(np.mean(stop_drift <= stop_drift_limit_m))
            if len(stop_drift)
            else 0.0
        ),
    }

    per_label: dict[str, Any] = {}
    for label in sorted(set(labels.tolist())):
        per_label[label] = _trajectory_error(
            predicted_trajectory, target_trajectory, labels == label
        )
    aggregate_labels: dict[str, Any] = {}
    categories: Mapping[str, np.ndarray] = {
        "red": np.char.find(labels, "|color:red|") >= 0,
        "blue": np.char.find(labels, "|color:blue|") >= 0,
        "left": np.char.find(labels, "|bearing:left|") >= 0,
        "right": np.char.find(labels, "|bearing:right|") >= 0,
        "3m": np.char.endswith(labels, "|3m"),
        "10m": np.char.endswith(labels, "|10m"),
        "stop": np.char.startswith(labels, "stop|"),
    }
    for name, indices in categories.items():
        aggregate_labels[name] = _trajectory_error(
            predicted_trajectory, target_trajectory, indices
        )
    return {
        "sample_count": sample_count,
        "ade_m": overall["ade_m"],
        "fde_m": overall["fde_m"],
        "stop_drift": stop_drift_metrics,
        "stop_classification": classification,
        "speed_constraint": {
            "maximum_step_m": maximum_step_m,
            "observed_maximum_step_m": float(np.max(increment_norm)),
            "violation_count": violation_count,
            "violation_rate": float(violation_count / violation.size),
        },
        "invalid_count": invalid_count,
        "per_label": per_label,
        "aggregate_labels": aggregate_labels,
    }


def fit_label_mean_baseline(
    trajectories: Any,
    task_labels: Iterable[str],
) -> dict[str, np.ndarray]:
    target = _as_trajectories(trajectories, "trajectories")
    labels = [str(label) for label in task_labels]
    if len(labels) != len(target):
        raise ValueError("task labels do not match trajectory sample count")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, trajectory in zip(labels, target):
        grouped[label].append(trajectory)
    return {
        label: np.mean(np.stack(rows), axis=0).astype(np.float32)
        for label, rows in sorted(grouped.items())
    }


def predict_label_mean_baseline(
    means: Mapping[str, np.ndarray],
    task_labels: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    labels = [str(label) for label in task_labels]
    missing = sorted(set(labels) - set(means))
    if missing:
        raise ValueError(f"mean baseline has no labels: {missing}")
    trajectories = np.stack([means[label] for label in labels]).astype(
        np.float32
    )
    stop_logits = np.asarray(
        [20.0 if label.startswith("stop|") else -20.0 for label in labels],
        dtype=np.float32,
    )
    return trajectories, stop_logits


def improvement_fraction(policy_value: float, baseline_value: float) -> float:
    if baseline_value <= 0.0:
        return 0.0
    return float((baseline_value - policy_value) / baseline_value)
