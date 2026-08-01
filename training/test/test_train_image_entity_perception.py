from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.train_image_entity_perception import (
    _acceptance_gate,
    _metrics,
    _read_samples,
)


ENTITY_IDS = ("target_red", "target_blue", "target_left", "target_right")


def _write_frame(
    episode: Path,
    *,
    frame_index: int,
    surge_velocity_mps: float,
) -> None:
    image_path = episode / "camera" / f"{frame_index:012d}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (20, 30, 40)).save(image_path, format="JPEG")
    frame = {
        "ego": {
            "rpy_ue_rad": [0.0, 0.0, 0.0],
            "surge_velocity_mps": surge_velocity_mps,
        },
        "camera": {"image_path": str(image_path.relative_to(episode))},
        "entities": {
            "items": [
                {
                    "entity_id": entity_id,
                    "relative_position_m": [3.0, 0.0, 0.0],
                    "visible": True,
                }
                for entity_id in ENTITY_IDS
            ]
        },
    }
    frame_path = episode / "frames" / f"{frame_index:012d}.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_text(json.dumps(frame), encoding="utf-8")


def test_read_samples_skips_excessive_surge_velocity(tmp_path: Path) -> None:
    episode = tmp_path / "RUN_001"
    (episode / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (episode / "manifest.json").write_text(
        json.dumps({"run_id": "RUN_001"}), encoding="utf-8"
    )
    _write_frame(episode, frame_index=0, surge_velocity_mps=0.2)
    _write_frame(episode, frame_index=1, surge_velocity_mps=1.2)

    features, targets, run_ids, skipped_far, skipped_yaw, skipped_speed = (
        _read_samples(
            tmp_path,
            max_primary_distance_m=5.0,
            max_abs_yaw_rad=0.1,
            max_abs_surge_velocity_mps=1.0,
        )
    )

    assert features.shape[0] == targets.shape[0] == len(run_ids) == 1
    assert skipped_far == 0
    assert skipped_yaw == 0
    assert skipped_speed == 1


def test_metrics_excludes_geometry_for_camera_invisible_slots() -> None:
    target = np.zeros((1, 16), dtype=np.float32)
    target[0, 4::4] = -1.0
    target[0, 0] = 1.0
    target[0, 1:4] = 1.0
    target[0, 5:8] = 100.0
    prediction = np.zeros_like(target)
    metrics = _metrics(prediction, target)
    assert metrics["visible_geometry_slots"] == 1
    assert metrics["geometry_rmse_normalized"] == 1.0


def test_acceptance_gate_passes_overall_and_each_validation_run() -> None:
    validation = {
        "visibility_accuracy": 0.97,
        "visible_geometry_rmse_m": 0.42,
    }
    validation_by_run = {
        "RUN_A": {
            "visibility_accuracy": 0.96,
            "visible_geometry_rmse_m": 0.49,
            "visible_geometry_slots": 12,
        },
        "RUN_B": {
            "visibility_accuracy": 0.98,
            "visible_geometry_rmse_m": 0.31,
            "visible_geometry_slots": 8,
        },
    }
    result = _acceptance_gate(
        validation,
        validation_by_run,
        velocity_output=False,
        geometry_metric_mask="camera_projected_visibility_only",
    )
    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["thresholds"]["min_visibility_accuracy"] == 0.95


def test_acceptance_gate_reports_failed_run_and_contract() -> None:
    validation = {
        "visibility_accuracy": 0.99,
        "visible_geometry_rmse_m": 0.20,
    }
    validation_by_run = {
        "RUN_BAD": {
            "visibility_accuracy": 0.94,
            "visible_geometry_rmse_m": 0.60,
            "visible_geometry_slots": 10,
        }
    }
    result = _acceptance_gate(
        validation,
        validation_by_run,
        velocity_output=True,
        geometry_metric_mask="unmasked",
    )
    assert result["passed"] is False
    assert "velocity_output_false" in result["failed_checks"]
    assert "geometry_metric_mask" in result["failed_checks"]
    assert "validation_runs" in result["failed_checks"]
    assert result["validation_by_run"]["RUN_BAD"]["passed"] is False
