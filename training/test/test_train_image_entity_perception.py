from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from training.train_image_entity_perception import _read_samples


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
