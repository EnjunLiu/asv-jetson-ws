"""Train the image-only entity geometry model from recorded UE episodes.

The input labels come from frame ``Entities`` only during training.  The
resulting model consumes RGB images alone at runtime; velocity is intentionally
not a model output and is supplied later by the temporal tracker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image

from asv_vla.image_entity_perception import (
    ENTITY_COUNT,
    ENTITY_IDS,
    OUTPUT_DIM,
    POSITION_SCALE_M,
    extract_image_features,
    save_model,
)
from asv_vla.visual_encoder import CameraProfile, project_target_to_pixel


CAMERA_PROFILE = CameraProfile()


def _read_samples(
    root: Path,
    *,
    max_primary_distance_m: float,
    max_abs_yaw_rad: float,
    max_abs_surge_velocity_mps: float,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int, int]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    run_ids: list[str] = []
    skipped_far = 0
    skipped_yaw = 0
    skipped_speed = 0
    for episode in sorted((p for p in root.iterdir() if p.is_dir())):
        manifest_path = episode / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_id", episode.name))
        frame_paths = sorted((episode / "frames").glob("*.json"))
        for frame_path in frame_paths:
            record = json.loads(frame_path.read_text(encoding="utf-8"))
            yaw = abs(float(record.get("ego", {}).get("rpy_ue_rad", [0.0, 0.0, 0.0])[2]))
            if yaw > max_abs_yaw_rad:
                skipped_yaw += 1
                continue
            surge_velocity = abs(
                float(record.get("ego", {}).get("surge_velocity_mps", 0.0))
            )
            if (
                not math.isfinite(surge_velocity)
                or surge_velocity > max_abs_surge_velocity_mps
            ):
                skipped_speed += 1
                continue
            distances = [
                math.hypot(
                    float(entity["relative_position_m"][0]),
                    float(entity["relative_position_m"][1]),
                )
                for entity in record["entities"]["items"]
            ]
            if not distances or min(distances) > max_primary_distance_m:
                skipped_far += 1
                continue
            image_path = episode / str(record["camera"]["image_path"])
            try:
                with Image.open(image_path) as image:
                    features.append(extract_image_features(image))
            except (OSError, KeyError, ValueError) as exc:
                raise RuntimeError(f"cannot read {frame_path}: {exc}") from exc
            by_id = {
                str(entity["entity_id"]): entity
                for entity in record["entities"]["items"]
            }
            output = np.zeros(OUTPUT_DIM, dtype=np.float32)
            for slot, entity_id in enumerate(ENTITY_IDS):
                entity = by_id.get(entity_id)
                if entity is None:
                    raise RuntimeError(f"{frame_path}: missing {entity_id}")
                x, y, z = entity["relative_position_m"]
                visible = bool(entity.get("visible", False))
                try:
                    project_target_to_pixel(x, y, z, CAMERA_PROFILE)
                    in_view = True
                except Exception:
                    in_view = False
                # UE visibility and camera visibility are distinct.  The
                # image model is trained on the latter so it cannot emit a
                # high-confidence entity that is outside the camera image.
                visible = visible and in_view
                offset = slot * 4
                output[offset] = 1.0 if visible else -1.0
                output[offset + 1 : offset + 4] = np.asarray(
                    (x, y, z), dtype=np.float32
                ) / POSITION_SCALE_M
            targets.append(output)
            run_ids.append(run_id)
    if not features:
        raise RuntimeError(f"no episode frames found under {root}")
    return (
        np.stack(features).astype(np.float32),
        np.stack(targets).astype(np.float32),
        run_ids,
        skipped_far,
        skipped_yaw,
        skipped_speed,
    )


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale = np.where(feature_scale < 1.0e-4, 1.0, feature_scale)
    normalized = (x - feature_mean) / feature_scale
    design = np.concatenate(
        [normalized, np.ones((len(normalized), 1), dtype=np.float32)], axis=1
    )
    # Dual form avoids inverting a 4k x 4k matrix when the number of recorded
    # frames is smaller than the pixel feature dimension.
    kernel = design @ design.T
    kernel.flat[:: kernel.shape[0] + 1] += float(ridge)
    alpha = np.linalg.solve(kernel, y)
    solution = design.T @ alpha
    return feature_mean, feature_scale, solution[:-1], solution[-1]


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    visibility = prediction[:, 0::4] >= 0.0
    target_visibility = target[:, 0::4] >= 0.0
    geometry_pred = prediction.reshape(-1, ENTITY_COUNT, 4)[:, :, 1:]
    geometry_target = target.reshape(-1, ENTITY_COUNT, 4)[:, :, 1:]
    rmse_normalized = float(np.sqrt(np.mean((geometry_pred - geometry_target) ** 2)))
    return {
        "visibility_accuracy": float(np.mean(visibility == target_visibility)),
        "geometry_rmse_normalized": rmse_normalized,
        "geometry_rmse_m": float(rmse_normalized * np.linalg.norm(POSITION_SCALE_M) / math.sqrt(3.0)),
        "frames": float(len(target)),
    }


def train(
    root: Path,
    output: Path,
    *,
    ridge: float = 1.0e-2,
    max_primary_distance_m: float = 5.0,
    max_abs_yaw_rad: float = 0.1,
    max_abs_surge_velocity_mps: float = 1.0,
) -> dict[str, object]:
    if max_primary_distance_m <= 0.0:
        raise ValueError("max_primary_distance_m must be positive")
    if max_abs_yaw_rad < 0.0:
        raise ValueError("max_abs_yaw_rad must be non-negative")
    if max_abs_surge_velocity_mps < 0.0:
        raise ValueError("max_abs_surge_velocity_mps must be non-negative")
    (
        x,
        y,
        run_ids,
        skipped_far,
        skipped_yaw,
        skipped_speed,
    ) = _read_samples(
        root,
        max_primary_distance_m=max_primary_distance_m,
        max_abs_yaw_rad=max_abs_yaw_rad,
        max_abs_surge_velocity_mps=max_abs_surge_velocity_mps,
    )
    unique_runs = sorted(set(run_ids))
    if len(unique_runs) < 2:
        raise RuntimeError("at least two runs are required for a group split")
    validation_runs = set(unique_runs[::5] or unique_runs[-1:])
    train_mask = np.asarray([run not in validation_runs for run in run_ids])
    val_mask = ~train_mask
    mean, scale, weights, bias = _ridge_fit(x[train_mask], y[train_mask], ridge)
    prediction = ((x - mean) / scale) @ weights + bias
    report = {
        "model_version": "image_entity_ridge_v1",
        "entity_ids": list(ENTITY_IDS),
        "input_shape": [18, 32, 7],
        "label_source": "frame_record_v1.entities",
        "velocity_output": False,
        "train_runs": sorted(set(np.asarray(run_ids)[train_mask].tolist())),
        "validation_runs": sorted(validation_runs),
        "train": _metrics(prediction[train_mask], y[train_mask]),
        "validation": _metrics(prediction[val_mask], y[val_mask]),
        "ridge": ridge,
        "max_primary_distance_m": max_primary_distance_m,
        "skipped_far_frames": skipped_far,
        "max_abs_yaw_rad": max_abs_yaw_rad,
        "skipped_yaw_frames": skipped_yaw,
        "max_abs_surge_velocity_mps": max_abs_surge_velocity_mps,
        "skipped_speed_frames": skipped_speed,
        "acceptance_ready": False,
        "acceptance_note": (
            "Collect additional close-range runs and retrain before claiming "
            "closed-loop perception acceptance."
        ),
    }
    save_model(
        output,
        feature_mean=mean,
        feature_scale=scale,
        weights=weights,
        bias=bias,
        metadata=report,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--max-primary-distance-m", type=float, default=5.0)
    parser.add_argument("--max-abs-yaw-rad", type=float, default=0.1)
    parser.add_argument(
        "--max-abs-surge-velocity-mps", type=float, default=1.0
    )
    args = parser.parse_args()
    report = train(
        args.episodes,
        args.output,
        ridge=args.ridge,
        max_primary_distance_m=args.max_primary_distance_m,
        max_abs_yaw_rad=args.max_abs_yaw_rad,
        max_abs_surge_velocity_mps=args.max_abs_surge_velocity_mps,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
