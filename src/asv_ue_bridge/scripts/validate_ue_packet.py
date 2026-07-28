#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _member(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_vector(
    value: Any,
    path: str,
    errors: list[str],
    axes: tuple[tuple[str, str], ...],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return
    for upper, lower in axes:
        if not _is_number(_member(value, upper, lower)):
            errors.append(f"{path}.{lower} must be finite")


def validate_packet(packet: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, dict):
        return ["packet must be a JSON object"]

    body = packet.get("Body", packet)
    if not isinstance(body, dict):
        return ["Body must be a JSON object"]

    run_id = _member(body, "Run_ID", "Run_Id", "RunId", "run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("Run_ID must be a non-empty string")

    scene_seed = _member(body, "Scene_Seed", "SceneSeed", "scene_seed")
    if not isinstance(scene_seed, int) or isinstance(scene_seed, bool):
        errors.append("Scene_Seed must be an integer")

    frame_index = _member(body, "Frame_Index", "FrameIndex", "frame_index")
    if (
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or frame_index < 0
    ):
        errors.append("Frame_Index must be a non-negative integer")

    simulation_time = _member(body, "Time", "time")
    if not _is_number(simulation_time) or simulation_time < 0:
        errors.append("Time must be a finite non-negative number")

    for field, aliases in (
        ("Surge_Velocity", ("Surge_Velocity", "SurgeVelocity", "surge_velocity")),
        (
            "Angular_Velocity",
            ("Angular_Velocity", "AngularVelocity", "angular_velocity", "YawRate"),
        ),
    ):
        if not _is_number(_member(body, *aliases)):
            errors.append(f"{field} must be finite")

    _check_vector(
        _member(body, "ASV_Location", "ASVLocation", "AsvLocation", "asv_location"),
        "ASV_Location",
        errors,
        (("X", "x"), ("Y", "y"), ("Z", "z")),
    )
    _check_vector(
        _member(
            body,
            "Target_Location",
            "TargetLocation",
            "target_location",
        ),
        "Target_Location",
        errors,
        (("X", "x"), ("Y", "y"), ("Z", "z")),
    )
    _check_vector(
        _member(body, "ASV_Rotation", "ASVRotation", "AsvRotation", "asv_rotation"),
        "ASV_Rotation",
        errors,
        (("Roll", "roll"), ("Pitch", "pitch"), ("Yaw", "yaw")),
    )

    entities = _member(body, "Entities", "entities")
    if not isinstance(entities, list):
        errors.append("Entities must be an array")
    elif len(entities) > 64:
        errors.append("Entities must contain at most 64 items")
    else:
        seen_ids: set[str] = set()
        for index, entity in enumerate(entities):
            path = f"Entities[{index}]"
            if not isinstance(entity, dict):
                errors.append(f"{path} must be an object")
                continue

            entity_id = _member(entity, "Entity_Id", "EntityId", "entity_id")
            if not isinstance(entity_id, str) or not entity_id.strip():
                errors.append(f"{path}.Entity_Id must be non-empty")
            elif entity_id in seen_ids:
                errors.append(f"{path}.Entity_Id is duplicated")
            else:
                seen_ids.add(entity_id)

            for field, aliases in (
                ("Class", ("Class", "class")),
                ("Color", ("Color", "color")),
            ):
                value = _member(entity, *aliases)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{path}.{field} must be non-empty")

            for field, aliases in (
                ("Is_Target", ("Is_Target", "IsTarget", "is_target")),
                ("Visible", ("Visible", "visible")),
            ):
                if not isinstance(_member(entity, *aliases), bool):
                    errors.append(f"{path}.{field} must be boolean")

            _check_vector(
                _member(
                    entity,
                    "RelativePosition",
                    "Relative_Position",
                    "relative_position",
                ),
                f"{path}.RelativePosition",
                errors,
                (("X", "x"), ("Y", "y"), ("Z", "z")),
            )
            _check_vector(
                _member(
                    entity,
                    "RelativeVelocity",
                    "Relative_Velocity",
                    "relative_velocity",
                ),
                f"{path}.RelativeVelocity",
                errors,
                (("X", "x"), ("Y", "y"), ("Z", "z")),
            )

    camera = _member(body, "Camera_Capture", "CameraCapture", "camera_capture")
    if camera is not None:
        if not isinstance(camera, list):
            errors.append("Camera_Capture must be an array")
        elif len(camera) > 8 * 1024 * 1024:
            errors.append("Camera_Capture exceeds 8 MiB")
        elif any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 255
            for value in camera
        ):
            errors.append("Camera_Capture values must be bytes")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one UE5 ObjectDeliverer JSON packet."
    )
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()

    try:
        payload = json.loads(args.packet.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"UE_PACKET_INVALID: {exc}", file=sys.stderr)
        return 1

    errors = validate_packet(payload)
    if errors:
        for error in errors:
            print(f"UE_PACKET_INVALID: {error}", file=sys.stderr)
        return 1

    print("UE_PACKET_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
