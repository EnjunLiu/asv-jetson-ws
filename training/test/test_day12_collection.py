from __future__ import annotations

import json
from pathlib import Path

from training.dataset_registry import build_registry
from training.day12_collection import discover_slots, load_plan, validate_slot


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plan() -> dict:
    return {
        "minimum_frames_per_run": 2,
        "required_execution_mode": "ue5_kinematic_expert_v1",
        "required_entity_ids": [
            "target_red",
            "target_blue",
            "target_left",
            "target_right",
        ],
        "relation_margin_m": 0.25,
        "relation_evaluation_frames": 2,
        "minimum_relation_pass_fraction": 0.8,
        "motion_evaluation_frames": 2,
        "minimum_motion_pass_fraction": 0.6,
        "minimum_pairwise_velocity_difference_mps": 0.03,
    }


def _slot() -> dict:
    return {
        "slot_id": "L1_S0_R1",
        "layout_id": "L1",
        "motion_state": "S0",
        "scene_seed": 120101,
        "relations": [
            ["nearer", "target_red", "target_blue"],
            ["left_of", "target_left", "target_right"],
        ],
    }


def _entity(
    entity_id: str,
    color: str,
    x_value: float,
    y_value: float,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
) -> dict:
    return {
        "entity_id": entity_id,
        "color": color,
        "relative_position_m": [x_value, y_value, 0.0],
        "relative_velocity_mps": [velocity_x, velocity_y, 0.0],
        "valid": True,
        "visible": True,
        "is_target": True,
    }


def _make_run(
    tmp_path: Path,
    *,
    swap_depth: bool = False,
    motion_state: str = "S0",
    distinct_motion: bool = False,
) -> tuple[Path, Path]:
    episode = tmp_path / "artifacts" / "day8_episode" / "RUN_001"
    supervision = (
        tmp_path / "artifacts" / "day10_supervised" / "RUN_001"
    )
    manifest = {
        "run_id": "RUN_001",
        "scene_seed": 120101,
        "frame_count": 2,
        "status": "complete",
        "execution_mode": "ue5_kinematic_expert_v1",
        "collection": {
            "slot_id": f"L1_{motion_state}_R1",
            "layout_id": "L1",
            "motion_state": motion_state,
        },
    }
    _write(episode / "manifest.json", manifest)
    _write(
        episode / "quality_report.json",
        {"passed": True, "run_id": "RUN_001", "frame_count": 2},
    )
    red_x, blue_x = ((4.0, 1.0) if swap_depth else (1.0, 4.0))
    for frame_index in range(2):
        _write(
            episode / "frames" / f"{frame_index:012d}.json",
            {
                "entities": {
                    "items": [
                        _entity(
                            "target_red",
                            "red",
                            red_x,
                            0.0,
                            velocity_y=0.08 if distinct_motion else 0.0,
                        ),
                        _entity("target_blue", "blue", blue_x, 0.0),
                        _entity("target_left", "white", 2.0, 1.0),
                        _entity("target_right", "white", 2.0, -1.0),
                    ]
                }
            },
        )
    _write(
        supervision / "manifest.json",
        {
            "source_episodes": [{"run_id": "RUN_001"}],
            "samples": {"frame_count": 2, "sample_count": 180},
            "label_coverage": {
                "complete": True,
                "observed_labels": [f"label_{index}" for index in range(9)],
                "required_labels": [f"label_{index}" for index in range(9)],
            },
        },
    )
    return episode, supervision


def test_repository_plan_has_twelve_unique_counterbalanced_slots() -> None:
    plan_path = (
        Path(__file__).parents[1]
        / "config"
        / "day12_collection_plan_v1.json"
    )
    plan = load_plan(plan_path)

    assert len(plan["slots"]) == 12
    assert len({slot["slot_id"] for slot in plan["slots"]}) == 12
    assert len({slot["scene_seed"] for slot in plan["slots"]}) == 12
    assert {slot["layout_id"] for slot in plan["slots"]} == {
        "L1",
        "L2",
        "L3",
        "L4",
    }
    assert {slot["motion_state"] for slot in plan["slots"]} == {"S0", "S1"}
    assert {
        slot["layout_id"]
        for slot in plan["slots"]
        if slot["motion_state"] == "S0"
    } == {"L1", "L2", "L3", "L4"}
    assert plan["relation_evaluation_frames"] == 1


def test_slot_validator_checks_observed_geometry(tmp_path: Path) -> None:
    episode, supervision = _make_run(tmp_path)

    report = validate_slot(_slot(), episode, supervision, _plan())

    assert report["passed"]
    assert report["relation_pass_fractions"] == [1.0, 1.0]


def test_slot_validator_rejects_manifest_claim_when_geometry_is_wrong(
    tmp_path: Path,
) -> None:
    episode, supervision = _make_run(tmp_path, swap_depth=True)

    report = validate_slot(_slot(), episode, supervision, _plan())

    assert not report["passed"]
    assert any("relation" in error for error in report["errors"])


def test_s1_slot_requires_observable_distinct_target_motion(
    tmp_path: Path,
) -> None:
    episode, supervision = _make_run(tmp_path, motion_state="S1")
    slot = _slot()
    slot["slot_id"] = "L1_S1_R1"
    slot["motion_state"] = "S1"

    report = validate_slot(slot, episode, supervision, _plan())

    assert not report["passed"]
    assert report["motion_pass_fraction"] == 0.0
    assert any("distinct target motion" in error for error in report["errors"])


def test_s1_slot_accepts_distinct_target_motion(tmp_path: Path) -> None:
    episode, supervision = _make_run(
        tmp_path,
        motion_state="S1",
        distinct_motion=True,
    )
    slot = _slot()
    slot["slot_id"] = "L1_S1_R1"
    slot["motion_state"] = "S1"

    report = validate_slot(slot, episode, supervision, _plan())

    assert report["passed"]
    assert report["motion_pass_fraction"] == 1.0


def test_registry_requires_day12_metadata_and_twelve_runs(
    tmp_path: Path,
) -> None:
    episode, _ = _make_run(tmp_path)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 80
    _write(manifest_path, manifest)
    quality_path = episode / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["frame_count"] = 80
    _write(quality_path, quality)
    supervision_path = (
        tmp_path
        / "artifacts"
        / "day10_supervised"
        / "RUN_001"
        / "manifest.json"
    )
    supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
    supervision["samples"]["frame_count"] = 80
    _write(supervision_path, supervision)

    registry = tmp_path / "registry" / "dataset_registry_v1.jsonl"
    report = build_registry(tmp_path, registry)
    entry = json.loads(registry.read_text(encoding="utf-8"))

    assert entry["training_eligible"] is True
    assert report["eligible_run_count"] == 1
    assert report["training_ready"] is False
    assert report["minimum_runs_for_training"] == 12


def test_registry_reads_legacy_static_pilot_but_does_not_count_it(
    tmp_path: Path,
) -> None:
    episode, _ = _make_run(tmp_path)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_mode"] = "static"
    manifest.pop("collection")
    _write(manifest_path, manifest)

    registry = tmp_path / "registry" / "legacy.jsonl"
    report = build_registry(tmp_path, registry)
    entry = json.loads(registry.read_text(encoding="utf-8"))

    assert entry["execution_mode"] == "static"
    assert entry["training_eligible"] is False
    assert report["eligible_run_count"] == 0


def test_latest_symlink_is_not_a_duplicate_collection_slot(
    tmp_path: Path,
) -> None:
    _make_run(tmp_path)
    latest = tmp_path / "artifacts" / "day8_episode" / "latest"
    latest.symlink_to("RUN_001", target_is_directory=True)

    discovered, errors = discover_slots(tmp_path)

    assert set(discovered) == {"L1_S0_R1"}
    assert errors == []
