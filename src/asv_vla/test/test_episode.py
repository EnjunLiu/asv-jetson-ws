from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from asv_vla.episode import (
    evaluate_episode,
    make_manifest,
    write_episode_frame,
    write_json_atomic,
)


def jpeg_bytes(width=1280, height=720):
    stream = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(
        stream, format="JPEG"
    )
    return stream.getvalue()


def messages(
    frame_index,
    *,
    run_id="RUN_001",
    scene_seed=12345,
    stamp_us=None,
):
    if stamp_us is None:
        stamp_us = 1_000_000 + frame_index * 100_000
    common = {
        "stamp_us": stamp_us,
        "run_id": run_id,
        "scene_seed": scene_seed,
        "frame_index": frame_index,
        "valid": True,
    }
    state = SimpleNamespace(
        **common,
        simulation_time=stamp_us / 1_000_000.0,
        position_x=1.0,
        position_y=2.0,
        position_z=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.1,
        surge_velocity=0.2,
        yaw_rate=0.01,
    )
    camera = SimpleNamespace(
        **common,
        encoding="jpeg",
        data=jpeg_bytes(),
    )
    entity = SimpleNamespace(
        entity_id="target_01",
        class_name="boat",
        color="red",
        is_target=True,
        visible=True,
        relative_x=1.5,
        relative_y=0.0,
        relative_z=-0.1,
        relative_velocity_x=0.0,
        relative_velocity_y=0.0,
        relative_velocity_z=0.0,
        valid=True,
    )
    entities = SimpleNamespace(
        **common,
        frame_id="base_link",
        entities=[entity],
    )
    return state, camera, entities


def write_test_episode(episode_dir: Path, frame_indices=(0, 1)):
    stamps = []
    for frame_index in frame_indices:
        state, camera, entities = messages(frame_index)
        write_episode_frame(
            episode_dir,
            task_text="follow the red boat",
            task_stamp_us=0,
            state=state,
            camera=camera,
            entities=entities,
        )
        stamps.append(state.stamp_us)
    manifest = make_manifest(
        run_id="RUN_001",
        scene_seed=12345,
        task_text="follow the red boat",
        frame_indices=frame_indices,
        stamp_values=stamps,
        status="complete",
    )
    write_json_atomic(episode_dir / "manifest.json", manifest)


def test_complete_episode_round_trip_and_quality_report(tmp_path):
    episode_dir = tmp_path / "episode"
    write_test_episode(episode_dir)

    report = evaluate_episode(episode_dir, min_frames=2)

    assert report["passed"]
    assert report["frame_count"] == 2
    assert report["frame_gaps"] == 0
    assert report["all_modalities_valid"]
    assert report["all_numbers_finite"]
    assert report["execution_mode"] == "observation_only"
    assert report["camera_shape_px"] == [1280, 720]
    assert json.loads(
        (episode_dir / "quality_report.json").read_text(encoding="utf-8")
    ) == report
    assert not list(episode_dir.rglob("*.tmp"))


def test_manifest_records_and_validates_execution_mode(tmp_path):
    episode_dir = tmp_path / "episode"
    write_test_episode(episode_dir)
    manifest_path = episode_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_mode"] = "ue5_kinematic_expert_v1"
    write_json_atomic(manifest_path, manifest)

    report = evaluate_episode(episode_dir, min_frames=2)

    assert report["passed"]
    assert report["execution_mode"] == "ue5_kinematic_expert_v1"

    manifest["execution_mode"] = "mixed_or_unknown"
    write_json_atomic(manifest_path, manifest)
    failed = evaluate_episode(episode_dir, min_frames=2)
    assert not failed["passed"]
    assert "manifest execution_mode is invalid" in failed["errors"]


def test_frame_gap_is_reported_as_warning_not_corruption(tmp_path):
    episode_dir = tmp_path / "episode"
    write_test_episode(episode_dir, frame_indices=(2, 5))

    report = evaluate_episode(episode_dir, min_frames=2)

    assert report["passed"]
    assert report["frame_gaps"] == 2
    assert report["warnings"] == ["UE transport dropped 2 frame(s)"]


def test_wrong_jpeg_shape_fails_quality_gate(tmp_path):
    episode_dir = tmp_path / "episode"
    write_test_episode(episode_dir, frame_indices=(0,))
    image_path = episode_dir / "camera" / "000000000000.jpg"
    image_path.write_bytes(jpeg_bytes(640, 480))

    report = evaluate_episode(episode_dir)

    assert not report["passed"]
    assert any("image shape" in error for error in report["errors"])


def test_manifest_mismatch_fails_quality_gate(tmp_path):
    episode_dir = tmp_path / "episode"
    write_test_episode(episode_dir)
    manifest_path = episode_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 999
    write_json_atomic(manifest_path, manifest)

    report = evaluate_episode(episode_dir)

    assert not report["passed"]
    assert "manifest frame_count does not match records" in report["errors"]


def test_equal_adjacent_timestamps_are_allowed_and_reported(tmp_path):
    episode_dir = tmp_path / "episode"
    shared_stamp = 1_234_567
    for frame_index in (0, 1):
        state, camera, entities = messages(
            frame_index, stamp_us=shared_stamp
        )
        write_episode_frame(
            episode_dir,
            task_text="follow the red boat",
            task_stamp_us=0,
            state=state,
            camera=camera,
            entities=entities,
        )
    manifest = make_manifest(
        run_id="RUN_001",
        scene_seed=12345,
        task_text="follow the red boat",
        frame_indices=[0, 1],
        stamp_values=[shared_stamp, shared_stamp],
        status="complete",
    )
    write_json_atomic(episode_dir / "manifest.json", manifest)

    report = evaluate_episode(episode_dir, min_frames=2)

    assert report["passed"]
    assert report["duplicate_timestamps"] == 1
    assert report["warnings"] == [
        "1 adjacent frame pair(s) share stamp_us"
    ]
