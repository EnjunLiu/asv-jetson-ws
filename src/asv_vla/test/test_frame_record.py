import copy
import json
from pathlib import Path

import pytest

from asv_vla.frame_record import (
    FrameRecordError,
    read_frame_record,
    validate_frame_record,
    write_frame_record,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PACKAGE_DIR / "examples" / "frame_record_v1.json"


def sample_record():
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def test_checked_in_frame_record_is_valid():
    validate_frame_record(sample_record())


def test_frame_record_round_trip_is_lossless_and_atomic(tmp_path):
    destination = tmp_path / "nested" / "frame.json"
    expected = sample_record()

    write_frame_record(destination, expected)

    assert read_frame_record(destination) == expected
    assert not list(destination.parent.glob("*.tmp"))


def test_wrong_shape_and_nonfinite_values_are_rejected():
    wrong_shape = sample_record()
    wrong_shape["ego"]["position_m"] = [0.0, 0.0]
    with pytest.raises(FrameRecordError, match="position_m"):
        validate_frame_record(wrong_shape)

    nonfinite = sample_record()
    nonfinite["entities"]["items"][0]["relative_position_m"][0] = float("nan")
    with pytest.raises(FrameRecordError, match="must be finite"):
        validate_frame_record(nonfinite)


def test_modality_mask_and_record_validity_must_agree():
    record = sample_record()
    record["camera"]["valid"] = False
    record["camera"]["image_path"] = None

    with pytest.raises(FrameRecordError, match="modality_mask.camera"):
        validate_frame_record(record)

    record["modality_mask"]["camera"] = False
    with pytest.raises(FrameRecordError, match="valid must equal"):
        validate_frame_record(record)

    record["valid"] = False
    with pytest.raises(FrameRecordError, match="must explain"):
        validate_frame_record(record)

    record["detail"] = "camera missing"
    validate_frame_record(record)


def test_synchronized_modalities_require_the_frame_timestamp():
    record = sample_record()
    record["camera"]["stamp_us"] += 1
    with pytest.raises(FrameRecordError, match="camera.stamp_us"):
        validate_frame_record(record)

    record = sample_record()
    record["task"]["stamp_us"] = record["stamp_us"] + 1
    with pytest.raises(FrameRecordError, match="must not be in the future"):
        validate_frame_record(record)


def test_duplicate_entities_and_unsafe_image_paths_are_rejected():
    duplicate = sample_record()
    duplicate["entities"]["items"].append(
        copy.deepcopy(duplicate["entities"]["items"][0])
    )
    with pytest.raises(FrameRecordError, match="duplicate entity_id"):
        validate_frame_record(duplicate)

    unsafe_path = sample_record()
    unsafe_path["camera"]["image_path"] = "../outside.jpg"
    with pytest.raises(FrameRecordError, match="safe relative path"):
        validate_frame_record(unsafe_path)


def test_camera_profile_is_frozen():
    record = sample_record()
    record["camera"]["width_px"] = 640
    with pytest.raises(FrameRecordError, match="1280"):
        validate_frame_record(record)

    record = sample_record()
    record["camera"]["mount_position_m"][0] = 0.0
    with pytest.raises(FrameRecordError, match="Day 4 profile"):
        validate_frame_record(record)

    record = sample_record()
    record["camera"]["mount_position_m"][0] = "not-a-number"
    with pytest.raises(FrameRecordError, match="mount_position_m"):
        validate_frame_record(record)


def test_reader_rejects_nonstandard_nan_and_duplicate_keys(tmp_path):
    nan_path = tmp_path / "nan.json"
    nan_path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(FrameRecordError, match="non-finite"):
        read_frame_record(nan_path)

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"run_id": "a", "run_id": "b"}', encoding="utf-8")
    with pytest.raises(FrameRecordError, match="duplicate JSON key"):
        read_frame_record(duplicate_path)
