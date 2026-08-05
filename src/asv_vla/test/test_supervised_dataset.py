import copy
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from asv_vla.episode import make_manifest, write_json_atomic
from asv_vla.frame_record import write_frame_record
from asv_vla.language_intervention_dataset import write_jsonl
from asv_vla.supervised_dataset import (
    SupervisedDatasetError,
    build_supervised_dataset,
    evaluate_supervised_dataset,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PACKAGE_DIR / "examples" / "frame_record_v1.json"


def _instruction(
    instruction_id,
    action,
    target_attribute,
    distance_bucket,
    split="train",
):
    return {
        "instruction_id": instruction_id,
        "text": f"instruction {instruction_id}",
        "action": action,
        "target_attribute": target_attribute,
        "distance_bucket": distance_bucket,
        "split": split,
    }


def _all_instructions():
    labels = [
        ("follow", "color:red", "3m"),
        ("follow", "color:red", "4m"),
        ("follow", "color:red", "10m"),
        ("follow", "color:blue", "3m"),
        ("follow", "color:blue", "4m"),
        ("follow", "color:blue", "10m"),
        ("follow", "bearing:left", "3m"),
        ("follow", "bearing:left", "4m"),
        ("follow", "bearing:left", "10m"),
        ("follow", "bearing:right", "3m"),
        ("follow", "bearing:right", "4m"),
        ("follow", "bearing:right", "10m"),
        ("stop", "none", "none"),
    ]
    splits = ["train", "validation", "test"]
    return [
        _instruction(
            f"instruction_{index:02d}",
            action,
            attribute,
            distance,
            splits[index % len(splits)],
        )
        for index, (action, attribute, distance) in enumerate(labels)
    ]


def _entity(entity_id, color, x, y):
    return {
        "entity_id": entity_id,
        "class_name": "boat",
        "color": color,
        "is_target": True,
        "visible": True,
        "relative_position_m": [x, y, 0.0],
        "relative_velocity_mps": [0.1, 0.0, 0.0],
        "valid": True,
    }


def _make_episode(tmp_path, *, all_targets=True, frame_count=2):
    episode = tmp_path / "episodes" / "RUN_01"
    template = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    entities = [_entity("target_red", "red", 8.0, 0.0)]
    if all_targets:
        entities.extend(
            [
                _entity("target_blue", "blue", 9.0, 0.0),
                _entity("target_left", "white", 10.0, 4.0),
                _entity("target_right", "white", 10.0, -4.0),
            ]
        )

    frame_indices = []
    stamps = []
    for frame_index in range(frame_count):
        stamp_us = 100000 + frame_index * 100000
        image_relative = f"camera/{frame_index:012d}.jpg"
        image_path = episode / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1280, 720), (10, 20, 30)).save(
            image_path, format="JPEG"
        )

        record = copy.deepcopy(template)
        record["run_id"] = "RUN_01"
        record["scene_seed"] = 24680
        record["frame_index"] = frame_index
        record["stamp_us"] = stamp_us
        record["task"]["stamp_us"] = 0
        record["task"]["text"] = "intervention scene"
        record["ego"]["stamp_us"] = stamp_us
        record["ego"]["simulation_time_s"] = stamp_us / 1_000_000.0
        record["camera"]["stamp_us"] = stamp_us
        record["camera"]["image_path"] = image_relative
        record["entities"]["stamp_us"] = stamp_us
        record["entities"]["items"] = copy.deepcopy(entities)
        write_frame_record(
            episode / f"frames/{frame_index:012d}.json",
            record,
            image_root=episode,
        )
        frame_indices.append(frame_index)
        stamps.append(stamp_us)

    manifest = make_manifest(
        run_id="RUN_01",
        scene_seed=24680,
        task_text="intervention scene",
        frame_indices=frame_indices,
        stamp_values=stamps,
        status="complete",
    )
    write_json_atomic(episode / "manifest.json", manifest)
    return episode


def _write_instructions(tmp_path, records=None):
    path = tmp_path / "dataset" / "language" / "instructions.jsonl"
    write_jsonl(path, records or _all_instructions())
    return path


def test_build_and_evaluate_complete_dataset(tmp_path):
    episode = _make_episode(tmp_path)
    instructions = _write_instructions(tmp_path)
    output = tmp_path / "artifacts" / "supervised" / "complete"

    build_report = build_supervised_dataset(
        [episode], instructions, output
    )
    evaluation = evaluate_supervised_dataset(
        output, require_all_labels=True
    )

    assert build_report["sample_count"] == 26
    assert build_report["coverage_complete"] is True
    assert evaluation["sample_count"] == 26
    assert evaluation["frame_count"] == 2
    assert evaluation["compatible_instruction_count"] == 13
    assert evaluation["observed_label_count"] == 13

    samples = [
        json.loads(line)
        for line in (output / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert samples
    for sample in samples:
        action = sample["expert"]["expert_action"]
        assert isinstance(action, list)
        assert len(action) == 2
        assert "horizon" not in sample["expert"]
        assert "delta_p_xy" not in sample["expert"]
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["expert_action_contract"]["action_shape"] == [2]
    assert "horizon" not in manifest["expert_action_contract"]


def test_build_is_byte_deterministic(tmp_path):
    episode = _make_episode(tmp_path)
    instructions = _write_instructions(tmp_path)
    first = tmp_path / "artifacts" / "first"
    second = tmp_path / "artifacts" / "second"

    build_supervised_dataset([episode], instructions, first)
    build_supervised_dataset([episode], instructions, second)

    assert (first / "samples.jsonl").read_bytes() == (
        second / "samples.jsonl"
    ).read_bytes()
    first_manifest = json.loads(
        (first / "manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (second / "manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest == second_manifest


def test_partial_entity_scene_is_reported_not_mislabeled(tmp_path):
    episode = _make_episode(tmp_path, all_targets=False)
    instructions = _write_instructions(tmp_path)
    output = tmp_path / "artifacts" / "partial"

    report = build_supervised_dataset([episode], instructions, output)
    checked = evaluate_supervised_dataset(output)

    assert report["coverage_complete"] is False
    assert report["observed_label_count"] == 4
    assert checked["sample_count"] == 8
    with pytest.raises(
        SupervisedDatasetError, match="missing required task labels"
    ):
        evaluate_supervised_dataset(output, require_all_labels=True)


def test_samples_hash_tampering_is_rejected(tmp_path):
    episode = _make_episode(tmp_path)
    instructions = _write_instructions(tmp_path)
    output = tmp_path / "artifacts" / "tampered_samples"
    build_supervised_dataset([episode], instructions, output)

    with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    with pytest.raises(SupervisedDatasetError, match="samples sha256"):
        evaluate_supervised_dataset(output)


@pytest.mark.parametrize("legacy_shape", ["fields", "length"])
def test_legacy_trajectory_fields_and_shapes_are_rejected(
    tmp_path, legacy_shape
):
    episode = _make_episode(tmp_path)
    instructions = _write_instructions(tmp_path)
    output = tmp_path / "artifacts" / f"legacy_{legacy_shape}"
    build_supervised_dataset([episode], instructions, output)

    samples_path = output / "samples.jsonl"
    samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]
    expert = samples[0]["expert"]
    if legacy_shape == "fields":
        expert.pop("expert_action")
        expert["horizon"] = 20
        expert["delta_p_xy"] = [[0.0, 0.0]] * 20
        expected_error = "legacy"
    else:
        expert["expert_action"] = [0.0, 0.0, 0.0]
        expected_error = "expert_action"
    samples_path.write_text(
        "\n".join(
            json.dumps(
                sample,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for sample in samples
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"]["sha256"] = hashlib.sha256(
        samples_path.read_bytes()
    ).hexdigest()
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(SupervisedDatasetError, match=expected_error):
        evaluate_supervised_dataset(output)


def test_source_image_tampering_is_rejected(tmp_path):
    episode = _make_episode(tmp_path)
    instructions = _write_instructions(tmp_path)
    output = tmp_path / "artifacts" / "tampered_image"
    build_supervised_dataset([episode], instructions, output)

    image_path = episode / "camera" / "000000000000.jpg"
    image_path.write_bytes(image_path.read_bytes() + b"changed")

    with pytest.raises(SupervisedDatasetError, match="image_sha256"):
        evaluate_supervised_dataset(output)


def test_existing_output_is_never_overwritten(tmp_path):
    episode = _make_episode(tmp_path)
    instructions = _write_instructions(tmp_path)
    output = tmp_path / "artifacts" / "existing"
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(SupervisedDatasetError, match="output already exists"):
        build_supervised_dataset([episode], instructions, output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_duplicate_instruction_id_is_rejected(tmp_path):
    episode = _make_episode(tmp_path)
    duplicate = _instruction(
        "duplicate", "follow", "color:red", "3m"
    )
    instructions = _write_instructions(
        tmp_path, [duplicate, copy.deepcopy(duplicate)]
    )

    with pytest.raises(SupervisedDatasetError, match="duplicate"):
        build_supervised_dataset(
            [episode], instructions, tmp_path / "artifacts" / "duplicate"
        )


def test_no_compatible_target_fails_closed(tmp_path):
    episode = _make_episode(tmp_path, all_targets=False)
    instructions = _write_instructions(
        tmp_path,
        [_instruction("blue_only", "follow", "color:blue", "3m")],
    )

    with pytest.raises(
        SupervisedDatasetError, match="no instruction is compatible"
    ):
        build_supervised_dataset(
            [episode], instructions, tmp_path / "artifacts" / "empty"
        )
