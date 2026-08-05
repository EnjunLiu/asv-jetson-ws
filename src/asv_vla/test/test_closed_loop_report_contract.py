"""Static consistency checks for the committed closed-loop evidence report."""

import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
REPORT = (
    REPOSITORY
    / "pc_datasets"
    / "reports"
    / "closed_loop_20260805"
    / "single_point_policy_dominant_v8"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "scene, directory", [("RED 4m", "red4m"), ("BLUE 3m", "blue3m")]
)
def test_scene_metrics_are_self_contained_and_match_combined(
    scene: str, directory: str
) -> None:
    scene_metrics = _load(REPORT / directory / "metrics.json")
    combined_metrics = _load(REPORT / "combined_metrics.json")
    assert scene_metrics[scene] == combined_metrics[scene]

    values = scene_metrics[scene]
    audit = values["audit"]
    assert audit["events_complete"] is True
    classified = sum(
        audit[name]
        for name in ("policy_driven", "backstop", "hold", "fail_closed")
    )
    assert audit["events"] == classified
    assert audit["policy_dominance_rate"] == pytest.approx(
        audit["policy_driven"] / classified
    )
    assert audit["backstop_rate"] == pytest.approx(audit["backstop"] / classified)

    for raw_path in [values["ue_log"], *values["jetson_logs"]]:
        path = Path(raw_path)
        assert not path.is_absolute()
        assert (REPORT / path).is_file()


def test_report_readme_names_all_audit_categories() -> None:
    readme = (REPORT / "README.md").read_text(encoding="utf-8")
    assert "red4m/metrics.json" in readme
    assert "blue3m/metrics.json" in readme
    assert "combined_metrics.json" in readme
    assert "hold" in readme
    assert "policy-driven + backstop + hold +" in readme
    assert "fail-closed = apply count" in readme


def test_checkpoint_manifest_covers_the_delivered_artifact() -> None:
    manifest = _load(REPORT / "checkpoint_manifest.json")
    assert manifest["checkpoint"]["sha256"] == (
        "f2dc38a141a3f230b2ddf55cef26841f00812bbd350f28aa84c84f5d5d1e2483"
    )
    assert manifest["checkpoint_payload"]["schema_version"] == (
        "synthetic_geometry_single_point_v1"
    )
    assert manifest["checkpoint_payload"]["metadata_contains_contract"] is False
    assert manifest["model"]["contract"]["outputs"]["action"] == {
        "shape": ["B", 2],
        "dtype": "float32",
        "frame": "base_link",
        "kind": "single_step_desired_displacement_m",
        "maximum_norm_m": 0.3,
    }
    training = manifest["training"]
    assert training["sample_count"] == 16384
    assert training["seed"] == 23
    assert training["dataset_seed"] == 23
    assert training["epochs"] == 250
    assert training["batch_size"] == 1024
    assert training["learning_rate"] == 0.002
    assert training["device"] == "cuda"
    assert training["language_embedding_sha256"] == (
        "c144affbb0b18ab61cd135179b54e3564a91b6c0fc97c5baa965037664ed5958"
    )
