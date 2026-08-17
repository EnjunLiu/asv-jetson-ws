"""ROS-independent text guards for entity message provenance fields."""

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
MESSAGE_DIR = REPOSITORY / "src/interfaces/msg"


def _fields(name: str) -> list[str]:
    return [
        line.strip()
        for line in (MESSAGE_DIR / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_entity_has_optional_provenance_and_observability_fields() -> None:
    fields = _fields("Entity.msg")
    assert fields[-8:] == [
        "string source",
        "float32 bbox_x_min",
        "float32 bbox_y_min",
        "float32 bbox_x_max",
        "float32 bbox_y_max",
        "bool bbox_valid",
        "float32 confidence",
        "bool velocity_valid",
    ]
    source = (MESSAGE_DIR / "Entity.msg").read_text(encoding="utf-8")
    for value in ("ue_truth", "image_perception", "temporal_tracker"):
        assert value in source


def test_entity_array_and_entity_features_retain_optional_instruction_metadata() -> None:
    array_fields = _fields("EntityArray.msg")
    assert array_fields[-3:] == [
        "string source",
        "string instruction_id",
        "string instruction",
    ]

    feature_fields = _fields("EntityFeatures.msg")
    assert feature_fields[-2:] == [
        "string instruction_id",
        "string instruction",
    ]


def test_entity_interfaces_remain_registered() -> None:
    cmake = (MESSAGE_DIR.parent / "CMakeLists.txt").read_text(encoding="utf-8")
    assert '"msg/Entity.msg"' in cmake
    assert '"msg/EntityArray.msg"' in cmake
    assert not (MESSAGE_DIR / "UEEntity.msg").exists()
    assert not (MESSAGE_DIR / "UEEntityArray.msg").exists()
    assert '"msg/EntityFeatures.msg"' in cmake
    assert not (MESSAGE_DIR / "TaskFeatures.msg").exists()
