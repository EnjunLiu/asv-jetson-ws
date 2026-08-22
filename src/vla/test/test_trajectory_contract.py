from pathlib import Path
from types import SimpleNamespace

from vla.decision import (
    DT_SEC,
    FRAME_ID,
    MAX_DISPLACEMENT_M,
    SAFE_STOP_SOURCE,
    is_safe_stop,
)


def message(**overrides):
    values = {
        "stamp_us": 1,
        "run_id": "test",
        "scene_seed": 1,
        "frame_index": 0,
        "frame_id": FRAME_ID,
        "source": SAFE_STOP_SOURCE,
        "step_dt": DT_SEC,
        "desired_x": 0.0,
        "desired_y": 0.0,
        "safe_stop": True,
        "valid": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_safe_stop_contract_accepts_invalid_zero_point():
    assert is_safe_stop(message())


def test_single_point_contract_matches_trained_action_limit():
    assert DT_SEC == 0.5
    assert MAX_DISPLACEMENT_M == 0.50


def test_safe_stop_contract_rejects_executable_zero_point():
    assert not is_safe_stop(message(valid=True))


def test_safe_stop_contract_rejects_nonfinite_or_nonzero_action():
    assert not is_safe_stop(message(desired_x=float("nan")))
    assert not is_safe_stop(message(desired_y=0.01))


def test_safe_stop_contract_rejects_wrong_frame_or_timing():
    assert not is_safe_stop(message(stamp_us=0))
    assert not is_safe_stop(message(run_id=""))
    assert not is_safe_stop(message(frame_id="map"))
    assert not is_safe_stop(message(step_dt=0.1))


def test_online_displacement_message_has_scalar_fields():
    repository = Path(__file__).resolve().parents[3]
    interface_dir = repository / "src/interfaces/msg"
    fields = [
        line.strip()
        for line in (interface_dir / "DesiredDisplacement.msg")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "float32 desired_x" in fields
    assert "float32 desired_y" in fields
    assert not any("delta_p_xy" in field for field in fields)
    cmake = (interface_dir.parent / "CMakeLists.txt").read_text(encoding="utf-8")
    assert '"msg/DesiredDisplacement.msg"' in cmake
    assert not (interface_dir / "DecisionPoint.msg").exists()
    assert not (interface_dir / "DecisionOutput.msg").exists()


def test_obsolete_candidate_and_world_model_interfaces_are_removed():
    repository = Path(__file__).resolve().parents[3]
    interface_dir = repository / "src/interfaces/msg"
    cmake = (repository / "src/interfaces/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    assert not (interface_dir / "TrajectoryCandidates.msg").exists()
    assert not (interface_dir / "WorldModelEvaluation.msg").exists()
    assert '"msg/TrajectoryCandidates.msg"' not in cmake
    assert '"msg/WorldModelEvaluation.msg"' not in cmake
