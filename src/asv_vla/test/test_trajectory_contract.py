from pathlib import Path
from types import SimpleNamespace

from asv_vla.trajectory_contract import (
    ACTION_DIM,
    DT_SEC,
    FRAME_ID,
    HORIZON,
    SAFE_STOP_MODEL_VERSION,
    is_day1_safe_stop,
)


def message(**overrides):
    values = {
        "stamp_us": 1,
        "run_id": "day1-test",
        "frame_id": FRAME_ID,
        "model_version": SAFE_STOP_MODEL_VERSION,
        "dt": DT_SEC,
        "horizon": HORIZON,
        "delta_p_xy": [0.0] * (HORIZON * ACTION_DIM),
        "safe_stop": True,
        "valid": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_day1_safe_stop_contract_accepts_well_formed_message():
    assert is_day1_safe_stop(message())


def test_day1_safe_stop_contract_rejects_wrong_shape():
    assert not is_day1_safe_stop(message(delta_p_xy=[0.0, 0.0]))


def test_day1_safe_stop_contract_rejects_nonfinite_or_nonzero_actions():
    assert not is_day1_safe_stop(
        message(delta_p_xy=[float("nan")] + [0.0] * 39)
    )
    assert not is_day1_safe_stop(message(delta_p_xy=[0.01] + [0.0] * 39))


def test_day1_safe_stop_contract_rejects_executable_or_invalid_container():
    assert not is_day1_safe_stop(message(safe_stop=False))
    assert not is_day1_safe_stop(message(valid=False))


def test_day1_safe_stop_contract_rejects_wrong_frame_or_timing():
    assert not is_day1_safe_stop(message(stamp_us=0))
    assert not is_day1_safe_stop(message(run_id=""))
    assert not is_day1_safe_stop(message(frame_id="map"))
    assert not is_day1_safe_stop(message(dt=0.1))


def test_obsolete_candidate_and_world_model_interfaces_are_removed():
    repository = Path(__file__).resolve().parents[3]
    interface_dir = repository / "src/asv_jetson_interfaces/msg"
    cmake = (
        repository / "src/asv_jetson_interfaces/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert not (interface_dir / "TrajectoryCandidates.msg").exists()
    assert not (interface_dir / "WorldModelEvaluation.msg").exists()
    assert '"msg/TrajectoryCandidates.msg"' not in cmake
    assert '"msg/WorldModelEvaluation.msg"' not in cmake


def test_selected_trajectory_message_matches_direct_policy_contract():
    repository = Path(__file__).resolve().parents[3]
    fields = [
        line.strip()
        for line in (
            repository / "src/asv_jetson_interfaces/msg/SelectedTrajectory.msg"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert fields == [
        "int64 stamp_us",
        "string run_id",
        "string frame_id",
        "string model_version",
        "float32 dt",
        "uint16 horizon",
        "float32[] delta_p_xy",
        "bool safe_stop",
        "bool valid",
        "string reason",
    ]
