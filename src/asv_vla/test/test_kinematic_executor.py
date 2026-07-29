from pathlib import Path
from types import SimpleNamespace

import pytest

from asv_vla.kinematic_executor import first_step_from_expert


def trajectory(
    *,
    valid=True,
    safe_stop=False,
    frame_id="base_link",
    run_id="run-1",
    dt=0.2,
    horizon=2,
    values=(0.3, 0.0, 0.6, 0.0),
    detail="ok",
):
    return SimpleNamespace(
        valid=valid,
        safe_stop=safe_stop,
        frame_id=frame_id,
        run_id=run_id,
        model_version="expert-v1",
        dt=dt,
        horizon=horizon,
        delta_p_xy=values,
        detail=detail,
    )


def test_extracts_only_first_cumulative_waypoint():
    step = first_step_from_expert(trajectory())

    assert step.valid
    assert not step.hold_position
    assert (step.delta_x_m, step.delta_y_m) == pytest.approx((0.3, 0.0))
    assert step.step_dt == pytest.approx(0.2)


def test_stop_becomes_valid_hold():
    step = first_step_from_expert(
        trajectory(
            safe_stop=True,
            values=(0.0, 0.0, 0.0, 0.0),
        )
    )

    assert step.valid
    assert step.hold_position
    assert (step.delta_x_m, step.delta_y_m) == (0.0, 0.0)
    assert step.reason.startswith("SAFE_STOP:")


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (trajectory(valid=False), "INVALID_EXPERT"),
        (trajectory(frame_id="map"), "INVALID_FRAME"),
        (trajectory(run_id=""), "INVALID_RUN_ID"),
        (trajectory(values=(0.1, 0.0)), "INVALID_SHAPE"),
        (
            trajectory(values=(float("nan"), 0.0, 0.0, 0.0)),
            "NONFINITE_TRAJECTORY",
        ),
        (trajectory(values=(0.36, 0.0, 0.6, 0.0)), "STEP_LIMIT"),
        (
            trajectory(
                safe_stop=True,
                values=(0.1, 0.0, 0.0, 0.0),
            ),
            "INVALID_SAFE_STOP",
        ),
    ],
)
def test_invalid_inputs_fail_to_non_executable_hold(source, reason):
    step = first_step_from_expert(source)

    assert not step.valid
    assert step.hold_position
    assert (step.delta_x_m, step.delta_y_m) == (0.0, 0.0)
    assert reason in step.reason


def test_message_and_bridge_contract_are_registered():
    repository = Path(__file__).resolve().parents[3]
    message = (
        repository
        / "src/asv_jetson_interfaces/msg/UEKinematicSetpoint.msg"
    ).read_text(encoding="utf-8")
    interfaces_cmake = (
        repository / "src/asv_jetson_interfaces/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    bridge = (
        repository
        / "src/asv_ue_bridge/src/ue_object_deliverer_bridge_node.cpp"
    ).read_text(encoding="utf-8")

    assert "float32 delta_x_m" in message
    assert "float32 delta_y_m" in message
    assert "bool hold_position" in message
    assert '"msg/UEKinematicSetpoint.msg"' in interfaces_cmake
    assert '"Command_Type", "Kinematic_Setpoint"' in bridge
    assert '"Delta_X_Cm"' in bridge
    assert '"Delta_Y_Cm"' in bridge
