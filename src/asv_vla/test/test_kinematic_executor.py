from pathlib import Path
from types import SimpleNamespace

import pytest

from asv_vla.kinematic_executor import first_step_from_expert


def action_message(
    *,
    valid=True,
    safe_stop=False,
    frame_id="base_link",
    run_id="run-1",
    dt=0.2,
    desired_x=0.3,
    desired_y=0.0,
    detail="ok",
):
    return SimpleNamespace(
        valid=valid,
        safe_stop=safe_stop,
        frame_id=frame_id,
        run_id=run_id,
        model_version="expert-v1",
        dt=dt,
        desired_x=desired_x,
        desired_y=desired_y,
        detail=detail,
    )


def test_consumes_one_action_without_horizon_or_trajectory_shape():
    step = first_step_from_expert(action_message())

    assert step.valid
    assert not step.hold_position
    assert (step.delta_x_m, step.delta_y_m) == pytest.approx((0.3, 0.0))
    assert step.step_dt == pytest.approx(0.2)


def test_stop_becomes_valid_hold():
    step = first_step_from_expert(
        action_message(
            safe_stop=True,
            desired_x=0.0,
            desired_y=0.0,
        )
    )

    assert step.valid
    assert step.hold_position
    assert (step.delta_x_m, step.delta_y_m) == (0.0, 0.0)
    assert step.reason.startswith("SAFE_STOP:")


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (action_message(valid=False), "INVALID_EXPERT"),
        (action_message(frame_id="map"), "INVALID_FRAME"),
        (action_message(run_id=""), "INVALID_RUN_ID"),
        (action_message(desired_x=float("nan")), "NONFINITE_ACTION"),
        (action_message(desired_x=0.36), "STEP_LIMIT"),
        (
            action_message(
                safe_stop=True,
                desired_x=0.1,
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


def test_legacy_trajectory_fields_and_shapes_are_rejected():
    legacy = SimpleNamespace(
        valid=True,
        safe_stop=False,
        frame_id="base_link",
        run_id="run-1",
        model_version="expert-v1",
        dt=0.2,
        horizon=2,
        delta_p_xy=(0.3, 0.0, 0.6, 0.0),
        detail="legacy",
    )

    step = first_step_from_expert(legacy)

    assert not step.valid
    assert step.hold_position
    assert "INVALID_ACTION_FIELDS" in step.reason


def test_missing_safety_field_fails_closed():
    malformed = action_message()
    del malformed.safe_stop

    step = first_step_from_expert(malformed)

    assert not step.valid
    assert step.hold_position
    assert "missing safe_stop" in step.reason


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
