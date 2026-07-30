"""Day 17 safety gate ROS probe.

Publishes test trajectories to ``/vla/policy_trajectory`` and verifies
that ``/vla/selected_trajectory`` respects the safety gate contract.
"""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.node import Node
from asv_jetson_interfaces.msg import SelectedTrajectory

from .safety_gate import (
    COLLISION_RISK,
    CURVATURE_LIMIT,
    ESTOP,
    INVALID_MODALITY,
    INVALID_SHAPE,
    NONFINITE,
    PASS,
    POLICY_STOP,
    SPEED_LIMIT,
    STALE_INPUT,
)
from .trajectory_contract import ACTION_DIM, DT_SEC, FRAME_ID, HORIZON

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))


def _step_trajectory(dx: float, dy: float) -> list[float]:
    result: list[float] = []
    for i in range(HORIZON):
        result.extend(((i + 1) * dx, (i + 1) * dy))
    return result


def _zero_trajectory() -> list[float]:
    return [0.0] * (HORIZON * ACTION_DIM)


class SafetyGateProbe(Node):
    def __init__(self) -> None:
        super().__init__("safety_gate_probe")
        self._received: list[SelectedTrajectory] = []

        self._pub = self.create_publisher(
            SelectedTrajectory, "/vla/policy_trajectory", 10
        )
        self._sub = self.create_subscription(
            SelectedTrajectory,
            "/vla/selected_trajectory",
            self._on_selected,
            10,
        )

    def _on_selected(self, message: SelectedTrajectory) -> None:
        self._received.append(message)

    def publish_and_wait(
        self,
        delta_p_xy: list[float],
        **overrides,
    ) -> SelectedTrajectory | None:
        before = len(self._received)
        msg = SelectedTrajectory()
        msg.stamp_us = int(overrides.get("stamp_us", int(time.time() * 1e6)))
        msg.run_id = str(overrides.get("run_id", "probe"))
        msg.frame_id = str(overrides.get("frame_id", FRAME_ID))
        msg.model_version = str(overrides.get("model_version", "test_policy"))
        msg.dt = float(overrides.get("dt", DT_SEC))
        msg.horizon = int(overrides.get("horizon", HORIZON))
        msg.delta_p_xy = list(delta_p_xy)
        msg.safe_stop = bool(overrides.get("safe_stop", False))
        msg.valid = bool(overrides.get("valid", True))
        msg.reason = str(overrides.get("reason", "test"))

        self._pub.publish(msg)
        deadline = time.monotonic() + 2.0
        while len(self._received) <= before and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if len(self._received) > before:
            return self._received[-1]
        return None

    def verify(
        self,
        name: str,
        delta_p_xy: list[float],
        expect_pass: bool,
        expect_reason: str | None = None,
        **overrides,
    ) -> None:
        response = self.publish_and_wait(delta_p_xy, **overrides)
        if response is None:
            _record(name, False, "no response")
            return
        if expect_pass:
            ok = bool(response.valid) and not bool(response.safe_stop)
            _record(name, ok, f"valid={response.valid} reason={response.reason}")
        elif expect_reason:
            ok = expect_reason in str(response.reason)
            _record(name, ok, f"expected={expect_reason} got={response.reason}")
        else:
            ok = not bool(response.valid)
            _record(name, ok, f"valid={response.valid}")


def main(args=None) -> None:
    rclpy.init(args=args)
    probe = SafetyGateProbe()

    # Give the safety gate node time to start.
    time.sleep(1.0)

    # 1. Normal FOLLOW trajectory passes.
    probe.verify("normal_follow", _step_trajectory(0.1, 0.0), expect_pass=True)

    # 2. Policy STOP produces safe_stop.
    response = probe.publish_and_wait(
        _zero_trajectory(), safe_stop=True, valid=True
    )
    if response:
        ok = response.safe_stop and response.valid
        _record("policy_stop", ok, f"safe_stop={response.safe_stop}")

    # 3. Invalid modality rejected.
    probe.verify(
        "invalid_frame",
        _zero_trajectory(),
        expect_pass=False,
        expect_reason=INVALID_MODALITY,
        frame_id="wrong",
    )

    # 4. NaN rejected.
    values = _zero_trajectory()
    values[5] = float("nan")
    probe.verify(
        "nan_trajectory",
        values,
        expect_pass=False,
        expect_reason=NONFINITE,
    )

    # 5. Speed limit rejected.
    probe.verify(
        "speed_limit",
        _step_trajectory(0.5, 0.0),
        expect_pass=False,
        expect_reason=SPEED_LIMIT,
    )

    # 6. Zero trajectory passes.
    probe.verify("zero_trajectory", _zero_trajectory(), expect_pass=True)

    # 7. Invalid policy rejected.
    probe.verify(
        "invalid_policy",
        _zero_trajectory(),
        expect_pass=False,
        expect_reason=INVALID_MODALITY,
        valid=False,
    )

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    all_ok = passed == total

    print(f"DAY17_SAFETY_GATE_PROBE {'PASS' if all_ok else 'FAIL'} {passed}/{total}")
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")

    probe.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
