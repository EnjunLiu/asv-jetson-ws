from __future__ import annotations

import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from asv_interfaces.msg import ASVWrench, ControlInput
from asv_jetson_interfaces.msg import (
    DecisionOutput,
    SelectedTrajectory,
    ThrusterCommand,
    TrajectoryCandidates,
)


NUM_MODES = 6
HORIZON = 20
ACTION_DIM = 2
STOP_INDEX = 5
TIMEOUT_SEC = 12.0


def zero(value: float) -> bool:
    return math.isfinite(value) and abs(value) <= 1.0e-9


class ContractProbe(Node):
    def __init__(self) -> None:
        super().__init__("day1_contract_probe")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.started = time.monotonic()
        self.done = False
        self.passed = False
        self.messages = {}

        self.create_subscription(
            TrajectoryCandidates,
            "/vla/candidate_trajectories",
            lambda msg: self.record("candidates", msg),
            qos,
        )
        self.create_subscription(
            SelectedTrajectory,
            "/vla/selected_trajectory",
            lambda msg: self.record("selected", msg),
            qos,
        )
        self.create_subscription(
            DecisionOutput,
            "/decision/output",
            lambda msg: self.record("decision", msg),
            qos,
        )
        self.create_subscription(
            ControlInput,
            "/control/control_input",
            lambda msg: self.record("control_input", msg),
            qos,
        )
        self.create_subscription(
            ASVWrench,
            "/control/asv_wrench",
            lambda msg: self.record("esp32_wrench", msg),
            qos,
        )
        self.create_subscription(
            ASVWrench,
            "/control/safe_wrench",
            lambda msg: self.record("safe_wrench", msg),
            qos,
        )
        self.create_subscription(
            ThrusterCommand,
            "/ue/thruster_command",
            lambda msg: self.record("thruster", msg),
            qos,
        )
        self.create_timer(0.1, self.evaluate)

    def record(self, name, message) -> None:
        self.messages[name] = message

    def evaluate(self) -> None:
        required = {
            "candidates",
            "selected",
            "decision",
            "control_input",
            "esp32_wrench",
            "safe_wrench",
            "thruster",
        }
        if required.issubset(self.messages):
            self.passed = self.check_contract()
            self.done = True
            marker = "DAY1_CONTRACT_PASS" if self.passed else "DAY1_CONTRACT_FAIL"
            self.get_logger().info(marker)
            return

        if time.monotonic() - self.started > TIMEOUT_SEC:
            missing = sorted(required.difference(self.messages))
            self.get_logger().error(f"DAY1_CONTRACT_FAIL missing={missing}")
            self.done = True
            self.passed = False

    def check_contract(self) -> bool:
        candidates = self.messages["candidates"]
        selected = self.messages["selected"]
        decision = self.messages["decision"]
        control_input = self.messages["control_input"]
        esp32_wrench = self.messages["esp32_wrench"]
        safe_wrench = self.messages["safe_wrench"]
        thruster = self.messages["thruster"]

        checks = {
            "candidate_modes": candidates.num_modes == NUM_MODES,
            "candidate_horizon": candidates.horizon == HORIZON,
            "candidate_shape":
                len(candidates.xy) == NUM_MODES * HORIZON * ACTION_DIM,
            "candidate_zero": all(zero(value) for value in candidates.xy),
            "candidate_invalid": not candidates.valid,
            "selected_index": selected.selected_index == STOP_INDEX,
            "selected_horizon": selected.horizon == HORIZON,
            "selected_shape": len(selected.xy) == HORIZON * ACTION_DIM,
            "selected_zero": all(zero(value) for value in selected.xy),
            "selected_safe_stop": selected.safe_stop,
            "selected_container_valid": selected.valid,
            "decision_x_zero": zero(decision.desired_x),
            "decision_y_zero": zero(decision.desired_y),
            "decision_invalid": not decision.valid,
            "control_x_zero": zero(control_input.desired_x),
            "control_y_zero": zero(control_input.desired_y),
            "control_invalid": not control_input.valid,
            "esp32_force_zero": zero(esp32_wrench.force),
            "esp32_moment_zero": zero(esp32_wrench.moment),
            "esp32_invalid": not esp32_wrench.valid,
            "safe_force_zero": zero(safe_wrench.force),
            "safe_moment_zero": zero(safe_wrench.moment),
            "safe_invalid": not safe_wrench.valid,
            "left_thruster_zero": zero(thruster.left_thruster),
            "right_thruster_zero": zero(thruster.right_thruster),
            "thruster_invalid": not thruster.valid,
        }

        for name, passed in checks.items():
            log = self.get_logger().info if passed else self.get_logger().error
            log(f"{name}={'PASS' if passed else 'FAIL'}")
        return all(checks.values())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ContractProbe()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        exit_code = 0 if node.passed else 1
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv)
