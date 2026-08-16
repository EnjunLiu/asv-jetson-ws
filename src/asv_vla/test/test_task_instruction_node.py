"""ROS-independent tests for the latched task instruction source."""

from __future__ import annotations

import ast
from pathlib import Path


NODE = (
    Path(__file__).resolve().parents[1]
    / "asv_vla"
    / "task_instruction_node.py"
)
LAUNCH = Path(__file__).resolve().parents[3] / "src/asv_bringup/launch/vla_closed_loop.launch.py"


def _load_validator():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_task_text"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NODE), "exec"), namespace)
    return namespace["_validate_task_text"]


def test_nonempty_instruction_has_publishable_payload():
    validate = _load_validator()
    assert validate("  跟随红色目标船，保持3米距离  ") == "跟随红色目标船，保持3米距离"


def test_empty_instruction_is_rejected_without_ros():
    validate = _load_validator()
    for value in ("", "   ", "\n\t"):
        try:
            validate(value)
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:
            raise AssertionError("empty task text was accepted")


def test_launch_wires_task_instruction_source():
    source = LAUNCH.read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument(\n                "task_text"' in source
    assert "跟随红色目标船，保持3米距离" in source
    assert 'executable="task_instruction"' in source
    assert 'LaunchConfiguration("task_text")' in source
    assert '"/task/text"' in NODE.read_text(encoding="utf-8")
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in NODE.read_text(encoding="utf-8")
