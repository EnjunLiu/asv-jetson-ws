"""Pure image-only standoff guard tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from asv_vla.visual_standoff_guard import (
    apply_standoff_guard,
    compute_standoff_step,
    desired_standoff_from_instruction,
    extract_target_observation,
)


def _task(
    instruction: str,
    *,
    x: float = 5.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    visible: bool = True,
    target_id: str = "target_red",
):
    rows = np.zeros((16, 16), dtype=np.float32)
    rows[0, :5] = (x / 20.0, y / 20.0, 0.0, vx / 5.0, vy / 5.0)
    return SimpleNamespace(
        valid=True,
        instruction=instruction,
        max_entities=16,
        feature_dim=16,
        entity_ids=[target_id] + [""] * 15,
        features=rows.reshape(-1).tolist(),
        mask=[visible] + [False] * 15,
    )


def test_red_left_geometry_produces_positive_left_step() -> None:
    task = _task("跟随红色目标船，保持3米距离", x=5.0, y=4.0)
    observation = extract_target_observation(task)
    assert observation is not None
    step = compute_standoff_step(observation, 3.0)
    assert step is not None
    assert step[0] > 0.0
    assert step[1] > 0.0
    assert np.linalg.norm(step) <= 0.15 + 1.0e-9


def test_three_meter_deadband_and_ten_meter_parser() -> None:
    near = _task("follow red boat keep 3m", x=3.1)
    near_observation = extract_target_observation(near)
    assert near_observation is not None
    assert compute_standoff_step(near_observation, 3.0) == (0.0, 0.0)

    far = _task("follow red boat keep 10m", x=12.0)
    far_observation = extract_target_observation(far)
    assert far_observation is not None
    assert desired_standoff_from_instruction(far.instruction) == 10.0
    assert compute_standoff_step(
        far_observation,
        desired_standoff_from_instruction(far.instruction),
    ) == (0.15, 0.0)


def test_velocity_prediction_changes_bounded_step() -> None:
    task = _task("follow red boat 3m", x=3.0, vx=2.0)
    observation = extract_target_observation(task)
    assert observation is not None
    step = compute_standoff_step(observation, 3.0)
    assert step == (0.15, 0.0)


def test_missing_or_nonfinite_target_is_fail_closed() -> None:
    missing = _task("follow red boat 3m", visible=False)
    assert extract_target_observation(missing) is None
    raw = np.tile(np.asarray((0.2, 0.0), dtype=np.float32), 20)
    guarded, applied = apply_standoff_guard(raw, missing)
    assert guarded is None
    assert applied is False

    nonfinite = _task("follow red boat 3m")
    nonfinite.features[0] = float("nan")
    assert extract_target_observation(nonfinite) is None


def test_policy_guard_replaces_only_first_waypoint_for_follow() -> None:
    task = _task("follow red boat keep 3m", x=5.0, y=4.0)
    raw = np.tile(np.asarray((0.2, 0.0), dtype=np.float32), 20)
    guarded, applied = apply_standoff_guard(raw, task)
    assert applied is True
    assert guarded is not None
    assert np.allclose(np.asarray(guarded).reshape(20, 2)[1:], raw.reshape(20, 2)[1:])
    assert np.allclose(np.asarray(guarded).reshape(20, 2)[0], (0.11713032, 0.09370426))

    non_follow = _task("stop now", x=5.0, y=4.0)
    unchanged, applied = apply_standoff_guard(raw, non_follow)
    assert applied is False
    assert unchanged == tuple(float(value) for value in raw)
