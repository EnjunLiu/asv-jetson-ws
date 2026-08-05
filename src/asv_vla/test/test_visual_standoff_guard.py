"""Pure tests for the image/tracker standoff backstop."""

from types import SimpleNamespace

import numpy as np

from asv_vla.visual_standoff_guard import (
    GUARD_BACKSTOP,
    GUARD_FAIL_CLOSED,
    GUARD_PASS_THROUGH,
    GUARD_POLICY_DRIVEN,
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


def test_lateral_geometry_produces_bounded_body_step():
    task = _task("follow left boat keep 3m", x=5.0, y=4.0, target_id="target_left")
    observation = extract_target_observation(task)
    assert observation is not None
    step = compute_standoff_step(observation, 3.0)
    assert step is not None
    assert step[0] > 0.0 and step[1] > 0.0
    assert np.linalg.norm(step) <= 0.15 + 1.0e-9


def test_distance_parser_and_deadband():
    near = _task("follow red boat keep 3m", x=3.1)
    observation = extract_target_observation(near)
    assert observation is not None
    assert compute_standoff_step(observation, 3.0) == (0.0, 0.0)
    far = _task("follow red boat keep 10m", x=12.0)
    assert desired_standoff_from_instruction(far.instruction) == 10.0


def test_velocity_prediction_changes_bounded_step():
    moving = _task("follow red boat 3m", x=3.0, vx=4.0)
    observation = extract_target_observation(moving)
    assert observation is not None
    assert compute_standoff_step(observation, 3.0) == (0.15, 0.0)


def test_missing_or_nonfinite_target_fails_closed():
    missing = _task("follow red boat 3m", visible=False)
    assert extract_target_observation(missing) is None
    assert apply_standoff_guard((0.1, 0.0), missing) == (None, GUARD_FAIL_CLOSED)
    nonfinite = _task("follow red boat 3m")
    nonfinite.features[0] = float("nan")
    assert apply_standoff_guard((0.1, 0.0), nonfinite) == (
        None,
        GUARD_FAIL_CLOSED,
    )


def test_full_model_trajectory_is_not_an_online_guard_contract():
    task = _task("follow red boat keep 3m")
    guarded, reason = apply_standoff_guard(np.zeros(40), task)
    assert guarded is None
    assert reason == GUARD_FAIL_CLOSED


def test_policy_step_toward_target_is_kept_as_one_point():
    task = _task("follow red boat keep 3m", x=5.0, y=4.0)
    observation = extract_target_observation(task)
    assert observation is not None
    displacement = compute_standoff_step(observation, 3.0)
    assert displacement is not None
    guarded, reason = apply_standoff_guard(displacement, task)
    assert reason == GUARD_POLICY_DRIVEN
    assert guarded == displacement


def test_away_or_frozen_policy_step_uses_radial_backstop():
    task = _task("follow red boat keep 3m", x=5.0, y=4.0)
    guarded, reason = apply_standoff_guard((-0.1, 0.0), task)
    assert reason == GUARD_BACKSTOP
    assert guarded is not None
    assert np.allclose(guarded, (0.11713032, 0.09370426))
    frozen, frozen_reason = apply_standoff_guard((0.0, 0.0), task)
    assert frozen_reason == GUARD_BACKSTOP
    assert frozen is not None and np.linalg.norm(frozen) > 0.0


def test_lateral_direction_is_checked_from_target_velocity():
    target = _task("follow red boat keep 3m", x=5.0, y=-0.6, vy=4.0)
    observation = extract_target_observation(target)
    assert observation is not None
    displacement = compute_standoff_step(observation, 3.0)
    assert displacement is not None
    guarded, reason = apply_standoff_guard(displacement, target)
    assert reason == GUARD_POLICY_DRIVEN
    assert guarded == displacement


def test_hold_at_standoff_is_valid_zero_point():
    task = _task("follow red boat keep 3m", x=3.05, y=0.0)
    guarded, reason = apply_standoff_guard((0.0, 0.0), task)
    assert reason == GUARD_POLICY_DRIVEN
    assert guarded == (0.0, 0.0)


def test_non_follow_passes_through_one_point():
    guarded, reason = apply_standoff_guard((0.1, 0.0), _task("stop now"))
    assert reason == GUARD_PASS_THROUGH
    assert guarded == (0.1, 0.0)
