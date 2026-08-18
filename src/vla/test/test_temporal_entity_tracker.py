from __future__ import annotations

import pytest
from pathlib import Path

from vla.perception import (
    FrameMetadata,
    GeometryObservation,
    TemporalEntityTracker,
    TemporalEntityTrackerError,
)


def observation(
    entity_id: str = "target",
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    run_id: str = "run-a",
    scene_seed: int = 7,
    frame_index: int = 0,
    stamp_us: int = 0,
    **kwargs,
) -> GeometryObservation:
    return GeometryObservation(
        entity_id,
        x,
        y,
        z,
        run_id=run_id,
        scene_seed=scene_seed,
        frame_index=frame_index,
        stamp_us=stamp_us,
        **kwargs,
    )


def test_first_observation_does_not_guess_velocity():
    record = TemporalEntityTracker().update([observation()])[0]

    assert not record.velocity_valid
    assert record.velocity == (0.0, 0.0, 0.0)


def test_adjacent_frames_use_explicit_finite_difference():
    tracker = TemporalEntityTracker()
    tracker.update([observation(stamp_us=0)])
    record = tracker.update([
        observation(x=2.0, y=-1.0, z=0.5, frame_index=1, stamp_us=1_000_000)
    ])[0]

    assert record.velocity_valid
    assert record.velocity == pytest.approx((2.0, -1.0, 0.5))
    assert record.frame_gap == 1


def test_run_or_scene_switch_hard_resets_history():
    tracker = TemporalEntityTracker()
    tracker.update([observation(x=1.0, stamp_us=0)])
    switched = tracker.update([
        observation(
            x=5.0,
            run_id="run-b",
            scene_seed=8,
            frame_index=0,
            stamp_us=0,
        )
    ])[0]

    assert tracker.identity == ("run-b", 8)
    assert not switched.velocity_valid


def test_dropped_frame_uses_elapsed_time_and_reports_gap():
    tracker = TemporalEntityTracker(ttl_frames=3)
    tracker.update([observation(stamp_us=0)])
    record = tracker.update([
        observation(x=0.8, frame_index=3, stamp_us=400_000)
    ])[0]

    assert record.velocity_valid
    assert record.velocity[0] == pytest.approx(2.0)
    assert record.frame_gap == 3


def test_reappearance_within_ttl_keeps_history():
    tracker = TemporalEntityTracker(ttl_frames=2)
    tracker.update([observation(stamp_us=0)])
    tracker.update([], frame=FrameMetadata("run-a", 7, 1, 500_000))
    record = tracker.update([
        observation(x=1.0, frame_index=2, stamp_us=1_000_000)
    ])[0]

    assert record.velocity_valid
    assert record.velocity[0] == pytest.approx(1.0)


def test_reappearance_after_ttl_is_a_new_track():
    tracker = TemporalEntityTracker(ttl_frames=1)
    tracker.update([observation(stamp_us=0)])
    tracker.update([], frame=FrameMetadata("run-a", 7, 1, 500_000))
    record = tracker.update([
        observation(x=1.0, frame_index=2, stamp_us=1_000_000)
    ])[0]

    assert not record.velocity_valid
    assert record.velocity == (0.0, 0.0, 0.0)


def test_nonmonotonic_frame_is_ignored_without_state_corruption():
    tracker = TemporalEntityTracker()
    tracker.update([observation(x=0.0, frame_index=2, stamp_us=200_000)])
    assert tracker.update([
        observation(x=10.0, frame_index=1, stamp_us=100_000)
    ]) == ()
    record = tracker.update([
        observation(x=1.0, frame_index=3, stamp_us=300_000)
    ])[0]

    assert record.velocity_valid
    assert record.velocity[0] == pytest.approx(10.0)


def test_nonmonotonic_stamp_invalidates_velocity_for_that_frame():
    tracker = TemporalEntityTracker()
    tracker.update([observation(stamp_us=200_000)])
    record = tracker.update([
        observation(x=1.0, frame_index=1, stamp_us=100_000)
    ])[0]

    assert not record.velocity_valid
    assert record.velocity == (0.0, 0.0, 0.0)


def test_ema_filter_smooths_after_a_valid_previous_velocity():
    tracker = TemporalEntityTracker(velocity_filter="ema", alpha=0.5)
    tracker.update([observation(stamp_us=0)])
    tracker.update([observation(x=1.0, frame_index=1, stamp_us=1_000_000)])
    record = tracker.update([
        observation(x=3.0, frame_index=2, stamp_us=2_000_000)
    ])[0]

    assert record.velocity_valid
    assert record.velocity[0] == pytest.approx(1.5)


def test_alpha_beta_filter_uses_position_residual():
    tracker = TemporalEntityTracker(
        velocity_filter="alpha_beta", alpha=0.5, beta=0.5
    )
    tracker.update([observation(stamp_us=0)])
    tracker.update([observation(x=1.0, frame_index=1, stamp_us=1_000_000)])
    record = tracker.update([
        observation(x=3.0, frame_index=2, stamp_us=2_000_000)
    ])[0]

    assert record.velocity_valid
    assert record.velocity[0] == pytest.approx(1.5)


def test_semantics_bbox_confidence_and_ue_adapter_are_preserved():
    record = TemporalEntityTracker().update([
        observation(
            class_name="boat",
            color="red",
            is_target=True,
            bbox=(1.0, 2.0, 10.0, 20.0),
            confidence=0.75,
        )
    ])[0]

    assert record.bbox == (1.0, 2.0, 10.0, 20.0)
    assert record.confidence == pytest.approx(0.75)
    assert record.source == "temporal_tracker"
    kwargs = record.as_entity_kwargs()
    assert kwargs["relative_velocity_x"] == 0.0
    assert kwargs["source"] == "temporal_tracker"
    assert kwargs["bbox_valid"] is True
    assert kwargs["confidence"] == pytest.approx(0.75)
    assert kwargs["velocity_valid"] is False


def test_duplicate_or_nonfinite_geometry_is_rejected():
    tracker = TemporalEntityTracker()
    with pytest.raises(TemporalEntityTrackerError, match="duplicate"):
        tracker.update([observation(), observation()])
    with pytest.raises(TemporalEntityTrackerError, match="finite"):
        observation(x=float("nan"))
