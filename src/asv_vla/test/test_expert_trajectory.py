import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from asv_vla.evaluate_expert_labels import evaluate_expert_labels
from asv_vla.expert_trajectory import (
    ExpertTask,
    ExpertTrajectoryError,
    generate_expert_trajectory,
    select_target,
    task_from_labels,
)
from asv_vla.language_intervention_dataset import read_jsonl
from asv_vla.trajectory_contract import ACTION_DIM, DT_SEC, HORIZON


def entity(
    entity_id,
    *,
    color="red",
    x=8.0,
    y=0.0,
    vx=0.0,
    vy=0.0,
    is_target=True,
    visible=True,
    valid=True,
):
    return SimpleNamespace(
        entity_id=entity_id,
        class_name="boat",
        color=color,
        is_target=is_target,
        visible=visible,
        relative_x=x,
        relative_y=y,
        relative_z=0.0,
        relative_velocity_x=vx,
        relative_velocity_y=vy,
        relative_velocity_z=0.0,
        valid=valid,
    )


def endpoint(result):
    return result.delta_p_xy[-2], result.delta_p_xy[-1]


def test_task_labels_accept_only_frozen_follow_stop_scope():
    assert task_from_labels("follow", "color:red", "3m") == ExpertTask(
        action="follow",
        target_attribute="color:red",
        desired_distance_m=3.0,
    )
    assert task_from_labels("STOP", "none", "none") == ExpertTask(
        action="stop",
        target_attribute="none",
        desired_distance_m=0.0,
    )

    with pytest.raises(ExpertTrajectoryError, match="unsupported action"):
        task_from_labels("dock", "none", "none")
    with pytest.raises(ExpertTrajectoryError, match="target_attribute"):
        task_from_labels("follow", "class:boat", "3m")
    with pytest.raises(ExpertTrajectoryError, match="distance_bucket"):
        task_from_labels("follow", "color:red", "5m")


def test_stop_is_an_explicit_safe_zero_label():
    task = task_from_labels("stop", "none", "none")
    result = generate_expert_trajectory(task, [])

    assert result.safe_stop
    assert result.selected_entity_id == ""
    assert result.delta_p_xy == (0.0,) * (HORIZON * ACTION_DIM)


def test_three_and_ten_metre_follow_labels_move_to_correct_standoff():
    target = entity("target", x=8.0)
    near = generate_expert_trajectory(
        task_from_labels("follow", "color:red", "3m"),
        [target],
    )
    far = generate_expert_trajectory(
        task_from_labels("follow", "color:red", "10m"),
        [target],
    )

    assert endpoint(near) == pytest.approx((5.0, 0.0), abs=1.0e-6)
    assert endpoint(far) == pytest.approx((-2.0, 0.0), abs=1.0e-6)
    assert near.delta_p_xy != far.delta_p_xy
    assert not near.safe_stop
    assert not far.safe_stop


def test_color_and_bearing_selectors_choose_different_targets():
    entities = [
        entity("red", color="red", x=8.0, y=1.0),
        entity("blue", color="blue", x=8.0, y=-1.0),
    ]

    assert select_target(entities, "color:red").entity_id == "red"
    assert select_target(entities, "color:blue").entity_id == "blue"
    assert select_target(entities, "bearing:left").entity_id == "red"
    assert select_target(entities, "bearing:right").entity_id == "blue"

    red = generate_expert_trajectory(
        task_from_labels("follow", "color:red", "3m"), entities
    )
    blue = generate_expert_trajectory(
        task_from_labels("follow", "color:blue", "3m"), entities
    )
    left = generate_expert_trajectory(
        task_from_labels("follow", "bearing:left", "3m"), entities
    )
    right = generate_expert_trajectory(
        task_from_labels("follow", "bearing:right", "3m"), entities
    )
    assert endpoint(red)[1] > 0.0
    assert endpoint(blue)[1] < 0.0
    assert left.selected_entity_id == "red"
    assert right.selected_entity_id == "blue"


def test_bearing_selector_ignores_centerline_float_noise():
    near_center = [
        entity("positive_noise", color="red", y=1.0e-8),
        entity("negative_noise", color="blue", y=-1.0e-8),
    ]

    with pytest.raises(ExpertTrajectoryError, match="no valid visible"):
        select_target(near_center, "bearing:left")
    with pytest.raises(ExpertTrajectoryError, match="no valid visible"):
        select_target(near_center, "bearing:right")


def test_constant_velocity_prediction_and_speed_bound_are_deterministic():
    task = task_from_labels("follow", "color:red", "3m")
    static = generate_expert_trajectory(task, [entity("target", x=8.0)])
    moving_target = entity("target", x=8.0, vx=1.0)
    moving = generate_expert_trajectory(task, [moving_target])
    repeated = generate_expert_trajectory(task, [moving_target])

    assert moving == repeated
    assert endpoint(moving)[0] > endpoint(static)[0]
    previous = (0.0, 0.0)
    for index in range(HORIZON):
        current = (
            moving.delta_p_xy[2 * index],
            moving.delta_p_xy[2 * index + 1],
        )
        assert math.dist(previous, current) <= 1.5 * DT_SEC + 1.0e-9
        previous = current


def test_invalid_or_ambiguous_target_data_fail_closed():
    task = task_from_labels("follow", "color:red", "3m")
    with pytest.raises(ExpertTrajectoryError, match="no valid visible"):
        generate_expert_trajectory(
            task, [entity("hidden", visible=False)]
        )
    with pytest.raises(ExpertTrajectoryError, match="NaN or Inf"):
        generate_expert_trajectory(
            task, [entity("nan", x=float("nan"))]
        )
    with pytest.raises(ExpertTrajectoryError, match="duplicate"):
        generate_expert_trajectory(
            task, [entity("same", x=8.0), entity("same", x=9.0)]
        )
    with pytest.raises(ExpertTrajectoryError, match="max_speed_mps"):
        generate_expert_trajectory(task, [entity("target")], max_speed_mps=0)


def test_all_labels_and_contrast_pairs_change_correctly():
    repository = Path(__file__).resolve().parents[3]
    instructions = read_jsonl(
        repository / "dataset/language/instructions.jsonl"
    )
    pairs = read_jsonl(
        repository / "dataset/language/contrast_pairs.jsonl"
    )

    report = evaluate_expert_labels(instructions, pairs)

    assert report["passed"]
    assert report["instruction_count"] == 90
    assert report["changed_contrast_pair_count"] == 24
    assert report["unique_task_label_count"] == 9
    assert report["trajectory_shape"] == [20, 2]


def test_expert_message_retains_full_frame_identity_and_is_registered():
    repository = Path(__file__).resolve().parents[3]
    message_path = (
        repository
        / "src/asv_jetson_interfaces/msg/ExpertTrajectory.msg"
    )
    fields = [
        line.strip()
        for line in message_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    cmake = (
        repository / "src/asv_jetson_interfaces/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert fields[:5] == [
        "int64 stamp_us",
        "string run_id",
        "int64 scene_seed",
        "uint64 frame_index",
        "string frame_id",
    ]
    assert "float32[] delta_p_xy" in fields
    assert '"msg/ExpertTrajectory.msg"' in cmake
