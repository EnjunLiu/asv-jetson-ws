"""Evaluate Day 9 expert labels against the frozen Day 3 dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .expert_trajectory import (
    ExpertTrajectoryError,
    generate_expert_trajectory,
    task_from_labels,
)
from .language_intervention_dataset import read_jsonl
from .trajectory_contract import ACTION_DIM, DT_SEC, HORIZON


EXPECTED_TARGET_IDS = {
    "color:red": "target_red",
    "color:blue": "target_blue",
    "bearing:left": "target_left",
    "bearing:right": "target_right",
}


def _entity(
    entity_id: str,
    color: str,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=entity_id,
        class_name="boat",
        color=color,
        is_target=True,
        visible=True,
        relative_x=x,
        relative_y=y,
        relative_z=0.0,
        relative_velocity_x=vx,
        relative_velocity_y=vy,
        relative_velocity_z=0.0,
        valid=True,
    )


def canonical_entities() -> list[SimpleNamespace]:
    # Color targets lie on the centerline so bearing selectors remain
    # independent. Positions deliberately differ so target interventions
    # produce distinct expert labels.
    return [
        _entity("target_red", "red", 12.0, 0.0, 0.2, 0.0),
        _entity("target_blue", "blue", 8.0, 0.0, 0.1, 0.0),
        _entity("target_left", "white", 10.0, 5.0, 0.0, 0.1),
        _entity("target_right", "white", 10.0, -5.0, 0.0, -0.1),
    ]


def evaluate_expert_labels(
    instructions: list[dict[str, Any]],
    contrast_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not instructions:
        raise ExpertTrajectoryError("instruction dataset is empty")
    entities = canonical_entities()
    trajectories_by_instruction: dict[str, tuple[float, ...]] = {}
    label_trajectories: dict[
        tuple[str, str, str], tuple[float, ...]
    ] = {}
    action_counts: Counter[str] = Counter()
    selector_counts: Counter[str] = Counter()

    for record in instructions:
        instruction_id = str(record.get("instruction_id", ""))
        task = task_from_labels(
            str(record.get("action", "")),
            str(record.get("target_attribute", "")),
            str(record.get("distance_bucket", "")),
        )
        result = generate_expert_trajectory(task, entities)
        repeated = generate_expert_trajectory(task, entities)
        if result != repeated:
            raise ExpertTrajectoryError(
                f"non-deterministic expert output: {instruction_id}"
            )
        if len(result.delta_p_xy) != HORIZON * ACTION_DIM:
            raise ExpertTrajectoryError(
                f"wrong shape for instruction: {instruction_id}"
            )
        if not all(math.isfinite(value) for value in result.delta_p_xy):
            raise ExpertTrajectoryError(
                f"NaN/Inf for instruction: {instruction_id}"
            )
        max_step = 0.0
        previous_x = 0.0
        previous_y = 0.0
        for index in range(HORIZON):
            x = result.delta_p_xy[index * ACTION_DIM]
            y = result.delta_p_xy[index * ACTION_DIM + 1]
            max_step = max(
                max_step, math.hypot(x - previous_x, y - previous_y)
            )
            previous_x, previous_y = x, y
        if max_step > 1.5 * DT_SEC + 1.0e-6:
            raise ExpertTrajectoryError(
                f"speed bound exceeded: {instruction_id}"
            )

        if task.action == "stop":
            if (
                not result.safe_stop
                or result.selected_entity_id
                or any(result.delta_p_xy)
            ):
                raise ExpertTrajectoryError(
                    f"invalid STOP label: {instruction_id}"
                )
        else:
            expected_id = EXPECTED_TARGET_IDS[task.target_attribute]
            if result.safe_stop or result.selected_entity_id != expected_id:
                raise ExpertTrajectoryError(
                    f"wrong FOLLOW target: {instruction_id}"
                )
            selector_counts[task.target_attribute] += 1

        key = (
            task.action,
            task.target_attribute,
            str(record.get("distance_bucket", "")),
        )
        existing = label_trajectories.setdefault(key, result.delta_p_xy)
        if existing != result.delta_p_xy:
            raise ExpertTrajectoryError(
                f"same label produced different trajectory: {key}"
            )
        trajectories_by_instruction[instruction_id] = result.delta_p_xy
        action_counts[task.action] += 1

    if len(label_trajectories) != 9:
        raise ExpertTrajectoryError(
            f"expected 9 task labels, got {len(label_trajectories)}"
        )
    if len(set(label_trajectories.values())) != len(label_trajectories):
        raise ExpertTrajectoryError(
            "different task labels produced identical trajectories"
        )

    changed_pairs = 0
    for pair in contrast_pairs:
        instruction_ids = pair.get("instruction_ids")
        if not isinstance(instruction_ids, list) or len(instruction_ids) != 2:
            raise ExpertTrajectoryError("invalid contrast pair instruction_ids")
        left = trajectories_by_instruction.get(str(instruction_ids[0]))
        right = trajectories_by_instruction.get(str(instruction_ids[1]))
        if left is None or right is None:
            raise ExpertTrajectoryError(
                f"contrast pair references unknown instruction: {instruction_ids}"
            )
        if left == right:
            raise ExpertTrajectoryError(
                f"contrast pair did not change trajectory: "
                f"{pair.get('pair_id')}"
            )
        changed_pairs += 1

    return {
        "passed": True,
        "instruction_count": len(instructions),
        "contrast_pair_count": len(contrast_pairs),
        "changed_contrast_pair_count": changed_pairs,
        "unique_task_label_count": len(label_trajectories),
        "action_counts": dict(sorted(action_counts.items())),
        "selector_counts": dict(sorted(selector_counts.items())),
        "trajectory_shape": [HORIZON, ACTION_DIM],
        "dt_sec": DT_SEC,
        "deterministic": True,
        "finite": True,
        "speed_bound_mps": 1.5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic Day 9 expert labels."
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        default=Path("dataset/language/instructions.jsonl"),
    )
    parser.add_argument(
        "--contrast-pairs",
        type=Path,
        default=Path("dataset/language/contrast_pairs.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = evaluate_expert_labels(
            read_jsonl(args.instructions),
            read_jsonl(args.contrast_pairs),
        )
    except (ExpertTrajectoryError, OSError, ValueError) as exc:
        print(f"EXPERT_LABELS_FAIL: {exc}")
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "EXPERT_LABELS_PASS "
        f"instructions={report['instruction_count']} "
        f"contrast_pairs={report['changed_contrast_pair_count']} "
        f"labels={report['unique_task_label_count']} "
        f"shape={HORIZON}x{ACTION_DIM}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
