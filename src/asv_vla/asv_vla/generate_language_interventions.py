from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .language_intervention_dataset import (
    default_dataset_dir,
    validate_language_dataset,
    write_jsonl,
)


FOLLOW_TARGETS = {
    "red": ("红色目标船", "color:red"),
    "blue": ("蓝色目标船", "color:blue"),
    "left": ("左侧目标船", "bearing:left"),
    "right": ("右侧目标船", "bearing:right"),
}

FOLLOW_TEMPLATES = (
    (
        "train",
        "train_direct",
        "跟随{target}，保持{distance}米距离",
    ),
    (
        "train",
        "train_gap",
        "跟在{target}后方，把间距维持在{distance}米",
    ),
    (
        "train",
        "train_lock",
        "锁定{target}并以{distance}米间隔持续跟踪",
    ),
    (
        "train",
        "train_task",
        "执行跟随任务：目标是{target}，期望距离{distance}米",
    ),
    (
        "train",
        "train_navigation",
        "驶向{target}的跟随位置，稳定保持{distance}米",
    ),
    (
        "train",
        "train_tracking",
        "追踪{target}，不要让距离偏离{distance}米",
    ),
    (
        "validation",
        "validation_polite",
        "请跟住{target}，与其相隔约{distance}米",
    ),
    (
        "validation",
        "validation_relative",
        "将{target}作为跟随对象，控制在{distance}米左右",
    ),
    (
        "test",
        "test_natural",
        "盯住{target}航行，留出{distance}米间距",
    ),
    (
        "test",
        "test_concise",
        "跟随{target}，距离设为{distance}米",
    ),
)

STOP_TEXTS = (
    "立即停止并保持安全停机状态",
    "停止当前跟随任务，不再继续前进",
    "终止航行指令，让无人船安全停下",
    "取消当前任务并执行确定性停止",
    "不要继续追踪目标，马上停船",
    "中止前进，进入安全停止状态",
    "请停止跟随并保持安全状态",
    "结束当前航行任务，立即停下",
    "先别再向前走，把船安全停住",
    "停止",
)


def instruction_id(intent_group: str, template_index: int) -> str:
    return f"{intent_group}_{template_index:02d}"


def build_instructions() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target_key, (target_text, target_attribute) in FOLLOW_TARGETS.items():
        for distance in (3, 10):
            intent_group = f"follow_{target_key}_{distance}m"
            for template_index, (
                split,
                template_family,
                template,
            ) in enumerate(FOLLOW_TEMPLATES, start=1):
                records.append({
                    "schema_version": 1,
                    "instruction_id": instruction_id(
                        intent_group, template_index
                    ),
                    "text": template.format(
                        target=target_text,
                        distance=distance,
                    ),
                    "intent_group": intent_group,
                    "action": "follow",
                    "target_attribute": target_attribute,
                    "distance_bucket": f"{distance}m",
                    "split": split,
                    "template_family": template_family,
                })

    for template_index, (
        split,
        template_family,
        _,
    ) in enumerate(FOLLOW_TEMPLATES, start=1):
        records.append({
            "schema_version": 1,
            "instruction_id": instruction_id("stop", template_index),
            "text": STOP_TEXTS[template_index - 1],
            "intent_group": "stop",
            "action": "stop",
            "target_attribute": "none",
            "distance_bucket": "none",
            "split": split,
            "template_family": template_family,
        })
    return records


def build_contrast_pairs(
    instructions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_id = {
        record["instruction_id"]: record for record in instructions
    }
    pairs: list[dict[str, Any]] = []

    def add_pair(
        intervention_type: str,
        left_group: str,
        right_group: str,
        template_index: int,
    ) -> None:
        left_id = instruction_id(left_group, template_index)
        right_id = instruction_id(right_group, template_index)
        left = records_by_id[left_id]
        right = records_by_id[right_id]
        if left["split"] != right["split"]:
            raise RuntimeError("contrast pair splits must match")
        ordinal = 1 + sum(
            pair["intervention_type"] == intervention_type
            for pair in pairs
        )
        pairs.append({
            "schema_version": 1,
            "pair_id": f"{intervention_type}_{ordinal:02d}",
            "scene_seed": 43000 + len(pairs) + 1,
            "split": left["split"],
            "intervention_type": intervention_type,
            "instruction_ids": [left_id, right_id],
            "expected_effect": "different_trajectory",
        })

    template_indices = (1, 2, 7, 8, 9, 10)
    for template_index, distance in zip(
        template_indices, (3, 10, 3, 10, 3, 10)
    ):
        add_pair(
            "target_color",
            f"follow_red_{distance}m",
            f"follow_blue_{distance}m",
            template_index,
        )

    for template_index, distance in zip(
        template_indices, (3, 10, 3, 10, 3, 10)
    ):
        add_pair(
            "target_bearing",
            f"follow_left_{distance}m",
            f"follow_right_{distance}m",
            template_index,
        )

    distance_targets = ("red", "blue", "left", "right", "red", "blue")
    for template_index, target in zip(
        template_indices, distance_targets
    ):
        add_pair(
            "distance",
            f"follow_{target}_3m",
            f"follow_{target}_10m",
            template_index,
        )

    action_targets = (
        "red_3m",
        "blue_10m",
        "left_3m",
        "right_10m",
        "red_10m",
        "blue_3m",
    )
    for template_index, target in zip(
        template_indices, action_targets
    ):
        add_pair(
            "action",
            f"follow_{target}",
            "stop",
            template_index,
        )
    return pairs


def generate(output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    instructions = build_instructions()
    pairs = build_contrast_pairs(instructions)
    validate_language_dataset(instructions, pairs)

    instructions_path = destination / "instructions.jsonl"
    pairs_path = destination / "contrast_pairs.jsonl"
    write_jsonl(instructions_path, instructions)
    write_jsonl(pairs_path, pairs)
    return instructions_path, pairs_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate paired language intervention data."
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_dataset_dir()),
        help="Directory for instructions.jsonl and contrast_pairs.jsonl.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    instructions_path, pairs_path = generate(args.output_dir)
    print(f"instructions={instructions_path}")
    print(f"contrast_pairs={pairs_path}")
    print("LANGUAGE_INTERVENTION_DATA_GENERATED")


if __name__ == "__main__":
    main()
