from pathlib import Path

import pytest

from asv_vla.generate_language_interventions import generate
from asv_vla.language_intervention_dataset import (
    LanguageDatasetError,
    load_and_validate,
    validate_language_dataset,
)


WORKSPACE = Path(__file__).resolve().parents[3]
DATASET_DIR = WORKSPACE / "dataset" / "language"


def load_checked_in_dataset():
    return load_and_validate(
        DATASET_DIR / "instructions.jsonl",
        DATASET_DIR / "contrast_pairs.jsonl",
    )


def test_checked_in_dataset_meets_coverage_contract():
    instructions, pairs, report = load_checked_in_dataset()

    assert len(instructions) == 90
    assert len(pairs) == 24
    assert report["conflicting_scene_count"] == 24
    assert set(report["intent_group_counts"].values()) == {10}
    assert report["split_counts"] == {
        "test": 18,
        "train": 54,
        "validation": 18,
    }
    assert report["intervention_counts"] == {
        "action": 6,
        "distance": 6,
        "target_bearing": 6,
        "target_color": 6,
    }
    assert all(report["acceptance"].values())


def test_split_template_families_are_disjoint():
    _, _, report = load_checked_in_dataset()
    families = {
        split: set(values)
        for split, values in report["template_families"].items()
    }

    assert not families["train"] & families["validation"]
    assert not families["train"] & families["test"]
    assert not families["validation"] & families["test"]


def test_generated_dataset_matches_checked_in_files(tmp_path):
    generated_instructions, generated_pairs = generate(tmp_path)

    assert generated_instructions.read_text(encoding="utf-8") == (
        DATASET_DIR / "instructions.jsonl"
    ).read_text(encoding="utf-8")
    assert generated_pairs.read_text(encoding="utf-8") == (
        DATASET_DIR / "contrast_pairs.jsonl"
    ).read_text(encoding="utf-8")


def test_nonconflicting_action_pair_is_rejected():
    instructions, pairs, _ = load_checked_in_dataset()
    invalid_pairs = [dict(pair) for pair in pairs]
    invalid_pairs[0] = {
        **invalid_pairs[0],
        "intervention_type": "action",
    }

    with pytest.raises(
        LanguageDatasetError,
        match="labels do not match intervention_type=action",
    ):
        validate_language_dataset(instructions, invalid_pairs)
