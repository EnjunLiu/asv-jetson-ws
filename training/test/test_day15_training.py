from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from training.dataset import (  # noqa: E402
    EpochSynonymDataset,
    InstructionMetadata,
)
from training.metrics import (  # noqa: E402
    compute_policy_metrics,
    fit_label_mean_baseline,
    improvement_fraction,
    predict_label_mean_baseline,
)
from training.train import _acceptance  # noqa: E402


class _FakeFeatureDataset:
    def __init__(self) -> None:
        self.samples = [
            ("RUN", "RUN:1:0:1", "red_a"),
            ("RUN", "RUN:1:0:1", "red_b"),
            ("RUN", "RUN:1:0:1", "stop_a"),
            ("RUN", "RUN:1:1:2", "red_a"),
            ("RUN", "RUN:1:1:2", "red_b"),
            ("RUN", "RUN:1:1:2", "stop_a"),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def sample_metadata(self, index: int):
        run_id, frame_key, instruction_id = self.samples[index]
        return {
            "run_id": run_id,
            "frame_key": frame_key,
            "sample_id": f"sample_{index}",
            "instruction_id": instruction_id,
            "target_stop": instruction_id == "stop_a",
        }

    def __getitem__(self, index: int):
        metadata = self.sample_metadata(index)
        return {
            "instruction_id": metadata["instruction_id"],
            "target_trajectory": torch.zeros(20, 2),
            "target_stop": torch.tensor(
                [float(metadata["target_stop"])], dtype=torch.float32
            ),
        }


def _instruction(
    instruction_id: str,
    intent_group: str,
    action: str,
    target_attribute: str,
    distance_bucket: str,
) -> InstructionMetadata:
    return InstructionMetadata(
        instruction_id=instruction_id,
        intent_group=intent_group,
        action=action,
        target_attribute=target_attribute,
        distance_bucket=distance_bucket,
        split="train",
    )


def test_epoch_synonym_dataset_selects_one_per_frame_label() -> None:
    instructions = {
        "red_a": _instruction(
            "red_a", "follow_red_3m", "follow", "color:red", "3m"
        ),
        "red_b": _instruction(
            "red_b", "follow_red_3m", "follow", "color:red", "3m"
        ),
        "stop_a": _instruction(
            "stop_a", "stop", "stop", "none", "none"
        ),
    }
    first = EpochSynonymDataset(_FakeFeatureDataset(), instructions, seed=17)
    second = EpochSynonymDataset(_FakeFeatureDataset(), instructions, seed=17)

    assert len(first) == 4
    assert [first[index]["instruction_id"] for index in range(len(first))] == [
        second[index]["instruction_id"] for index in range(len(second))
    ]
    first.set_epoch(1)
    groups = [
        (
            first[index]["metadata"]["task_label"],
            first[index]["instruction_id"],
        )
        for index in range(len(first))
    ]
    assert sum(label.startswith("follow|") for label, _ in groups) == 2
    assert sum(label.startswith("stop|") for label, _ in groups) == 2


def test_metrics_expert_and_speed_contract() -> None:
    target = np.zeros((2, 20, 2), dtype=np.float32)
    target[0, :, 0] = np.arange(1, 21, dtype=np.float32) * 0.3
    stop = np.asarray([False, True])
    labels = ["follow|color:red|3m", "stop|none|none"]
    logits = np.asarray([-20.0, 20.0], dtype=np.float32)

    metrics = compute_policy_metrics(target, target, logits, stop, labels)

    assert metrics["ade_m"] == pytest.approx(0.0)
    assert metrics["fde_m"] == pytest.approx(0.0)
    assert metrics["stop_classification"]["f1"] == pytest.approx(1.0)
    assert metrics["stop_drift"]["within_0_10m_rate"] == pytest.approx(1.0)
    assert metrics["speed_constraint"]["violation_count"] == 0
    assert metrics["aggregate_labels"]["red"]["sample_count"] == 1
    assert metrics["aggregate_labels"]["3m"]["sample_count"] == 1


def test_label_mean_baseline_and_improvement() -> None:
    trajectories = np.zeros((4, 20, 2), dtype=np.float32)
    trajectories[1, :, 0] = 2.0
    trajectories[2, :, 1] = 1.0
    trajectories[3, :, 1] = 3.0
    labels = ["a", "a", "b", "b"]

    means = fit_label_mean_baseline(trajectories, labels)
    prediction, logits = predict_label_mean_baseline(means, ["a", "b"])

    assert np.allclose(prediction[0, :, 0], 1.0)
    assert np.allclose(prediction[1, :, 1], 2.0)
    assert np.all(logits == -20.0)
    assert improvement_fraction(0.7, 1.0) == pytest.approx(0.3)


def test_acceptance_requires_every_frozen_gate() -> None:
    metrics = {
        "ade_m": 0.6,
        "fde_m": 0.6,
        "stop_drift": {"within_0_10m_rate": 0.95},
        "stop_classification": {"f1": 0.95},
        "speed_constraint": {"violation_rate": 0.0},
        "invalid_count": 0,
    }
    baseline = {"ade_m": 1.0, "fde_m": 1.0}
    config = {
        "minimum_ade_improvement_over_label_mean": 0.30,
        "minimum_fde_improvement_over_label_mean": 0.30,
        "minimum_stop_within_0_10m_rate": 0.95,
        "minimum_stop_f1": 0.95,
        "maximum_speed_violation_rate": 0.0,
        "maximum_invalid_count": 0,
    }

    result = _acceptance(metrics, baseline, config)

    assert result["passed"] is True
    metrics["stop_classification"]["f1"] = 0.94
    assert _acceptance(metrics, baseline, config)["passed"] is False
