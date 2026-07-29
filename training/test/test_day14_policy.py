from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from training.dataset import (  # noqa: E402
    FORBIDDEN_POLICY_FIELDS,
    FrozenFeatureDataset,
    policy_inputs_from_batch,
)
from training.day14_contract import run_contract  # noqa: E402
from training.losses import trajectory_policy_loss  # noqa: E402
from training.model import SmallPolicyConfig, SmallTrajectoryPolicy  # noqa: E402


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "model_small_v1.yaml"
)


def _inputs(batch_size: int, *, requires_grad: bool = False):
    entity_mask = torch.ones(batch_size, 16, dtype=torch.bool)
    return {
        "language": torch.randn(
            batch_size, 256, requires_grad=requires_grad
        ),
        "global_visual": torch.randn(
            batch_size, 576, requires_grad=requires_grad
        ),
        "entity_visual": torch.randn(
            batch_size, 16, 576, requires_grad=requires_grad
        ),
        "entity_geometry": torch.randn(
            batch_size, 16, 16, requires_grad=requires_grad
        ),
        "ego": torch.randn(batch_size, 2, requires_grad=requires_grad),
        "language_valid": torch.ones(batch_size, dtype=torch.bool),
        "global_visual_mask": torch.ones(batch_size, dtype=torch.bool),
        "entity_visual_mask": entity_mask.clone(),
        "entity_geometry_mask": entity_mask.clone(),
        "ego_valid": torch.ones(batch_size, dtype=torch.bool),
        "policy_input_valid": torch.ones(batch_size, dtype=torch.bool),
    }


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_policy_shapes_bounds_and_finite_values(batch_size: int) -> None:
    torch.manual_seed(42)
    model = SmallTrajectoryPolicy()
    inputs = _inputs(batch_size)

    output = model(**inputs)

    assert output.trajectory.shape == (batch_size, 20, 2)
    assert output.stop_logit.shape == (batch_size, 1)
    assert output.valid_mask.shape == (batch_size,)
    assert torch.isfinite(output.trajectory).all()
    assert torch.isfinite(output.stop_logit).all()
    assert (
        torch.max(torch.linalg.vector_norm(output.increments, dim=-1))
        <= 0.3 + 1.0e-6
    )


def test_all_entity_masks_false_remains_finite_and_deterministic() -> None:
    torch.manual_seed(17)
    model = SmallTrajectoryPolicy().eval()
    inputs = _inputs(2)
    inputs["entity_visual_mask"].zero_()
    inputs["entity_geometry_mask"].zero_()
    inputs["entity_visual"].fill_(float("nan"))
    inputs["entity_geometry"].fill_(float("nan"))

    first = model(**inputs)
    second = model(**inputs)

    assert torch.isfinite(first.trajectory).all()
    assert torch.isfinite(first.stop_logit).all()
    assert torch.equal(first.trajectory, second.trajectory)
    assert first.valid_mask.all()


def test_v2_entity_attention_is_language_conditioned() -> None:
    config = SmallPolicyConfig(
        language_conditioned_entity_attention=True,
    )
    model = SmallTrajectoryPolicy(config)

    assert model.entity_language_query is not None
    assert model.trainable_parameter_count() <= config.maximum_trainable_parameters

    language_a = torch.zeros(2, config.language_hidden)
    language_b = torch.ones(2, config.language_hidden)
    entity_tokens = torch.randn(2, config.entity_count, config.entity_hidden)
    query_a = model.entity_language_query(language_a)
    query_b = model.entity_language_query(language_b)
    score_a = torch.sum(entity_tokens * query_a.unsqueeze(1), dim=-1)
    score_b = torch.sum(entity_tokens * query_b.unsqueeze(1), dim=-1)

    assert torch.count_nonzero(score_a) == 0
    assert not torch.equal(score_a, score_b)


def test_language_only_attention_configuration_is_validated() -> None:
    config = SmallPolicyConfig(
        language_conditioned_entity_attention=True,
        entity_attention_mode="language_only",
    )
    model = SmallTrajectoryPolicy(config)

    assert model.entity_language_query is not None
    with pytest.raises(ValueError, match="requires"):
        SmallPolicyConfig(entity_attention_mode="language_only")


def test_geometry_remains_available_when_entity_visual_is_not_projectable() -> None:
    torch.manual_seed(19)
    model = SmallTrajectoryPolicy().eval()
    first_inputs = _inputs(1)
    first_inputs["entity_visual_mask"].zero_()
    first_inputs["entity_visual"].fill_(float("nan"))
    first_inputs["entity_geometry"].zero_()
    second_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in first_inputs.items()
    }
    second_inputs["entity_geometry"].fill_(1.0)

    first = model(**first_inputs)
    second = model(**second_inputs)

    assert torch.isfinite(first.trajectory).all()
    assert torch.isfinite(second.trajectory).all()
    assert not torch.equal(first.trajectory, second.trajectory)


def test_missing_required_modality_fails_closed_without_nan() -> None:
    model = SmallTrajectoryPolicy()
    inputs = _inputs(2)
    inputs["global_visual_mask"][0] = False
    inputs["global_visual"].data[0].fill_(float("nan"))

    output = model(**inputs)

    assert not bool(output.valid_mask[0])
    assert torch.count_nonzero(output.trajectory[0]) == 0
    assert output.stop_logit[0, 0] == 20.0
    assert torch.isfinite(output.trajectory).all()


def test_stop_logit_hard_gates_published_trajectory_motion() -> None:
    torch.manual_seed(29)
    model = SmallTrajectoryPolicy().eval()
    inputs = _inputs(1)
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(-20.0)
    moving = model(**inputs)
    with torch.no_grad():
        model.stop_head.bias.fill_(20.0)
    stopped = model(**inputs)

    assert torch.max(torch.abs(moving.trajectory)) > 1.0e-3
    assert torch.count_nonzero(stopped.increments) == 0
    assert torch.count_nonzero(stopped.trajectory) == 0


def test_active_nan_is_rejected() -> None:
    model = SmallTrajectoryPolicy()
    inputs = _inputs(1)
    inputs["ego"].data[0, 0] = float("nan")

    with pytest.raises(ValueError, match="ego contains NaN or Inf"):
        model(**inputs)


def test_cached_backbones_are_detached_but_policy_gradients_flow() -> None:
    torch.manual_seed(23)
    model = SmallTrajectoryPolicy()
    inputs = _inputs(8, requires_grad=True)
    output = model(**inputs)
    losses = trajectory_policy_loss(
        output,
        torch.zeros_like(output.trajectory),
        torch.zeros(8, 1),
    )

    losses["total"].backward()

    assert inputs["language"].grad is None
    assert inputs["global_visual"].grad is None
    assert inputs["entity_visual"].grad is None
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_parameter_count_below_frozen_limit() -> None:
    config = SmallPolicyConfig()
    model = SmallTrajectoryPolicy(config)

    assert model.trainable_parameter_count() < 2_000_000
    assert model.trainable_parameter_count() <= (
        config.maximum_trainable_parameters
    )


def test_loss_filters_invalid_samples_and_rejects_nan_targets() -> None:
    model = SmallTrajectoryPolicy()
    inputs = _inputs(2)
    inputs["policy_input_valid"][0] = False
    output = model(**inputs)
    target = torch.zeros_like(output.trajectory)
    stop = torch.zeros(2, 1)

    losses = trajectory_policy_loss(output, target, stop)

    assert int(losses["valid_samples"]) == 1
    assert torch.isfinite(losses["total"])
    target[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="target trajectory"):
        trajectory_policy_loss(output, target, stop)


def _write_cache(path: Path, *, color_leak: bool = False) -> None:
    path.mkdir(parents=True)
    frame_count = 3
    instruction_count = 2
    sample_count = 6
    language = np.arange(
        instruction_count * 256, dtype=np.float32
    ).reshape(instruction_count, 256)
    np.savez_compressed(
        path / "language.npz",
        instruction_ids=np.asarray(["train_0", "validation_0"]),
        instruction_texts=np.asarray(["训练指令", "验证指令"]),
        language_splits=np.asarray(["train", "validation"]),
        embeddings=language,
    )
    entity_geometry = np.zeros((frame_count, 16, 16), dtype=np.float32)
    if color_leak:
        entity_geometry[0, 0, 14] = 1.0
    entity_mask = np.zeros((frame_count, 16), dtype=np.bool_)
    entity_mask[:, :2] = True
    visual_mask = entity_mask.copy()
    np.savez_compressed(
        path / "frames_000.npz",
        frame_indices=np.arange(frame_count, dtype=np.int64),
        frame_stamps_us=np.arange(
            100_000, 100_000 + frame_count * 100_000, 100_000
        ),
        frame_keys=np.asarray(
            [
                f"RUN_DAY14:140001:{index}:{(index + 1) * 100000}"
                for index in range(frame_count)
            ]
        ),
        source_frame_sha256=np.asarray(["a" * 64] * frame_count),
        image_sha256=np.asarray(["b" * 64] * frame_count),
        global_visual=np.ones((frame_count, 576), dtype=np.float32),
        global_visual_mask=np.asarray([True, False, True]),
        entity_visual=np.ones(
            (frame_count, 16, 576), dtype=np.float32
        )
        * visual_mask[:, :, None],
        entity_visual_mask=visual_mask,
        entity_features=entity_geometry,
        entity_mask=entity_mask,
        entity_ids=np.asarray(
            [["target_0", "target_1"] + [""] * 14] * frame_count
        ),
        ego=np.zeros((frame_count, 2), dtype=np.float32),
        ego_valid=np.ones(frame_count, dtype=np.bool_),
        policy_input_valid=np.asarray([True, False, True]),
        sample_ids=np.asarray(
            [f"sample_{index}" for index in range(sample_count)]
        ),
        sample_frame_rows=np.repeat(
            np.arange(frame_count, dtype=np.int32), instruction_count
        ),
        sample_instruction_rows=np.tile(
            np.arange(instruction_count, dtype=np.int16), frame_count
        ),
        expert_trajectories=np.zeros(
            (sample_count, 20, 2), dtype=np.float32
        ),
        expert_safe_stop=np.asarray(
            [False, True, False, True, False, True]
        ),
        expert_selected_entity_ids=np.asarray([""] * sample_count),
    )
    (path / "manifest.json").write_text(
        json.dumps({"run_id": "RUN_DAY14"}), encoding="utf-8"
    )


def test_dataset_filters_runs_frames_and_language_without_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "RUN_DAY14"
    _write_cache(cache)
    monkeypatch.setattr(
        "training.dataset.validate_feature_cache", lambda _: {"passed": True}
    )

    dataset = FrozenFeatureDataset(
        [cache],
        selected_split="train",
        split_assignments={"RUN_DAY14": "train"},
        allowed_language_splits={"train"},
        frame_stride=2,
    )
    item = dataset[0]

    assert len(dataset) == 2
    assert not (FORBIDDEN_POLICY_FIELDS & set(item))
    assert item["language"].shape == (256,)
    assert item["entity_visual"].shape == (16, 576)
    assert item["entity_geometry"].shape == (16, 16)
    assert torch.count_nonzero(item["entity_geometry"][:, 14:16]) == 0
    assert item["target_trajectory"].shape == (20, 2)
    batch = {
        key: value.unsqueeze(0)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
        and key not in {"target_trajectory", "target_stop"}
    }
    assert set(policy_inputs_from_batch(batch)) == {
        "language",
        "global_visual",
        "entity_visual",
        "entity_geometry",
        "ego",
        "language_valid",
        "global_visual_mask",
        "entity_visual_mask",
        "entity_geometry_mask",
        "ego_valid",
        "policy_input_valid",
    }


def test_dataset_rejects_privileged_color_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "RUN_DAY14"
    _write_cache(cache, color_leak=True)
    monkeypatch.setattr(
        "training.dataset.validate_feature_cache", lambda _: {"passed": True}
    )

    with pytest.raises(ValueError, match="privileged entity color"):
        FrozenFeatureDataset([cache])


def test_executable_contract_writes_resource_report(tmp_path: Path) -> None:
    report_path = tmp_path / "day14_report.json"

    report = run_contract(
        CONFIG_PATH,
        report_path,
        device_name="cpu",
        seed=42,
    )

    assert report["passed"] is True
    assert report["output_shapes"] == {
        "1": [1, 20, 2],
        "2": [2, 20, 2],
        "8": [8, 20, 2],
    }
    assert report["trainable_parameter_count"] < 2_000_000
    assert report["checkpoint_size_bytes"] > 0
    assert report["peak_memory_bytes"] > 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
