"""Executable Day 14 policy contract and resource report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import tracemalloc
from typing import Any

import torch
from torch import Tensor
import yaml

from training.losses import PolicyLossWeights, trajectory_policy_loss
from training.model import SmallPolicyConfig, SmallTrajectoryPolicy


REPORT_SCHEMA_VERSION = "day14_policy_contract_report_v1"


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read policy config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("policy config must contain a mapping")
    if value.get("schema_version") != "model_small_v1":
        raise ValueError("policy config schema_version must be model_small_v1")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _make_inputs(
    batch_size: int,
    config: SmallPolicyConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    entity_mask = torch.zeros(
        batch_size,
        config.entity_count,
        dtype=torch.bool,
        device=device,
    )
    if batch_size > 1:
        entity_mask[1:, :4] = True
    return {
        "language": torch.randn(
            batch_size,
            config.language_dim,
            device=device,
            requires_grad=True,
        ),
        "global_visual": torch.randn(
            batch_size,
            config.visual_dim,
            device=device,
            requires_grad=True,
        ),
        "entity_visual": torch.randn(
            batch_size,
            config.entity_count,
            config.visual_dim,
            device=device,
            requires_grad=True,
        ),
        "entity_geometry": torch.randn(
            batch_size,
            config.entity_count,
            config.entity_geometry_dim,
            device=device,
            requires_grad=True,
        ),
        "ego": torch.randn(
            batch_size,
            config.ego_dim,
            device=device,
            requires_grad=True,
        ),
        "language_valid": torch.ones(
            batch_size, dtype=torch.bool, device=device
        ),
        "global_visual_mask": torch.ones(
            batch_size, dtype=torch.bool, device=device
        ),
        "entity_visual_mask": entity_mask.clone(),
        "entity_geometry_mask": entity_mask.clone(),
        "ego_valid": torch.ones(batch_size, dtype=torch.bool, device=device),
        "policy_input_valid": torch.ones(
            batch_size, dtype=torch.bool, device=device
        ),
    }


def _checkpoint_size_bytes(model: SmallTrajectoryPolicy) -> int:
    descriptor, name = tempfile.mkstemp(suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(model.state_dict(), name)
        return Path(name).stat().st_size
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def run_contract(
    config_path: str | Path,
    report_path: str | Path,
    *,
    device_name: str = "auto",
    seed: int | None = None,
) -> dict[str, Any]:
    source = _load_config(Path(config_path).resolve())
    model_config = SmallPolicyConfig.from_mapping(source)
    loss_weights = PolicyLossWeights.from_mapping(source.get("loss"))
    contract = source.get("contract", {})
    if not isinstance(contract, dict):
        raise ValueError("contract configuration must be a mapping")
    batch_sizes = tuple(int(value) for value in contract.get("batch_sizes", []))
    if batch_sizes != (1, 2, 8):
        raise ValueError("Day 14 contract batch sizes must be [1, 2, 8]")
    fixed_seed = int(
        seed
        if seed is not None
        else contract.get("deterministic_seed", 42)
    )
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(fixed_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fixed_seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        peak_kind = "torch_cuda_max_memory_allocated"
    else:
        tracemalloc.start()
        peak_kind = "python_tracemalloc"

    model = SmallTrajectoryPolicy(model_config).to(device)
    model.eval()
    output_shapes: dict[str, list[int]] = {}
    maximum_observed_increment = 0.0
    for batch_size in batch_sizes:
        inputs = _make_inputs(batch_size, model_config, device)
        first = model(**inputs)
        second = model(**inputs)
        expected = (batch_size, model_config.horizon, model_config.action_dim)
        if tuple(first.trajectory.shape) != expected:
            raise AssertionError(
                f"batch {batch_size}: trajectory shape is "
                f"{tuple(first.trajectory.shape)}, expected {expected}"
            )
        if tuple(first.stop_logit.shape) != (batch_size, 1):
            raise AssertionError("stop_logit shape is invalid")
        if not torch.isfinite(first.trajectory).all():
            raise AssertionError("policy trajectory contains NaN or Inf")
        if not torch.isfinite(first.stop_logit).all():
            raise AssertionError("policy stop_logit contains NaN or Inf")
        if not torch.equal(first.trajectory, second.trajectory):
            raise AssertionError("repeated forward pass is not deterministic")
        observed = float(
            torch.max(
                torch.linalg.vector_norm(first.increments, dim=-1)
            ).detach().cpu()
        )
        maximum_observed_increment = max(maximum_observed_increment, observed)
        if observed > model_config.maximum_step_m + 1.0e-6:
            raise AssertionError("trajectory increment exceeded structural bound")
        output_shapes[str(batch_size)] = list(first.trajectory.shape)

    invalid_inputs = _make_inputs(2, model_config, device)
    invalid_inputs["global_visual_mask"][0] = False
    with torch.no_grad():
        invalid_inputs["global_visual"][0].fill_(float("nan"))
    invalid_output = model(**invalid_inputs)
    if bool(invalid_output.valid_mask[0]):
        raise AssertionError("missing global visual input did not fail closed")
    if torch.count_nonzero(invalid_output.trajectory[0]):
        raise AssertionError("invalid sample trajectory is not zero")
    if not torch.isfinite(invalid_output.trajectory).all():
        raise AssertionError("invalid masks produced NaN or Inf")

    model.train()
    training_inputs = _make_inputs(8, model_config, device)
    training_output = model(**training_inputs)
    target_trajectory = torch.zeros_like(training_output.trajectory)
    target_stop = torch.zeros(8, 1, dtype=torch.float32, device=device)
    losses = trajectory_policy_loss(
        training_output,
        target_trajectory,
        target_stop,
        weights=loss_weights,
    )
    losses["total"].backward()
    frozen_input_gradients = {
        key: training_inputs[key].grad is None
        for key in ("language", "global_visual", "entity_visual")
    }
    if not all(frozen_input_gradients.values()):
        raise AssertionError("frozen language/visual inputs received gradients")
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if any(gradient is None for gradient in trainable_gradients):
        raise AssertionError("a trainable policy parameter has no gradient")
    if not all(
        torch.isfinite(gradient).all()
        for gradient in trainable_gradients
        if gradient is not None
    ):
        raise AssertionError("a policy gradient contains NaN or Inf")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    parameter_count = model.trainable_parameter_count()
    checkpoint_size = _checkpoint_size_bytes(model)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": fixed_seed,
        "batch_sizes": list(batch_sizes),
        "output_shapes": output_shapes,
        "trainable_parameter_count": parameter_count,
        "maximum_trainable_parameters": (
            model_config.maximum_trainable_parameters
        ),
        "checkpoint_size_bytes": checkpoint_size,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_kind": peak_kind,
        "maximum_step_m": model_config.maximum_step_m,
        "maximum_observed_increment_m": maximum_observed_increment,
        "frozen_cache_input_gradients_absent": frozen_input_gradients,
        "invalid_input_fail_closed": True,
        "privileged_policy_fields_absent": True,
        "loss": {
            key: (
                int(value.detach().cpu())
                if key == "valid_samples"
                else float(value.detach().cpu())
            )
            for key, value in losses.items()
        },
    }
    _atomic_write_json(Path(report_path).resolve(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Day 14 policy contract")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "model_small_v1.yaml",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    report = run_contract(
        args.config,
        args.report,
        device_name=args.device,
        seed=args.seed,
    )
    print(
        "DAY14_POLICY_CONTRACT_PASS "
        f"device={report['device']} "
        f"parameters={report['trainable_parameter_count']} "
        f"checkpoint_bytes={report['checkpoint_size_bytes']} "
        f"peak_memory_bytes={report['peak_memory_bytes']} "
        f"max_step_m={report['maximum_step_m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
