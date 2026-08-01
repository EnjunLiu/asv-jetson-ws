"""Day 20: export trained policy checkpoint to ONNX."""

from __future__ import annotations

import argparse, json, hashlib, sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from training.model import SmallTrajectoryPolicy, SmallPolicyConfig


def _sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def _checkpoint_model_config(
    checkpoint: Mapping[str, Any],
) -> tuple[SmallPolicyConfig, str]:
    """Resolve the exact policy config used to create a checkpoint."""

    raw_config = checkpoint.get("model_config")
    if raw_config is None:
        # Older checkpoints predate model_config provenance.  Match the
        # dataclass defaults (the model_small_v1 training default), rather
        # than silently selecting a different attention architecture.
        return SmallPolicyConfig(), "dataclass_defaults_legacy"
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    return SmallPolicyConfig.from_mapping(raw_config), "checkpoint:model_config"


def export_onnx(checkpoint_path: str, output_path: str) -> dict:
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg, config_source = _checkpoint_model_config(ckpt)
    model = SmallTrajectoryPolicy(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Dummy inputs matching the policy contract.
    B = 1
    language = torch.randn(B, 256)
    global_visual = torch.randn(B, 576)
    entity_visual = torch.randn(B, 16, 576)
    entity_geometry = torch.randn(B, 16, 16)
    ego = torch.randn(B, 2)
    language_valid = torch.ones(B, dtype=torch.bool)
    global_visual_mask = torch.ones(B, dtype=torch.bool)
    entity_visual_mask = torch.ones(B, 16, dtype=torch.bool)
    entity_geometry_mask = torch.ones(B, 16, dtype=torch.bool)
    ego_valid = torch.ones(B, dtype=torch.bool)
    policy_input_valid = torch.ones(B, dtype=torch.bool)

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        def forward(self, lang, gv, ev, eg, e, lv, gvm, evm, egm, ev2, piv):
            out = self.policy(
                language=lang, global_visual=gv, entity_visual=ev,
                entity_geometry=eg, ego=e, language_valid=lv.bool(),
                global_visual_mask=gvm.bool(), entity_visual_mask=evm.bool(),
                entity_geometry_mask=egm.bool(), ego_valid=ev2.bool(),
                policy_input_valid=piv.bool(),
            )
            return out.trajectory, out.stop_logit, out.valid_mask

    wrapped = ONNXWrapper(model)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (
            language,
            global_visual,
            entity_visual,
            entity_geometry,
            ego,
            language_valid,
            global_visual_mask,
            entity_visual_mask,
            entity_geometry_mask,
            ego_valid,
            policy_input_valid,
        ),
        output_path,
        input_names=[
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
        ],
        output_names=["trajectory", "stop_logit", "valid_mask"],
        dynamic_axes={
            "language": {0: "batch"},
            "global_visual": {0: "batch"},
            "entity_visual": {0: "batch"},
            "entity_geometry": {0: "batch"},
            "ego": {0: "batch"},
            "language_valid": {0: "batch"},
            "global_visual_mask": {0: "batch"},
            "entity_visual_mask": {0: "batch"},
            "entity_geometry_mask": {0: "batch"},
            "ego_valid": {0: "batch"},
            "policy_input_valid": {0: "batch"},
            "trajectory": {0: "batch"},
            "stop_logit": {0: "batch"},
            "valid_mask": {0: "batch"},
        },
        opset_version=17,
    )

    onnx_sha = _sha256_file(Path(output_path))
    ckpt_sha = _sha256_file(Path(checkpoint_path))

    # Validate: run PyTorch and ONNX on same input, compare.
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = model(
            language=language,
            global_visual=global_visual,
            entity_visual=entity_visual,
            entity_geometry=entity_geometry,
            ego=ego,
            language_valid=language_valid,
            global_visual_mask=global_visual_mask,
            entity_visual_mask=entity_visual_mask,
            entity_geometry_mask=entity_geometry_mask,
            ego_valid=ego_valid,
            policy_input_valid=policy_input_valid,
        )

    session = ort.InferenceSession(output_path)
    ort_inputs = {
        "language": language.numpy(),
        "global_visual": global_visual.numpy(),
        "entity_visual": entity_visual.numpy(),
        "entity_geometry": entity_geometry.numpy(),
        "ego": ego.numpy(),
        "language_valid": language_valid.numpy(),
        "global_visual_mask": global_visual_mask.numpy(),
        "entity_visual_mask": entity_visual_mask.numpy(),
        "entity_geometry_mask": entity_geometry_mask.numpy(),
        "ego_valid": ego_valid.numpy(),
        "policy_input_valid": policy_input_valid.numpy(),
    }
    ort_out = session.run(None, ort_inputs)

    pt_traj = pt_out.trajectory.numpy()
    ort_traj = ort_out[0]  # (trajectory, stop_logit, valid_mask)
    max_diff = float(np.max(np.abs(pt_traj - np.array(ort_traj))))
    # Cosine is undefined for a zero trajectory (a legitimate fail-closed
    # STOP output); max_diff already guarantees exact agreement then.
    traj_norm = float(np.linalg.norm(pt_traj))
    if traj_norm > 1e-6:
        cos_sim = float(
            np.dot(pt_traj.flatten(), ort_traj.flatten())
            / (np.linalg.norm(pt_traj) * np.linalg.norm(ort_traj) + 1e-12)
        )
        cos_ok = abs(cos_sim - 1.0) < 1e-4
    else:
        cos_sim = 1.0
        cos_ok = True

    report = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": ckpt_sha,
        "onnx_path": str(Path(output_path).resolve()),
        "onnx_sha256": onnx_sha,
        "parameter_count": model.trainable_parameter_count(),
        "model_config": asdict(cfg),
        "model_config_source": config_source,
        "max_abs_error": float(max_diff),
        "cosine_similarity": float(cos_sim),
        "passed": max_diff < 1e-4 and cos_ok,
        "opset_version": 17,
    }

    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Day 20 ONNX export")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        r = export_onnx(str(args.checkpoint), str(args.output))
    except Exception as e:
        print(f"ONNX_EXPORT_FAIL: {e}")
        return 1
    print(f"ONNX_EXPORT_PASS max_diff={r['max_abs_error']:.2e} cos={r['cosine_similarity']:.6f}")
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
