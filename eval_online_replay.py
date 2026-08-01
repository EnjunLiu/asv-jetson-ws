"""Replay captured online inputs (visual + task + language) through the
deployed policy exactly as the online node builds them.

Compares the policy's first-step direction against the red/blue bearings
from the same frame, and also tests cross-frame mismatches (visual frame N
with task frame N+k) to check whether the online selection failure comes
from input desynchronisation.

Usage:
    python eval_online_replay.py --checkpoint <best.pt> \
        --features <r16_feats.json> --language <embedding.npy> \
        --model-config <model_small_v2.yaml>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def _bearing_deg(x: float, y: float) -> float:
    return math.degrees(math.atan2(y, x))


def _angle_between_deg(a_deg: float, b_deg: float) -> float:
    delta = abs(a_deg - b_deg) % 360.0
    return min(delta, 360.0 - delta)


def _frame_geometry(task_frame: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the 16x16 geometry tensor from captured task features."""
    ids = list(task_frame["ids"])
    feat = np.asarray(task_frame["feat"], dtype=np.float32)
    mask = np.asarray(task_frame["mask"], dtype=bool)
    n = len(ids)
    geometry = np.zeros((16, 16), dtype=np.float32)
    geom_mask = np.zeros((16,), dtype=bool)
    bearings: dict[str, float] = {}
    for slot in range(min(n, 16)):
        if not mask[slot] or not ids[slot]:
            continue
        geometry[slot, :] = feat[slot * 16:(slot + 1) * 16]
        geom_mask[slot] = True
        if ids[slot] in ("target_red", "target_blue"):
            x = float(feat[slot * 16 + 0]) * 20.0
            y = float(feat[slot * 16 + 1]) * 20.0
            bearings[ids[slot]] = _bearing_deg(x, y)
    return geometry, geom_mask, bearings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--language", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    from training.model import SmallTrajectoryPolicy, SmallPolicyConfig

    import yaml

    model_cfg = SmallPolicyConfig.from_mapping(
        yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    )
    model = SmallTrajectoryPolicy(model_cfg).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    language = torch.from_numpy(
        np.load(args.language, allow_pickle=False).reshape(1, 256)
    ).float().to(args.device)

    data = json.loads(args.features.read_text(encoding="utf-8"))
    visuals = data["visual"]
    tasks = data["task"]
    n = min(len(visuals), len(tasks))

    correct = 0
    total = 0
    toward_red = 0
    toward_blue = 0
    stops = 0
    with torch.no_grad():
        for i in range(n):
            vis = visuals[i]
            t = tasks[min(i + args.offset, len(tasks) - 1)]
            if not vis["data"] or not t["feat"]:
                continue
            geometry, geom_mask, bearings = _frame_geometry(t)
            if "target_red" not in bearings or "target_blue" not in bearings:
                continue
            global_visual = np.asarray(vis["data"][:576], dtype=np.float32)
            entity_visual = np.zeros((16, 576), dtype=np.float32)
            evis_mask = np.zeros((16,), dtype=bool)
            tokens = int(vis["tokens"])
            vis_mask = list(vis["mask"])
            for slot in range(min(tokens - 1, 16)):
                base = 576 + slot * 576
                entity_visual[slot] = vis["data"][base:base + 576]
                evis_mask[slot] = bool(vis_mask[slot + 1])
            item = {
                "language": language,
                "global_visual": torch.from_numpy(
                    global_visual[None]
                ).to(args.device),
                "entity_visual": torch.from_numpy(
                    entity_visual[None]
                ).to(args.device),
                "entity_geometry": torch.from_numpy(
                    geometry[None]
                ).to(args.device),
                "ego": torch.zeros(1, 2, dtype=torch.float32, device=args.device),
                "language_valid": torch.tensor([True], dtype=torch.bool),
                "global_visual_mask": torch.tensor(
                    [bool(vis_mask[0])], dtype=torch.bool
                ),
                "entity_visual_mask": torch.from_numpy(
                    evis_mask[None]
                ).to(args.device),
                "entity_geometry_mask": torch.from_numpy(
                    geom_mask[None]
                ).to(args.device),
                "ego_valid": torch.tensor([True], dtype=torch.bool),
            }
            out = model(**item)
            traj = out.trajectory[0].cpu().numpy()
            stop = float(out.stop_logit[0][0]) > 0.0
            dx, dy = float(traj[0, 0]), float(traj[0, 1])
            if math.hypot(dx, dy) < 1e-4:
                stops += 1
                continue
            step = _bearing_deg(dx, dy)
            to_red = _angle_between_deg(step, bearings["target_red"])
            to_blue = _angle_between_deg(step, bearings["target_blue"])
            toward_red += int(to_red < to_blue)
            toward_blue += int(to_blue < to_red)
            correct += int(to_red <= min(to_blue, 45.0))
            total += 1

    if total == 0:
        print(f"REPLAY_FAIL: no evaluable frames (stops={stops})")
        return 1
    print(
        f"REPLAY offset={args.offset} frames={total} stops={stops} "
        f"toward_red={toward_red} toward_blue={toward_blue} "
        f"selection_rate={correct / total:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
