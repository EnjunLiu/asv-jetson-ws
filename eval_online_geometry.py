"""Evaluate the deployed policy on real online geometry (offline replay).

Feeds the entities captured during a headless closed-loop run into the
trained policy (geometry + language only; visual features are not
available offline) and checks the first-step direction against the red and
blue bearings.  This isolates whether the online selection failure is a
model problem (wrong direction even with correct geometry) or an input
problem (visual crops / ordering).

Usage:
    python eval_online_geometry.py --checkpoint <best.pt> \
        --entities <r14_ents.json> --language <embedding.npy> \
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--language", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
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
    frames = json.loads(args.entities.read_text(encoding="utf-8"))

    correct = 0
    total = 0
    toward_red = 0
    toward_blue = 0
    with torch.no_grad():
        for frame in frames:
            # Build the 16-slot geometry tensor in the same order as the
            # online task tensor: targets first (by distance), then normal.
            ents = [e for e in frame["entities"] if e["vis"]]
            ents.sort(key=lambda e: math.hypot(e["x"], e["y"]))
            geometry = np.zeros((16, 16), dtype=np.float32)
            mask = np.zeros((16,), dtype=bool)
            red_bearing = blue_bearing = None
            for slot, e in enumerate(ents[:16]):
                x, y = float(e["x"]), float(e["y"])
                # Normalise like task_entity_tensor: pos /20, vel /5.
                geometry[slot, 0] = x / 20.0
                geometry[slot, 1] = y / 20.0
                geometry[slot, 6] = math.hypot(x, y) / 20.0
                geometry[slot, 7] = math.sin(math.atan2(y, x))
                geometry[slot, 8] = math.cos(math.atan2(y, x))
                mask[slot] = True
                if e["id"] == "target_red":
                    red_bearing = _bearing_deg(x, y)
                elif e["id"] == "target_blue":
                    blue_bearing = _bearing_deg(x, y)
            if red_bearing is None or blue_bearing is None:
                continue
            item = {
                "language": language,
                "global_visual": torch.zeros(
                    1, 576, dtype=torch.float32, device=args.device
                ),
                "entity_visual": torch.zeros(
                    1, 16, 576, dtype=torch.float32, device=args.device
                ),
                "entity_geometry": torch.from_numpy(
                    geometry[None]
                ).to(args.device),
                "ego": torch.zeros(1, 2, dtype=torch.float32, device=args.device),
                "language_valid": torch.tensor([True], dtype=torch.bool),
                "global_visual_mask": torch.tensor([False], dtype=torch.bool),
                "entity_visual_mask": torch.zeros(
                    1, 16, dtype=torch.bool, device=args.device
                ),
                "entity_geometry_mask": torch.from_numpy(
                    mask[None]
                ).to(args.device),
                "ego_valid": torch.tensor([True], dtype=torch.bool),
            }
            out = model(**item)
            traj = out.trajectory[0].cpu().numpy()
            stop = float(out.stop_logit[0][0]) > 0.0
            dx, dy = float(traj[0, 0]), float(traj[0, 1])
            if math.hypot(dx, dy) < 1e-4:
                continue
            step = _bearing_deg(dx, dy)
            to_red = _angle_between_deg(step, red_bearing)
            to_blue = _angle_between_deg(step, blue_bearing)
            toward_red += int(to_red < to_blue)
            toward_blue += int(to_blue < to_red)
            correct += int(to_red <= min(to_blue, 45.0))
            total += 1

    if total == 0:
        print("EVAL_FAIL: no evaluable frames")
        return 1
    print(
        f"ONLINE_GEOM_EVAL frames={total} "
        f"toward_red={toward_red} toward_blue={toward_blue} "
        f"selection_rate={correct / total:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
