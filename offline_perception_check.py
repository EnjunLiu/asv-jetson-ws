"""Offline perception check: detection rate for center vs lateral targets.

Loads the deployed image_entity_color_calibrated_v1 model and runs it over
held frames from the near S2 collection, splitting by the UE-truth target
lateral offset (|y| <= 0.5 m center, y <= -1.5 m left) to check whether the
model detects the target across the scene positions used online.
"""

import glob
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = r"C:\Users\LIU\Documents\jetson_ws\asv_vla"
import sys
sys.path.insert(0, os.path.join(PROJECT, "src", "asv_vla"))

from asv_vla.image_entity_perception import ImageEntityModel, parse_task_instruction


def main() -> int:
    frames = sorted(
        glob.glob(
            r"C:/Users/LIU/Documents/jetson_ws/pc_datasets/"
            r"extracted_sine_near/artifacts/day8_episode/*/frames/*.json"
        )
    )
    random.seed(7)
    sample = random.sample(frames, min(300, len(frames)))
    model = ImageEntityModel.load(
        r"C:\Users\LIU\Documents\jetson_ws\pc_datasets\models"
        r"\image_entity_color_calibrated_v1.npz"
    )
    task = parse_task_instruction("跟随红色目标船，保持3米距离")

    stats = {"center": [0, 0], "left": [0, 0], "right": [0, 0]}
    for path in sample:
        try:
            record = json.load(open(path))
        except Exception:
            continue
        red = None
        for item in record.get("entities", {}).get("items", []):
            if item.get("entity_id") == "target_red" and item.get("visible"):
                red = item["relative_position_m"]
        if red is None:
            continue
        y = red[1]
        if abs(y) <= 0.5:
            bucket = "center"
        elif y <= -1.5:
            bucket = "left"
        elif y >= 1.5:
            bucket = "right"
        else:
            continue
        image_path = Path(path).parent.parent / record["camera"]["image_path"]
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        predictions = model.predict(image, task, device="cuda")
        detected = any(p.entity_id == "target_red" and p.visible for p in predictions)
        stats[bucket][0] += 1
        stats[bucket][1] += int(detected)

    for bucket, (total, detected) in stats.items():
        rate = detected / total if total else 0.0
        print(f"{bucket:8s} detected {detected}/{total} = {rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
