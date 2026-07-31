import sys, os

PROJECT = r"C:\Users\LIU\Documents\jetson_ws\asv_vla"
sys.path.insert(0, os.path.join(PROJECT, "src", "asv_vla"))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from pathlib import Path
from training.train import train_validation_suite
import argparse

args = argparse.Namespace(
    config=Path(r"training\config\train_sine_v1.yaml"),
    model_config=Path(r"training\config\model_small_v2.yaml"),
    features=Path(r"C:\Users\LIU\Documents\jetson_ws\pc_datasets\features_sine"),
    split=Path(r"C:\Users\LIU\Documents\jetson_ws\pc_datasets\registry\sine_group_split_v1.json"),
    instructions=Path(r"dataset\language\instructions.jsonl"),
    output_root=Path(r"C:\Users\LIU\Documents\jetson_ws\pc_datasets\checkpoints\sine_formation_v2"),
    git_sha="sine-formation-v2",
    device="cuda",
)
print(f"Git SHA: {args.git_sha}")
print(f"Output: {args.output_root}")
sys.exit(train_validation_suite(args))
