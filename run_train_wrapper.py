import sys, os
sys.path.insert(0, r'C:\Users\LIU\Documents\jetson_ws\day11_kinematic_work\src\asv_vla')
sys.path.insert(0, r'C:\Users\LIU\Documents\jetson_ws\day11_kinematic_work')
os.chdir(r'C:\Users\LIU\Documents\jetson_ws\day11_kinematic_work')

from pathlib import Path
from training.train import train_validation_suite
import argparse

args = argparse.Namespace(
    config=Path(r'training\config\train_30_v8_strong_pairwise.yaml'),  # pairwise=0.50 + cross-run loader
    model_config=Path(r'training\config\model_small_v2.yaml'),
    features=Path(r'C:\Users\LIU\Documents\jetson_ws\pc_datasets\features_pc_eb832f3'),
    split=Path(r'C:\Users\LIU\Documents\jetson_ws\pc_datasets\registry\group_split_30_v1.json'),
    instructions=Path(r'dataset\language\instructions.jsonl'),
    output_root=Path(r'C:\Users\LIU\Documents\jetson_ws\pc_datasets\checkpoints\day21_label_fix_v1'),
    git_sha="local-label-fix",
    device='cuda',
)
print(f"Git SHA: {args.git_sha}")
print(f"Output: {args.output_root}")
sys.exit(train_validation_suite(args))
