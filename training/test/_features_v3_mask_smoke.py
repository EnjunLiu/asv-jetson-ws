from pathlib import Path

import numpy as np

from training.dataset import (
    FrozenFeatureDataset,
    discover_feature_caches,
    load_split_assignments,
)


root = Path(r"C:\Temp\asv_vla_retrain_20260805\features_v3")
caches = discover_feature_caches(root)
split = load_split_assignments(
    r"C:\Temp\asv_vla_retrain_20260805\group_split_v2.json"
)
dataset = FrozenFeatureDataset(
    caches,
    split_assignments=split,
    allowed_language_splits={"train", "validation", "test"},
    require_valid=True,
)
counts = np.fromiter(
    (int(dataset[index]["entity_geometry_mask"].sum()) for index in range(len(dataset))),
    dtype=np.int8,
)
invalid = sum(
    not bool(dataset[index]["policy_input_valid"])
    for index in range(len(dataset))
)
stops = sum(
    bool(dataset[index]["target_stop"].item())
    for index in range(len(dataset))
)
print(
    "FEATURES_V3_MASK_SMOKE "
    f"samples={len(dataset)} caches={len(caches)} "
    f"active_one={int((counts == 1).sum())} "
    f"active_zero={int((counts == 0).sum())} "
    f"invalid={invalid} stops={stops} "
    f"min={int(counts.min())} max={int(counts.max())}"
)
