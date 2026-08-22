from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vla"))

from decision import build_entity_features, compute_entity_metrics


def test_policy_tensor_excludes_tracker_velocity_but_safety_metrics_keep_it() -> None:
    entity = SimpleNamespace(
        entity_id="target_red",
        color="red",
        is_target=True,
        visible=True,
        valid=True,
        relative_x=5.0,
        relative_y=0.0,
        relative_z=0.0,
        relative_velocity_x=-1.0,
        relative_velocity_y=0.25,
        relative_velocity_z=0.0,
    )

    metrics = compute_entity_metrics(entity)
    result = build_entity_features([entity])
    row = result.features[0]

    assert metrics.closing_speed_mps == 1.0
    np.testing.assert_allclose(row[[3, 4, 5, 9, 10, 11]], 0.0)
    assert row[0] > 0.0
    assert row[6] > 0.0
    assert row[12] == 1.0
