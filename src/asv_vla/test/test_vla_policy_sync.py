"""Pure unit tests for the VLA policy frame synchronizer.

The ROS runtime is not required by this test suite.  Extracting the small
``FrameSyncCache`` definition from the policy node keeps these tests focused
on exact-key matching, cache bounds, and fail-closed expiry.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import math
import time


POLICY = Path(__file__).resolve().parents[1] / "asv_vla" / "vla_policy_node.py"


def _load_sync_cache():
    tree = ast.parse(POLICY.read_text(encoding="utf-8"), filename=str(POLICY))
    wanted = {
        "FrameKey",
        "_SyncEntry",
        "FrameSyncCache",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name in wanted)
        or (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "FrameKey")
    ]
    namespace = {
        "OrderedDict": OrderedDict,
        "dataclass": dataclass,
        "math": math,
        "time": time,
        "Any": object,
        "SYNC_CACHE_SIZE": 256,
        "SYNC_CACHE_TTL_SEC": 5.0,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(POLICY), "exec"), namespace)
    return namespace["FrameSyncCache"]


FrameSyncCache = _load_sync_cache()


def _feature(run_id: str, scene_seed: int, frame_index: int):
    return SimpleNamespace(
        run_id=run_id,
        scene_seed=scene_seed,
        frame_index=frame_index,
    )


def test_exact_key_match_does_not_use_latest_mismatched_frame() -> None:
    cache = FrameSyncCache(cache_size=4, ttl_sec=10.0)
    visual_1 = _feature("run-a", 7, 1)
    entities_2 = _feature("run-a", 7, 2)
    visual_2 = _feature("run-a", 7, 2)

    cache.put_visual(visual_1, received_at=0.0)
    key_2, switched = cache.put_entities(entities_2, received_at=0.1)
    assert switched is False
    pair, status = cache.match(key_2, now=0.1)
    assert pair is None
    assert status == "NO_MATCH"

    cache.put_visual(visual_2, received_at=0.2)
    pair, status = cache.match(key_2, now=0.2)
    assert status == "MATCH"
    assert pair == (visual_2, entities_2)


def test_run_switch_clears_both_modality_caches() -> None:
    cache = FrameSyncCache(cache_size=4, ttl_sec=10.0)
    visual_a = _feature("run-a", 1, 3)
    entities_a = _feature("run-a", 1, 3)
    entities_b = _feature("run-b", 2, 0)

    key_a, _ = cache.put_visual(visual_a, received_at=0.0)
    cache.put_entities(entities_a, received_at=0.0)
    key_b, switched = cache.put_entities(entities_b, received_at=0.1)

    assert switched is True
    assert cache.active_run == ("run-b", 2)
    assert cache.visual_size == 0
    assert cache.entity_size == 1
    assert cache.match(key_a, now=0.1) == (None, "RUN_MISMATCH")
    assert cache.match(key_b, now=0.1)[1] == "NO_MATCH"


def test_missing_or_expired_counterpart_is_fail_closed() -> None:
    cache = FrameSyncCache(cache_size=4, ttl_sec=1.0)
    entities = _feature("run-a", 1, 9)
    key, _ = cache.put_entities(entities, received_at=0.0)

    pair, status = cache.match(key, now=0.5)
    assert pair is None
    assert status == "NO_MATCH"

    visual = _feature("run-a", 1, 9)
    cache.put_visual(visual, received_at=0.0)
    pair, status = cache.match(key, now=1.1)
    assert pair is None
    assert status == "STALE"
    assert cache.visual_size == 0
    assert cache.entity_size == 0


def test_cache_size_is_bounded() -> None:
    cache = FrameSyncCache(cache_size=2, ttl_sec=10.0)
    for index in range(5):
        cache.put_visual(_feature("run-a", 1, index), received_at=float(index))
    assert cache.visual_size == 2
