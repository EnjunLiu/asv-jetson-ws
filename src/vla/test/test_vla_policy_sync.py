"""Pure tests for the structured-entity policy frame cache."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import math
import time


POLICY = Path(__file__).resolve().parents[1] / "vla" / "vla_policy_node.py"


def _load_sync_cache():
    tree = ast.parse(POLICY.read_text(encoding="utf-8"), filename=str(POLICY))
    wanted = {"FrameKey", "_SyncEntry", "FrameSyncCache"}
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


def test_entity_frame_is_retrieved_by_identity() -> None:
    cache = FrameSyncCache(cache_size=4, ttl_sec=10.0)
    message = _feature("run-a", 7, 2)
    key, switched = cache.put_entities(message, received_at=0.0)
    assert switched is False
    assert cache.entity_for(key) is message
    assert cache.active_run == ("run-a", 7)


def test_scene_switch_clears_previous_structured_frames() -> None:
    cache = FrameSyncCache(cache_size=4, ttl_sec=10.0)
    old_key, _ = cache.put_entities(_feature("run-a", 1, 3), received_at=0.0)
    new_key, switched = cache.put_entities(_feature("run-b", 2, 0), received_at=0.1)
    assert switched is True
    assert cache.active_run == ("run-b", 2)
    assert cache.entity_for(old_key) is None
    assert cache.entity_for(new_key) is not None


def test_expiry_is_fail_closed() -> None:
    cache = FrameSyncCache(cache_size=4, ttl_sec=1.0)
    key, _ = cache.put_entities(_feature("run-a", 1, 9), received_at=0.0)
    assert cache.expire(now=0.5) == 0
    assert cache.entity_for(key) is not None
    assert cache.expire(now=1.1) == 1
    assert cache.entity_for(key) is None


def test_cache_size_is_bounded() -> None:
    cache = FrameSyncCache(cache_size=2, ttl_sec=10.0)
    for index in range(5):
        cache.put_entities(_feature("run-a", 1, index), received_at=float(index))
    assert cache.entity_size == 2
