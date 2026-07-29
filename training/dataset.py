"""Day 14 loader for immutable Day 13 feature caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from training.feature_cache import (
    FRAME_SHARD_NAME,
    LANGUAGE_FILE_NAME,
    validate_feature_cache,
)


POLICY_INPUT_KEYS = frozenset(
    {
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
    }
)
FORBIDDEN_POLICY_FIELDS = frozenset(
    {
        "action",
        "target_attribute",
        "distance_bucket",
        "task_label",
        "color",
        "color_red",
        "color_blue",
        "entity_ids",
        "expert_selected_entity_ids",
    }
)
METADATA_KEYS = frozenset(
    {"run_id", "frame_key", "sample_id", "instruction_id"}
)


@dataclass(frozen=True)
class _SampleRef:
    cache_index: int
    sample_row: int


@dataclass(frozen=True)
class _RunCache:
    run_id: str
    instruction_ids: np.ndarray
    language_splits: np.ndarray
    language: np.ndarray
    frame_indices: np.ndarray
    frame_keys: np.ndarray
    global_visual: np.ndarray
    global_visual_mask: np.ndarray
    entity_visual: np.ndarray
    entity_visual_mask: np.ndarray
    entity_geometry: np.ndarray
    entity_geometry_mask: np.ndarray
    ego: np.ndarray
    ego_valid: np.ndarray
    policy_input_valid: np.ndarray
    sample_ids: np.ndarray
    sample_frame_rows: np.ndarray
    sample_instruction_rows: np.ndarray
    target_trajectories: np.ndarray
    target_safe_stop: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_split_assignments(path: str | Path) -> dict[str, str]:
    manifest = _read_json(Path(path).expanduser().resolve())
    assignments = manifest.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("split manifest assignments must be an object")
    normalized: dict[str, str] = {}
    for run_id, split in assignments.items():
        current_id = str(run_id).strip()
        current_split = str(split).strip().casefold()
        if not current_id:
            raise ValueError("split manifest contains an empty Run ID")
        if current_split not in {"train", "validation", "test"}:
            raise ValueError(
                f"run_id={current_id}: invalid split {current_split!r}"
            )
        normalized[current_id] = current_split
    return normalized


def discover_feature_caches(root: str | Path) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"feature root does not exist: {base}")
    caches = sorted(
        path.parent
        for path in base.glob("*/manifest.json")
        if (path.parent / LANGUAGE_FILE_NAME).is_file()
        and (path.parent / FRAME_SHARD_NAME).is_file()
    )
    if not caches:
        raise ValueError(f"no feature caches found under {base}")
    return caches


def _validate_frame_key(value: str, run_id: str) -> None:
    parts = value.rsplit(":", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"incomplete frame key: {value!r}")
    if parts[0] != run_id:
        raise ValueError(
            f"frame key Run ID {parts[0]!r} does not match {run_id!r}"
        )
    try:
        scene_seed, frame_index, stamp_us = map(int, parts[1:])
    except ValueError as exc:
        raise ValueError(f"frame key contains a non-integer field: {value!r}") from exc
    if scene_seed < 0 or frame_index < 0 or stamp_us < 0:
        raise ValueError(f"frame key contains a negative field: {value!r}")


def _load_cache(path: Path) -> _RunCache:
    validate_feature_cache(path)
    manifest = _read_json(path / "manifest.json")
    run_id = str(manifest.get("run_id", "")).strip()
    if not run_id:
        raise ValueError(f"{path}: manifest has no Run ID")
    try:
        with np.load(path / LANGUAGE_FILE_NAME, allow_pickle=False) as source:
            instruction_ids = np.asarray(source["instruction_ids"]).copy()
            language_splits = np.asarray(source["language_splits"]).copy()
            language = np.asarray(source["embeddings"], dtype=np.float32).copy()
        with np.load(path / FRAME_SHARD_NAME, allow_pickle=False) as source:
            arrays = {
                name: np.asarray(source[name]).copy()
                for name in (
                    "frame_indices",
                    "frame_keys",
                    "global_visual",
                    "global_visual_mask",
                    "entity_visual",
                    "entity_visual_mask",
                    "entity_features",
                    "entity_mask",
                    "ego",
                    "ego_valid",
                    "policy_input_valid",
                    "sample_ids",
                    "sample_frame_rows",
                    "sample_instruction_rows",
                    "expert_trajectories",
                    "expert_safe_stop",
                )
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"cannot load feature cache {path}: {exc}") from exc

    if np.any(arrays["entity_features"][:, :, 14:16] != 0.0):
        raise ValueError(f"{path}: privileged entity color columns are non-zero")
    for frame_key in arrays["frame_keys"]:
        _validate_frame_key(str(frame_key), run_id)
    if len(language_splits) != len(instruction_ids):
        raise ValueError(f"{path}: language split and ID counts differ")
    if any(
        str(split).casefold() not in {"train", "validation", "test"}
        for split in language_splits
    ):
        raise ValueError(f"{path}: invalid language template split")

    return _RunCache(
        run_id=run_id,
        instruction_ids=instruction_ids,
        language_splits=language_splits,
        language=language,
        frame_indices=arrays["frame_indices"],
        frame_keys=arrays["frame_keys"],
        global_visual=np.asarray(arrays["global_visual"], dtype=np.float32),
        global_visual_mask=np.asarray(
            arrays["global_visual_mask"], dtype=np.bool_
        ),
        entity_visual=np.asarray(arrays["entity_visual"], dtype=np.float32),
        entity_visual_mask=np.asarray(
            arrays["entity_visual_mask"], dtype=np.bool_
        ),
        entity_geometry=np.asarray(
            arrays["entity_features"], dtype=np.float32
        ),
        entity_geometry_mask=np.asarray(arrays["entity_mask"], dtype=np.bool_),
        ego=np.asarray(arrays["ego"], dtype=np.float32),
        ego_valid=np.asarray(arrays["ego_valid"], dtype=np.bool_),
        policy_input_valid=np.asarray(
            arrays["policy_input_valid"], dtype=np.bool_
        ),
        sample_ids=arrays["sample_ids"],
        sample_frame_rows=np.asarray(
            arrays["sample_frame_rows"], dtype=np.int64
        ),
        sample_instruction_rows=np.asarray(
            arrays["sample_instruction_rows"], dtype=np.int64
        ),
        target_trajectories=np.asarray(
            arrays["expert_trajectories"], dtype=np.float32
        ),
        target_safe_stop=np.asarray(
            arrays["expert_safe_stop"], dtype=np.bool_
        ),
    )


class FrozenFeatureDataset(Dataset[dict[str, Tensor | str]]):
    """Expose only permitted policy inputs, expert targets, and audit metadata."""

    def __init__(
        self,
        cache_dirs: Sequence[str | Path],
        *,
        selected_split: str | None = None,
        split_assignments: Mapping[str, str] | None = None,
        allowed_language_splits: Iterable[str] | None = None,
        frame_stride: int = 1,
        require_valid: bool = True,
    ) -> None:
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        normalized_split = (
            str(selected_split).strip().casefold()
            if selected_split is not None
            else None
        )
        if normalized_split not in {None, "train", "validation", "test"}:
            raise ValueError(f"invalid selected_split={selected_split!r}")
        if normalized_split is not None and split_assignments is None:
            raise ValueError(
                "selected_split requires explicit Run-level split assignments"
            )
        allowed = (
            {str(value).strip().casefold() for value in allowed_language_splits}
            if allowed_language_splits is not None
            else None
        )
        if allowed is not None and (
            not allowed or not allowed <= {"train", "validation", "test"}
        ):
            raise ValueError(f"invalid allowed_language_splits={sorted(allowed)}")

        self._caches: list[_RunCache] = []
        self._samples: list[_SampleRef] = []
        seen_run_ids: set[str] = set()
        for candidate in sorted(Path(path).resolve() for path in cache_dirs):
            cache = _load_cache(candidate)
            if cache.run_id in seen_run_ids:
                raise ValueError(f"duplicate feature cache Run ID: {cache.run_id}")
            seen_run_ids.add(cache.run_id)
            if split_assignments is not None:
                assigned = split_assignments.get(cache.run_id)
                if assigned is None:
                    raise ValueError(
                        f"Run ID {cache.run_id} has no split assignment"
                    )
                if normalized_split is not None and assigned != normalized_split:
                    continue

            cache_index = len(self._caches)
            self._caches.append(cache)
            first_frame_index = int(np.min(cache.frame_indices))
            for sample_row, (frame_row, instruction_row) in enumerate(
                zip(
                    cache.sample_frame_rows,
                    cache.sample_instruction_rows,
                )
            ):
                frame_row_int = int(frame_row)
                instruction_row_int = int(instruction_row)
                if require_valid and not bool(
                    cache.policy_input_valid[frame_row_int]
                ):
                    continue
                if (
                    int(cache.frame_indices[frame_row_int]) - first_frame_index
                ) % frame_stride:
                    continue
                language_split = str(
                    cache.language_splits[instruction_row_int]
                ).casefold()
                if allowed is not None and language_split not in allowed:
                    continue
                self._samples.append(_SampleRef(cache_index, sample_row))

        if not self._caches:
            raise ValueError("no feature cache matches the selected Run split")
        if not self._samples:
            raise ValueError("no feature-cache samples match the loader filters")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        reference = self._samples[index]
        cache = self._caches[reference.cache_index]
        sample_row = reference.sample_row
        frame_row = int(cache.sample_frame_rows[sample_row])
        instruction_row = int(cache.sample_instruction_rows[sample_row])
        return {
            "language": torch.from_numpy(
                cache.language[instruction_row].copy()
            ),
            "global_visual": torch.from_numpy(
                cache.global_visual[frame_row].copy()
            ),
            "entity_visual": torch.from_numpy(
                cache.entity_visual[frame_row].copy()
            ),
            "entity_geometry": torch.from_numpy(
                cache.entity_geometry[frame_row].copy()
            ),
            "ego": torch.from_numpy(cache.ego[frame_row].copy()),
            "language_valid": torch.tensor(True, dtype=torch.bool),
            "global_visual_mask": torch.tensor(
                bool(cache.global_visual_mask[frame_row]), dtype=torch.bool
            ),
            "entity_visual_mask": torch.from_numpy(
                cache.entity_visual_mask[frame_row].copy()
            ),
            "entity_geometry_mask": torch.from_numpy(
                cache.entity_geometry_mask[frame_row].copy()
            ),
            "ego_valid": torch.tensor(
                bool(cache.ego_valid[frame_row]), dtype=torch.bool
            ),
            "policy_input_valid": torch.tensor(
                bool(cache.policy_input_valid[frame_row]), dtype=torch.bool
            ),
            "target_trajectory": torch.from_numpy(
                cache.target_trajectories[sample_row].copy()
            ),
            "target_stop": torch.tensor(
                [float(cache.target_safe_stop[sample_row])],
                dtype=torch.float32,
            ),
            "run_id": cache.run_id,
            "frame_key": str(cache.frame_keys[frame_row]),
            "sample_id": str(cache.sample_ids[sample_row]),
            "instruction_id": str(cache.instruction_ids[instruction_row]),
        }


def policy_inputs_from_batch(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    missing = POLICY_INPUT_KEYS - set(batch)
    if missing:
        raise ValueError(f"policy batch is missing keys: {sorted(missing)}")
    forbidden = FORBIDDEN_POLICY_FIELDS & set(batch)
    if forbidden:
        raise ValueError(f"policy batch contains privileged fields: {sorted(forbidden)}")
    return {key: batch[key] for key in POLICY_INPUT_KEYS}
