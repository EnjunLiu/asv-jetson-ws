"""Day 13 frozen multimodal feature cache.

The cache is deliberately split into frame-level perception and sample-level
supervision.  A camera frame is encoded once, while each compatible language
instruction keeps its own expert trajectory row.  This avoids duplicating the
same 576-D visual tensors roughly ninety times per frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import time
from typing import Any, Iterable

import numpy as np

from asv_vla.episode import load_episode_records, write_json_atomic
from asv_vla.frame_record import read_frame_record
from asv_vla.language_encoder import USVLanguageEncoder
from asv_vla.language_intervention_dataset import read_jsonl
from asv_vla.task_entity_tensor import (
    FEATURE_DIM as ENTITY_FEATURE_DIM,
    MAX_ENTITIES,
    EntityTensorResult,
    build_entity_tensor,
)
from asv_vla.trajectory_contract import ACTION_DIM, HORIZON
from asv_vla.visual_encoder import (
    BACKBONE_ID,
    FEATURE_DIM as VISUAL_FEATURE_DIM,
    CameraProfile,
    FrozenMobileNetEncoder,
    InvalidImageError,
    TargetProjectionError,
    VisualEncoderError,
    decode_camera_image,
    make_target_crop,
)


FEATURE_CACHE_SCHEMA_VERSION = "feature_cache_v1"
PREPROCESS_VERSION = "day13_camera_entity_crop_v1"
LANGUAGE_FEATURE_DIM = 256
COLOR_PRIVILEGE_COLUMNS = (14, 15)
FRAME_SHARD_NAME = "frames_000.npz"
LANGUAGE_FILE_NAME = "language.npz"
QUALITY_FILE_NAME = "quality_report.json"
_SHA256_LENGTH = 64


class FeatureCacheError(ValueError):
    """Raised when a feature cache cannot be built or validated."""


class FeatureCacheMiss(FeatureCacheError):
    """Raised when an existing cache was built from a different cache key."""


@dataclass(frozen=True)
class ModelFingerprint:
    model_id: str
    weights_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        _validate_sha256(self.weights_sha256, "weights_sha256")


@dataclass(frozen=True)
class PolicyEntityTensor:
    features: np.ndarray
    mask: np.ndarray
    entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrameVisualFeatures:
    global_token: np.ndarray
    global_valid: bool
    entity_tokens: np.ndarray
    entity_visual_mask: np.ndarray
    detail: str


def _validate_sha256(value: str, field: str) -> None:
    normalized = str(value).strip().casefold()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FeatureCacheError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def hash_weight_tree(path: str | Path) -> str:
    """Hash model weight files plus their relative names deterministically."""

    root = Path(path).expanduser().resolve()
    if root.is_file():
        return _sha256_file(root)
    if not root.is_dir():
        raise FeatureCacheError(f"model path does not exist: {root}")
    suffixes = {".bin", ".onnx", ".pt", ".pth", ".safetensors"}
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in suffixes
    )
    if not files:
        raise FeatureCacheError(f"no model weight files found under {root}")
    if len(files) == 1:
        return _sha256_file(files[0])
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def hash_torch_module_state(module: Any) -> str:
    """Hash a frozen torch module without relying on pickle serialization."""

    state_dict_method = getattr(module, "state_dict", None)
    if not callable(state_dict_method):
        raise FeatureCacheError("visual backbone does not expose state_dict()")
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict_method().items()):
        name_bytes = str(name).encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        descriptor = f"{array.dtype.str}:{array.shape}".encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entity_objects(record: dict[str, Any]) -> list[SimpleNamespace]:
    output: list[SimpleNamespace] = []
    for item in record["entities"]["items"]:
        position = item["relative_position_m"]
        velocity = item["relative_velocity_mps"]
        output.append(
            SimpleNamespace(
                entity_id=item["entity_id"],
                class_name=item["class_name"],
                color=item["color"],
                is_target=item["is_target"],
                visible=item["visible"],
                relative_x=position[0],
                relative_y=position[1],
                relative_z=position[2],
                relative_velocity_x=velocity[0],
                relative_velocity_y=velocity[1],
                relative_velocity_z=velocity[2],
                valid=item["valid"],
            )
        )
    return output


def build_policy_entity_tensor(
    entities: Iterable[Any],
) -> PolicyEntityTensor:
    """Build Day 7 geometry while removing the two privileged color fields."""

    result: EntityTensorResult = build_entity_tensor(entities)
    features = np.ascontiguousarray(result.features.copy(), dtype=np.float32)
    features[:, COLOR_PRIVILEGE_COLUMNS[0] : COLOR_PRIVILEGE_COLUMNS[1] + 1] = 0.0
    if np.any(features[:, 14:16] != 0.0):
        raise FeatureCacheError("privileged entity color fields were not removed")
    return PolicyEntityTensor(
        features=features,
        mask=np.ascontiguousarray(result.mask, dtype=np.bool_),
        entity_ids=result.entity_ids,
    )


def encode_frame_visual(
    record: dict[str, Any],
    episode_dir: str | Path,
    policy_entities: PolicyEntityTensor,
    visual_encoder: Any,
    *,
    profile: CameraProfile | None = None,
) -> FrameVisualFeatures:
    """Encode global and per-entity images with fail-closed image semantics."""

    camera_profile = profile or CameraProfile()
    zeros_global = np.zeros(VISUAL_FEATURE_DIM, dtype=np.float32)
    zeros_entities = np.zeros(
        (MAX_ENTITIES, VISUAL_FEATURE_DIM), dtype=np.float32
    )
    zeros_mask = np.zeros(MAX_ENTITIES, dtype=np.bool_)
    camera = record.get("camera", {})
    if not bool(record.get("valid")) or not bool(camera.get("valid")):
        return FrameVisualFeatures(
            zeros_global, False, zeros_entities, zeros_mask, "camera_invalid"
        )

    image_path = Path(episode_dir) / str(camera.get("image_path", ""))
    try:
        image = decode_camera_image(
            image_path.read_bytes(),
            str(camera.get("encoding", "")),
        )
    except (OSError, InvalidImageError) as exc:
        return FrameVisualFeatures(
            zeros_global,
            False,
            zeros_entities,
            zeros_mask,
            f"image_invalid:{type(exc).__name__}",
        )
    if image.size != (camera_profile.width, camera_profile.height):
        return FrameVisualFeatures(
            zeros_global,
            False,
            zeros_entities,
            zeros_mask,
            f"image_shape:{image.width}x{image.height}",
        )

    entity_by_id = {
        str(entity.entity_id): entity for entity in _entity_objects(record)
    }
    batch = [image]
    batch_slots: list[int] = []
    for slot, entity_id in enumerate(policy_entities.entity_ids):
        if not policy_entities.mask[slot] or not entity_id:
            continue
        entity = entity_by_id.get(entity_id)
        if entity is None:
            raise FeatureCacheError(
                f"entity tensor references missing entity_id={entity_id!r}"
            )
        try:
            crop, _ = make_target_crop(image, entity, camera_profile)
        except (TargetProjectionError, InvalidImageError):
            continue
        batch.append(crop)
        batch_slots.append(slot)

    try:
        encoded = np.asarray(
            visual_encoder.encode_images(batch), dtype=np.float32
        )
    except (VisualEncoderError, ValueError) as exc:
        raise FeatureCacheError(f"visual inference failed: {exc}") from exc
    expected_shape = (len(batch), VISUAL_FEATURE_DIM)
    if encoded.shape != expected_shape or not np.all(np.isfinite(encoded)):
        raise FeatureCacheError(
            f"visual encoder returned {encoded.shape}; expected {expected_shape}"
        )

    global_token = np.ascontiguousarray(encoded[0], dtype=np.float32)
    entity_tokens = zeros_entities
    entity_visual_mask = zeros_mask
    for batch_index, slot in enumerate(batch_slots, start=1):
        entity_tokens[slot] = encoded[batch_index]
        entity_visual_mask[slot] = True
    return FrameVisualFeatures(
        global_token=global_token,
        global_valid=True,
        entity_tokens=entity_tokens,
        entity_visual_mask=entity_visual_mask,
        detail="ok",
    )


def make_cache_key(
    *,
    source_frames: list[dict[str, Any]],
    language_model: ModelFingerprint,
    visual_model: ModelFingerprint,
    git_sha: str,
    preprocess_version: str = PREPROCESS_VERSION,
    feature_schema_version: str = FEATURE_CACHE_SCHEMA_VERSION,
) -> dict[str, Any]:
    if not source_frames:
        raise FeatureCacheError("source_frames must not be empty")
    if not str(git_sha).strip():
        raise FeatureCacheError("git_sha must not be empty")
    return {
        "source_frame_sha256": {
            str(item["frame_key"]): str(item["source_frame_sha256"])
            for item in source_frames
        },
        "image_sha256": {
            str(item["frame_key"]): str(item["image_sha256"])
            for item in source_frames
        },
        "language_model_id": language_model.model_id,
        "language_weights_sha256": language_model.weights_sha256,
        "visual_model_id": visual_model.model_id,
        "visual_weights_sha256": visual_model.weights_sha256,
        "preprocess_version": preprocess_version,
        "feature_schema_version": feature_schema_version,
        "git_sha": str(git_sha).strip(),
    }


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    try:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise FeatureCacheError(f"failed to write {path}: {exc}") from exc


def _frame_sources(
    records: list[dict[str, Any]],
    episode_dir: Path,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for record in records:
        index = int(record["frame_index"])
        frame_path = episode_dir / "frames" / f"{index:012d}.json"
        image_path = episode_dir / str(record["camera"]["image_path"])
        sources.append(
            {
                "frame_key": (
                    f"{record['run_id']}:{record['scene_seed']}:{index}:"
                    f"{record['stamp_us']}"
                ),
                "frame_index": index,
                "source_frame_sha256": _sha256_file(frame_path),
                "image_sha256": _sha256_file(image_path),
            }
        )
    return sources


def _load_supervision_samples(path: Path) -> list[dict[str, Any]]:
    samples = read_jsonl(path)
    if not samples:
        raise FeatureCacheError("supervision samples are empty")
    return samples


def encode_language_instructions(
    instructions: list[dict[str, Any]],
    language_encoder: Any,
) -> np.ndarray:
    """Encode every instruction exactly once with the frozen language model."""

    embeddings = []
    for instruction in instructions:
        embedding = np.asarray(
            language_encoder.encode(str(instruction.get("text", ""))),
            dtype=np.float32,
        ).reshape(-1)
        if embedding.shape != (LANGUAGE_FEATURE_DIM,):
            raise FeatureCacheError(
                f"language embedding shape {embedding.shape} is invalid"
            )
        if not np.all(np.isfinite(embedding)):
            raise FeatureCacheError("language embedding contains NaN or Inf")
        embeddings.append(embedding)
    if not embeddings:
        raise FeatureCacheError("instruction dataset is empty")
    return np.stack(embeddings).astype(np.float32, copy=False)


def _build_sample_arrays(
    samples: list[dict[str, Any]],
    *,
    run_id: str,
    frame_row_by_index: dict[int, int],
    instruction_index: dict[str, int],
    source_by_index: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    sample_ids: list[str] = []
    frame_rows: list[int] = []
    instruction_rows: list[int] = []
    trajectories: list[np.ndarray] = []
    safe_stop: list[bool] = []
    selected_entity_ids: list[str] = []
    for sample in samples:
        source = sample.get("source", {})
        if str(source.get("run_id")) != run_id:
            raise FeatureCacheError("supervision contains a different run_id")
        frame_index = int(source.get("frame_index"))
        if frame_index not in frame_row_by_index:
            raise FeatureCacheError(
                f"supervision references unknown frame_index={frame_index}"
            )
        expected_source = source_by_index[frame_index]
        if source.get("frame_record_sha256") != expected_source[
            "source_frame_sha256"
        ]:
            raise FeatureCacheError(
                f"frame SHA mismatch for frame_index={frame_index}"
            )
        if source.get("image_sha256") != expected_source["image_sha256"]:
            raise FeatureCacheError(
                f"image SHA mismatch for frame_index={frame_index}"
            )
        instruction_id = str(
            sample.get("instruction", {}).get("instruction_id", "")
        )
        if instruction_id not in instruction_index:
            raise FeatureCacheError(
                f"unknown instruction_id={instruction_id!r}"
            )
        trajectory = np.asarray(
            sample.get("expert", {}).get("delta_p_xy"), dtype=np.float32
        )
        if trajectory.shape != (HORIZON, ACTION_DIM):
            raise FeatureCacheError(
                f"sample {sample.get('sample_id')} trajectory shape "
                f"{trajectory.shape} is invalid"
            )
        if not np.all(np.isfinite(trajectory)):
            raise FeatureCacheError("expert trajectory contains NaN or Inf")
        sample_ids.append(str(sample["sample_id"]))
        frame_rows.append(frame_row_by_index[frame_index])
        instruction_rows.append(instruction_index[instruction_id])
        trajectories.append(trajectory)
        safe_stop.append(bool(sample.get("expert", {}).get("safe_stop")))
        selected_entity_ids.append(
            str(sample.get("expert", {}).get("selected_entity_id") or "")
        )

    return {
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "sample_frame_rows": np.asarray(frame_rows, dtype=np.int32),
        "sample_instruction_rows": np.asarray(
            instruction_rows, dtype=np.int16
        ),
        "expert_trajectories": np.stack(trajectories).astype(
            np.float32, copy=False
        ),
        "expert_safe_stop": np.asarray(safe_stop, dtype=np.bool_),
        "expert_selected_entity_ids": np.asarray(
            selected_entity_ids, dtype=np.str_
        ),
    }


def build_feature_cache(
    episode_dir: str | Path,
    supervision_dir: str | Path,
    instructions_path: str | Path,
    output_root: str | Path,
    *,
    language_encoder: Any | None,
    visual_encoder: Any,
    language_model: ModelFingerprint,
    visual_model: ModelFingerprint,
    git_sha: str,
    preprocess_version: str = PREPROCESS_VERSION,
    precomputed_language_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build or validate one Run's immutable feature cache."""

    episode = Path(episode_dir).resolve()
    supervision = Path(supervision_dir).resolve()
    instructions_source = Path(instructions_path).resolve()
    output_base = Path(output_root).resolve()
    records = load_episode_records(episode)
    if not records:
        raise FeatureCacheError("episode contains no frames")
    run_ids = {str(record["run_id"]) for record in records}
    if len(run_ids) != 1:
        raise FeatureCacheError("episode contains multiple run_ids")
    run_id = next(iter(run_ids))
    sources = _frame_sources(records, episode)
    cache_key = make_cache_key(
        source_frames=sources,
        language_model=language_model,
        visual_model=visual_model,
        git_sha=git_sha,
        preprocess_version=preprocess_version,
    )
    cache_key_sha256 = _json_digest(cache_key)
    output = output_base / run_id
    if output.exists():
        manifest_path = output / "manifest.json"
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FeatureCacheMiss(
                f"existing cache manifest is unreadable: {exc}"
            ) from exc
        if existing.get("cache_key_sha256") != cache_key_sha256:
            raise FeatureCacheMiss(
                "existing feature cache key differs; weights, preprocessing, "
                "source frames, image bytes, git SHA, or schema changed"
            )
        report = validate_feature_cache(output)
        return {
            "run_id": run_id,
            "output": str(output),
            "cached": True,
            **report,
        }

    instructions = read_jsonl(instructions_source)
    if not instructions:
        raise FeatureCacheError("instruction dataset is empty")
    instruction_ids = [str(item.get("instruction_id", "")).strip() for item in instructions]
    if any(not value for value in instruction_ids):
        raise FeatureCacheError("instruction dataset contains an empty ID")
    if len(instruction_ids) != len(set(instruction_ids)):
        raise FeatureCacheError("instruction dataset contains duplicate IDs")
    if precomputed_language_embeddings is None:
        if language_encoder is None:
            raise FeatureCacheError(
                "language_encoder is required without precomputed embeddings"
            )
        language_array = encode_language_instructions(
            instructions, language_encoder
        )
    else:
        language_array = np.ascontiguousarray(
            precomputed_language_embeddings, dtype=np.float32
        )
        expected_language_shape = (
            len(instructions),
            LANGUAGE_FEATURE_DIM,
        )
        if language_array.shape != expected_language_shape:
            raise FeatureCacheError(
                f"precomputed language shape {language_array.shape}; "
                f"expected {expected_language_shape}"
            )
        if not np.all(np.isfinite(language_array)):
            raise FeatureCacheError(
                "precomputed language embeddings contain NaN or Inf"
            )
    instruction_index = {
        instruction_id: index
        for index, instruction_id in enumerate(instruction_ids)
    }

    frame_indices: list[int] = []
    frame_stamps: list[int] = []
    frame_keys: list[str] = []
    frame_sha256: list[str] = []
    image_sha256: list[str] = []
    global_visual: list[np.ndarray] = []
    global_visual_mask: list[bool] = []
    entity_visual: list[np.ndarray] = []
    entity_visual_mask: list[np.ndarray] = []
    entity_features: list[np.ndarray] = []
    entity_mask: list[np.ndarray] = []
    entity_ids: list[tuple[str, ...]] = []
    ego: list[tuple[float, float]] = []
    ego_valid: list[bool] = []
    policy_input_valid: list[bool] = []
    visual_details: list[str] = []

    for record, source in zip(records, sources):
        entities = build_policy_entity_tensor(_entity_objects(record))
        visual = encode_frame_visual(
            record, episode, entities, visual_encoder
        )
        ego_block = record.get("ego", {})
        ego_values = (
            float(ego_block.get("surge_velocity_mps", 0.0)),
            float(ego_block.get("yaw_rate_radps", 0.0)),
        )
        current_ego_valid = bool(ego_block.get("valid")) and all(
            math.isfinite(value) for value in ego_values
        )
        frame_indices.append(int(record["frame_index"]))
        frame_stamps.append(int(record["stamp_us"]))
        frame_keys.append(str(source["frame_key"]))
        frame_sha256.append(str(source["source_frame_sha256"]))
        image_sha256.append(str(source["image_sha256"]))
        global_visual.append(visual.global_token)
        global_visual_mask.append(visual.global_valid)
        entity_visual.append(visual.entity_tokens)
        entity_visual_mask.append(visual.entity_visual_mask)
        entity_features.append(entities.features)
        entity_mask.append(entities.mask)
        entity_ids.append(entities.entity_ids)
        ego.append(ego_values)
        ego_valid.append(current_ego_valid)
        policy_input_valid.append(
            bool(record.get("valid"))
            and visual.global_valid
            and current_ego_valid
        )
        visual_details.append(visual.detail)

    frame_row_by_index = {
        frame_index: row for row, frame_index in enumerate(frame_indices)
    }
    if len(frame_row_by_index) != len(frame_indices):
        raise FeatureCacheError("episode contains duplicate frame_index values")
    source_by_index = {
        int(item["frame_index"]): item for item in sources
    }
    samples_path = supervision / "samples.jsonl"
    samples = _load_supervision_samples(samples_path)
    sample_arrays = _build_sample_arrays(
        samples,
        run_id=run_id,
        frame_row_by_index=frame_row_by_index,
        instruction_index=instruction_index,
        source_by_index=source_by_index,
    )

    output_base.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_base)
    )
    try:
        _write_npz(
            temporary / LANGUAGE_FILE_NAME,
            instruction_ids=np.asarray(instruction_ids, dtype=np.str_),
            instruction_texts=np.asarray(
                [str(item.get("text", "")) for item in instructions],
                dtype=np.str_,
            ),
            language_splits=np.asarray(
                [str(item.get("split", "")) for item in instructions],
                dtype=np.str_,
            ),
            embeddings=language_array,
        )
        _write_npz(
            temporary / FRAME_SHARD_NAME,
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            frame_stamps_us=np.asarray(frame_stamps, dtype=np.int64),
            frame_keys=np.asarray(frame_keys, dtype=np.str_),
            source_frame_sha256=np.asarray(frame_sha256, dtype=np.str_),
            image_sha256=np.asarray(image_sha256, dtype=np.str_),
            global_visual=np.stack(global_visual).astype(
                np.float32, copy=False
            ),
            global_visual_mask=np.asarray(
                global_visual_mask, dtype=np.bool_
            ),
            entity_visual=np.stack(entity_visual).astype(
                np.float32, copy=False
            ),
            entity_visual_mask=np.stack(entity_visual_mask).astype(
                np.bool_, copy=False
            ),
            entity_features=np.stack(entity_features).astype(
                np.float32, copy=False
            ),
            entity_mask=np.stack(entity_mask).astype(np.bool_, copy=False),
            entity_ids=np.asarray(entity_ids, dtype=np.str_),
            ego=np.asarray(ego, dtype=np.float32),
            ego_valid=np.asarray(ego_valid, dtype=np.bool_),
            policy_input_valid=np.asarray(
                policy_input_valid, dtype=np.bool_
            ),
            **sample_arrays,
        )
        invalid_image_rows = [
            index
            for index, valid in enumerate(global_visual_mask)
            if not valid
        ]
        quality_report = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "run_id": run_id,
            "passed": not invalid_image_rows,
            "frame_count": len(records),
            "instruction_count": len(instructions),
            "sample_count": len(samples),
            "invalid_image_frame_rows": invalid_image_rows,
            "entity_projection_count": int(
                np.count_nonzero(np.stack(entity_visual_mask))
            ),
            "privileged_color_columns_zero": True,
            "visual_details": visual_details,
        }
        write_json_atomic(temporary / QUALITY_FILE_NAME, quality_report)
        manifest = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "scene_seed": int(records[0]["scene_seed"]),
            "cache_key": cache_key,
            "cache_key_sha256": cache_key_sha256,
            "models": {
                "language": {
                    "model_id": language_model.model_id,
                    "weights_sha256": language_model.weights_sha256,
                    "feature_dim": LANGUAGE_FEATURE_DIM,
                },
                "visual": {
                    "model_id": visual_model.model_id,
                    "weights_sha256": visual_model.weights_sha256,
                    "feature_dim": VISUAL_FEATURE_DIM,
                },
            },
            "preprocess_version": preprocess_version,
            "git_sha": git_sha,
            "source": {
                "episode_manifest_sha256": _sha256_file(
                    episode / "manifest.json"
                ),
                "supervision_manifest_sha256": _sha256_file(
                    supervision / "manifest.json"
                ),
                "supervision_samples_sha256": _sha256_file(samples_path),
                "instructions_sha256": _sha256_file(instructions_source),
                "frames": sources,
            },
            "counts": {
                "frames": len(records),
                "instructions": len(instructions),
                "samples": len(samples),
            },
            "files": {
                LANGUAGE_FILE_NAME: _sha256_file(
                    temporary / LANGUAGE_FILE_NAME
                ),
                FRAME_SHARD_NAME: _sha256_file(
                    temporary / FRAME_SHARD_NAME
                ),
                QUALITY_FILE_NAME: _sha256_file(
                    temporary / QUALITY_FILE_NAME
                ),
            },
        }
        write_json_atomic(temporary / "manifest.json", manifest)
        if not quality_report["passed"]:
            raise FeatureCacheError(
                f"{len(invalid_image_rows)} frames have invalid images; "
                "policy input is fail-closed and cache acceptance failed"
            )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    report = validate_feature_cache(output)
    return {
        "run_id": run_id,
        "output": str(output),
        "cached": False,
        **report,
    }


def validate_feature_cache(cache_dir: str | Path) -> dict[str, Any]:
    root = Path(cache_dir).resolve()
    try:
        manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        quality = json.loads(
            (root / QUALITY_FILE_NAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(f"cache metadata is unreadable: {exc}") from exc
    if manifest.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise FeatureCacheError("feature cache schema version mismatch")
    if _json_digest(manifest.get("cache_key")) != manifest.get(
        "cache_key_sha256"
    ):
        raise FeatureCacheError("cache key digest mismatch")
    for filename, expected_hash in manifest.get("files", {}).items():
        if _sha256_file(root / filename) != expected_hash:
            raise FeatureCacheError(f"cache file SHA mismatch: {filename}")

    try:
        with np.load(root / LANGUAGE_FILE_NAME, allow_pickle=False) as language:
            embeddings = np.asarray(language["embeddings"])
            instruction_ids = np.asarray(language["instruction_ids"])
        with np.load(root / FRAME_SHARD_NAME, allow_pickle=False) as frames:
            frame_indices = np.asarray(frames["frame_indices"])
            global_visual = np.asarray(frames["global_visual"])
            global_mask = np.asarray(frames["global_visual_mask"])
            entity_visual = np.asarray(frames["entity_visual"])
            entity_visual_mask = np.asarray(frames["entity_visual_mask"])
            entity_features = np.asarray(frames["entity_features"])
            entity_mask = np.asarray(frames["entity_mask"])
            entity_ids = np.asarray(frames["entity_ids"])
            ego = np.asarray(frames["ego"])
            expert = np.asarray(frames["expert_trajectories"])
            sample_frame_rows = np.asarray(frames["sample_frame_rows"])
            sample_instruction_rows = np.asarray(
                frames["sample_instruction_rows"]
            )
            policy_input_valid = np.asarray(frames["policy_input_valid"])
    except (OSError, KeyError, ValueError) as exc:
        raise FeatureCacheError(f"cache arrays are unreadable: {exc}") from exc

    counts = manifest.get("counts", {})
    frame_count = int(counts.get("frames", -1))
    instruction_count = int(counts.get("instructions", -1))
    sample_count = int(counts.get("samples", -1))
    expected_shapes = {
        "language": ((instruction_count, LANGUAGE_FEATURE_DIM), embeddings.shape),
        "global_visual": ((frame_count, VISUAL_FEATURE_DIM), global_visual.shape),
        "global_mask": ((frame_count,), global_mask.shape),
        "entity_visual": (
            (frame_count, MAX_ENTITIES, VISUAL_FEATURE_DIM),
            entity_visual.shape,
        ),
        "entity_visual_mask": (
            (frame_count, MAX_ENTITIES),
            entity_visual_mask.shape,
        ),
        "entity_features": (
            (frame_count, MAX_ENTITIES, ENTITY_FEATURE_DIM),
            entity_features.shape,
        ),
        "entity_mask": ((frame_count, MAX_ENTITIES), entity_mask.shape),
        "entity_ids": ((frame_count, MAX_ENTITIES), entity_ids.shape),
        "ego": ((frame_count, 2), ego.shape),
        "expert": ((sample_count, HORIZON, ACTION_DIM), expert.shape),
        "sample_frame_rows": ((sample_count,), sample_frame_rows.shape),
        "sample_instruction_rows": (
            (sample_count,),
            sample_instruction_rows.shape,
        ),
        "policy_input_valid": (
            (frame_count,),
            policy_input_valid.shape,
        ),
    }
    shape_errors = [
        f"{name}: expected {expected}, got {actual}"
        for name, (expected, actual) in expected_shapes.items()
        if expected != actual
    ]
    if shape_errors:
        raise FeatureCacheError("; ".join(shape_errors))
    numeric_arrays = (
        embeddings,
        global_visual,
        entity_visual,
        entity_features,
        ego,
        expert,
    )
    if not all(np.all(np.isfinite(array)) for array in numeric_arrays):
        raise FeatureCacheError("cache contains NaN or Inf")
    if np.any(entity_features[:, :, 14:16] != 0.0):
        raise FeatureCacheError("privileged entity color fields are non-zero")
    if np.any(entity_visual[~entity_visual_mask] != 0.0):
        raise FeatureCacheError("masked entity visual tokens must be zero")
    if np.any(global_visual[~global_mask] != 0.0):
        raise FeatureCacheError("invalid global visual tokens must be zero")
    if np.any(sample_frame_rows < 0) or np.any(
        sample_frame_rows >= frame_count
    ):
        raise FeatureCacheError("sample frame row is out of range")
    if np.any(sample_instruction_rows < 0) or np.any(
        sample_instruction_rows >= instruction_count
    ):
        raise FeatureCacheError("sample instruction row is out of range")
    if len(np.unique(frame_indices)) != frame_count:
        raise FeatureCacheError("frame indices are not unique")
    if len(np.unique(instruction_ids)) != instruction_count:
        raise FeatureCacheError("instruction IDs are not unique")
    if not bool(quality.get("passed")):
        raise FeatureCacheError("quality report did not pass")
    return {
        "passed": True,
        "frame_count": frame_count,
        "instruction_count": instruction_count,
        "sample_count": sample_count,
        "cache_key_sha256": manifest["cache_key_sha256"],
    }


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat = np.asarray(left, dtype=np.float64).reshape(left.shape[0], -1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(right.shape[0], -1)
    denominator = np.linalg.norm(left_flat, axis=1) * np.linalg.norm(
        right_flat, axis=1
    )
    if np.any(denominator <= 1.0e-12):
        raise FeatureCacheError("cannot compare zero-norm feature rows")
    return np.sum(left_flat * right_flat, axis=1) / denominator


def compare_feature_caches(
    left_cache: str | Path,
    right_cache: str | Path,
    *,
    sample_count: int = 20,
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Compare independently generated PC and Jetson caches."""

    if sample_count < 20:
        raise ValueError("Day 13 consistency requires at least 20 samples")
    if not 0.0 < cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must be in (0, 1]")
    left_root = Path(left_cache).resolve()
    right_root = Path(right_cache).resolve()
    validate_feature_cache(left_root)
    validate_feature_cache(right_root)
    with np.load(left_root / LANGUAGE_FILE_NAME, allow_pickle=False) as left_lang:
        left_language = np.asarray(left_lang["embeddings"])
        left_instruction_ids = np.asarray(left_lang["instruction_ids"])
    with np.load(right_root / LANGUAGE_FILE_NAME, allow_pickle=False) as right_lang:
        right_language = np.asarray(right_lang["embeddings"])
        right_instruction_ids = np.asarray(right_lang["instruction_ids"])
    with np.load(left_root / FRAME_SHARD_NAME, allow_pickle=False) as left_frames:
        left_keys = np.asarray(left_frames["frame_keys"])
        left_global = np.asarray(left_frames["global_visual"])
        left_entity = np.asarray(left_frames["entity_visual"])
        left_entity_mask = np.asarray(left_frames["entity_visual_mask"])
        left_entity_ids = np.asarray(left_frames["entity_ids"])
    with np.load(right_root / FRAME_SHARD_NAME, allow_pickle=False) as right_frames:
        right_keys = np.asarray(right_frames["frame_keys"])
        right_global = np.asarray(right_frames["global_visual"])
        right_entity = np.asarray(right_frames["entity_visual"])
        right_entity_mask = np.asarray(right_frames["entity_visual_mask"])
        right_entity_ids = np.asarray(right_frames["entity_ids"])

    if left_language.shape != right_language.shape or not np.array_equal(
        left_instruction_ids, right_instruction_ids
    ):
        raise FeatureCacheError("language cache shapes or IDs differ")
    if left_global.shape != right_global.shape:
        raise FeatureCacheError("global visual shapes differ")
    if len(left_keys) < sample_count or len(right_keys) < sample_count:
        raise FeatureCacheError(
            f"need {sample_count} common frames for consistency"
        )
    if not np.array_equal(left_keys[:sample_count], right_keys[:sample_count]):
        raise FeatureCacheError("fixed consistency frame keys differ")
    if not np.array_equal(
        left_entity_ids[:sample_count], right_entity_ids[:sample_count]
    ):
        raise FeatureCacheError("entity ID alignment differs")
    if not np.array_equal(
        left_entity_mask[:sample_count],
        right_entity_mask[:sample_count],
    ):
        raise FeatureCacheError("entity visual masks differ")

    language_cosine = _cosine_rows(left_language, right_language)
    global_cosine = _cosine_rows(
        left_global[:sample_count], right_global[:sample_count]
    )
    valid = left_entity_mask[:sample_count]
    left_valid_entity = left_entity[:sample_count][valid]
    right_valid_entity = right_entity[:sample_count][valid]
    if left_valid_entity.size == 0:
        raise FeatureCacheError("no projectable entity token to compare")
    entity_cosine = _cosine_rows(left_valid_entity, right_valid_entity)
    minimum = float(
        min(
            np.min(language_cosine),
            np.min(global_cosine),
            np.min(entity_cosine),
        )
    )
    passed = minimum >= cosine_threshold
    return {
        "passed": passed,
        "sample_count": sample_count,
        "language_min_cosine": float(np.min(language_cosine)),
        "global_visual_min_cosine": float(np.min(global_cosine)),
        "entity_visual_min_cosine": float(np.min(entity_cosine)),
        "minimum_cosine": minimum,
        "threshold": cosine_threshold,
    }


def _resolve_sha(value: str, model_path: Path | None = None) -> str:
    if value.strip().casefold() == "auto":
        if model_path is None:
            raise FeatureCacheError("auto SHA requires a model path")
        return hash_weight_tree(model_path)
    _validate_sha256(value, "weights_sha256")
    return value.strip().casefold()


def _clear_torch_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_with_cuda_retry(
    operation: Any,
    *,
    component: str,
    device: str,
    attempts: int,
) -> Any:
    if attempts <= 0:
        raise ValueError("CUDA load attempts must be positive")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if not device.startswith("cuda") or attempt >= attempts:
                raise
            print(
                "DAY13_CUDA_RETRY "
                f"component={component} attempt={attempt}/{attempts} "
                f"error={type(exc).__name__}"
            )
            _clear_torch_cuda()
            time.sleep(1.0)
    raise FeatureCacheError(
        f"{component} failed after {attempts} attempts: {last_error}"
    )


def _main_build(args: argparse.Namespace) -> int:
    language_sha = _resolve_sha(
        args.language_weights_sha256, args.language_model_path
    )
    instructions = read_jsonl(args.instructions.resolve())

    def encode_language_stage() -> np.ndarray:
        encoder = USVLanguageEncoder(
            str(args.language_model_path),
            device=args.device,
            cache_size=max(90, args.language_cache_size),
        )
        return encode_language_instructions(instructions, encoder)

    language_embeddings = _run_with_cuda_retry(
        encode_language_stage,
        component="language",
        device=args.device,
        attempts=args.cuda_load_attempts,
    )
    # Qwen is no longer needed after the 90 unique instructions are encoded.
    # Release it before MobileNet is constructed so both frozen models never
    # compete for Jetson unified memory.
    _clear_torch_cuda()

    visual_encoder = _run_with_cuda_retry(
        lambda: FrozenMobileNetEncoder(device=args.device),
        component="visual",
        device=args.device,
        attempts=args.cuda_load_attempts,
    )
    visual_sha = (
        hash_torch_module_state(visual_encoder.backbone)
        if args.visual_weights_sha256.casefold() == "auto"
        else _resolve_sha(args.visual_weights_sha256)
    )
    result = build_feature_cache(
        args.episode,
        args.supervision,
        args.instructions,
        args.output_root,
        language_encoder=None,
        visual_encoder=visual_encoder,
        language_model=ModelFingerprint(
            args.language_model_id, language_sha
        ),
        visual_model=ModelFingerprint(args.visual_model_id, visual_sha),
        git_sha=args.git_sha,
        precomputed_language_embeddings=language_embeddings,
    )
    print(
        "DAY13_FEATURE_CACHE_PASS "
        f"run_id={result['run_id']} frames={result['frame_count']} "
        f"instructions={result['instruction_count']} "
        f"samples={result['sample_count']} cached={result['cached']} "
        f"key={result['cache_key_sha256']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 13 feature-cache tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--episode", type=Path, required=True)
    build.add_argument("--supervision", type=Path, required=True)
    build.add_argument("--instructions", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--language-model-path", type=Path, required=True)
    build.add_argument(
        "--language-model-id",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    build.add_argument("--language-weights-sha256", default="auto")
    build.add_argument("--visual-model-id", default=BACKBONE_ID)
    build.add_argument("--visual-weights-sha256", default="auto")
    build.add_argument("--git-sha", required=True)
    build.add_argument("--device", default="cuda")
    build.add_argument("--language-cache-size", type=int, default=128)
    build.add_argument("--cuda-load-attempts", type=int, default=2)

    validate = subparsers.add_parser("validate")
    validate.add_argument("cache", type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--sample-count", type=int, default=20)
    compare.add_argument("--cosine-threshold", type=float, default=0.999)

    args = parser.parse_args()
    try:
        if args.command == "build":
            return _main_build(args)
        if args.command == "validate":
            result = validate_feature_cache(args.cache)
            print(
                "DAY13_FEATURE_CACHE_PASS "
                f"frames={result['frame_count']} "
                f"instructions={result['instruction_count']} "
                f"samples={result['sample_count']} "
                f"key={result['cache_key_sha256']}"
            )
            return 0
        result = compare_feature_caches(
            args.left,
            args.right,
            sample_count=args.sample_count,
            cosine_threshold=args.cosine_threshold,
        )
        status = (
            "DAY13_CONSISTENCY_PASS"
            if result["passed"]
            else "DAY13_CONSISTENCY_FAIL"
        )
        print(
            f"{status} samples={result['sample_count']} "
            f"minimum_cosine={result['minimum_cosine']:.9f} "
            f"threshold={result['threshold']:.9f}"
        )
        return 0 if result["passed"] else 1
    except (FeatureCacheError, ValueError) as exc:
        print(f"DAY13_FEATURE_CACHE_FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
