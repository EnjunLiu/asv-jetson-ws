"""Image-only entity geometry model used by the online VLA path.

This is deliberately small and dependency-light: the PC trainer learns a
multi-output ridge regressor from RGB image tiles to the calibrated relative
geometry of four canonical boat slots.  At runtime the Jetson only needs
NumPy and Pillow.  UE ``Entities`` never enter this module; they are used by
the trainer as supervision labels only.

It is a first perception model, not a claim of general-purpose object
detection.  The manifest records its data split and metrics, and a later
detector can replace this file without changing the ROS topic contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


MODEL_VERSION = "image_entity_ridge_v1"
GRID_WIDTH = 32
GRID_HEIGHT = 18
CHANNELS = 7  # RGB plus red/blue/white/bright spatial evidence maps
FEATURE_DIM = GRID_WIDTH * GRID_HEIGHT * CHANNELS
ENTITY_IDS = ("target_red", "target_blue", "target_left", "target_right")
ENTITY_COUNT = len(ENTITY_IDS)
OUTPUT_DIM = ENTITY_COUNT * 4  # visible logit + relative x/y/z per slot
POSITION_SCALE_M = np.asarray((40.0, 40.0, 5.0), dtype=np.float32)


class ImageEntityPerceptionError(RuntimeError):
    """Raised when the image perception model or input is unusable."""


def extract_image_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Resize an RGB image to the frozen, deterministic model input."""

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != CHANNELS:
            raise ImageEntityPerceptionError(
                f"expected HxWx3 image, got {array.shape}"
            )
        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if not isinstance(image, Image.Image):
        raise ImageEntityPerceptionError("image must be a PIL image or RGB array")
    rgb = image.convert("RGB").resize(
        (GRID_WIDTH, GRID_HEIGHT), Image.Resampling.BILINEAR
    )
    result = np.asarray(rgb, dtype=np.float32) / 255.0
    red = np.maximum(result[..., 0] - np.maximum(result[..., 1], result[..., 2]), 0.0)
    blue = np.maximum(result[..., 2] - np.maximum(result[..., 0], result[..., 1]), 0.0)
    brightness = np.mean(result, axis=-1)
    saturation = np.max(result, axis=-1) - np.min(result, axis=-1)
    white = np.clip(brightness - 1.5 * saturation, 0.0, 1.0)
    bright = np.clip(brightness - 0.45, 0.0, 1.0)
    result = np.concatenate(
        [result, red[..., None], blue[..., None], white[..., None], bright[..., None]],
        axis=-1,
    )
    result = np.ascontiguousarray(result.reshape(-1), dtype=np.float32)
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ImageEntityPerceptionError("image feature vector is invalid")
    return result


@dataclass(frozen=True)
class ImageEntityPrediction:
    entity_id: str
    visible: bool
    confidence: float
    relative_x: float
    relative_y: float
    relative_z: float


@dataclass(frozen=True)
class ImageEntityModel:
    """Immutable inference weights loaded from the PC trainer output."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    model_version: str = MODEL_VERSION
    visibility_threshold: float = 0.0

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        scale = np.asarray(self.feature_scale, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        if mean.shape != (FEATURE_DIM,) or scale.shape != (FEATURE_DIM,):
            raise ImageEntityPerceptionError("invalid feature normalization shape")
        if weights.shape != (FEATURE_DIM, OUTPUT_DIM) or bias.shape != (OUTPUT_DIM,):
            raise ImageEntityPerceptionError("invalid perception weight shape")
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(bias))
            or np.any(scale <= 0.0)
        ):
            raise ImageEntityPerceptionError("perception weights contain invalid values")
        object.__setattr__(self, "feature_mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "feature_scale", np.ascontiguousarray(scale))
        object.__setattr__(self, "weights", np.ascontiguousarray(weights))
        object.__setattr__(self, "bias", np.ascontiguousarray(bias))

    @classmethod
    def load(cls, path: str | Path) -> "ImageEntityModel":
        model_path = Path(path).expanduser()
        if not model_path.is_file():
            raise ImageEntityPerceptionError(f"model not found: {model_path}")
        try:
            with np.load(model_path, allow_pickle=False) as data:
                version = str(data["model_version"].item())
                return cls(
                    feature_mean=data["feature_mean"],
                    feature_scale=data["feature_scale"],
                    weights=data["weights"],
                    bias=data["bias"],
                    model_version=version,
                    visibility_threshold=float(data["visibility_threshold"].item()),
                )
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ImageEntityPerceptionError(
                f"cannot load perception model {model_path}: {exc}"
            ) from exc

    def predict(self, image: Image.Image | np.ndarray) -> tuple[ImageEntityPrediction, ...]:
        features = extract_image_features(image)
        normalized = (features - self.feature_mean) / self.feature_scale
        output = normalized @ self.weights + self.bias
        if output.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(output)):
            raise ImageEntityPerceptionError("perception output is non-finite")
        predictions: list[ImageEntityPrediction] = []
        for index, entity_id in enumerate(ENTITY_IDS):
            offset = index * 4
            visible_logit = float(output[offset])
            visible = visible_logit >= self.visibility_threshold
            confidence = float(1.0 / (1.0 + math.exp(-np.clip(visible_logit, -30.0, 30.0))))
            geometry = output[offset + 1 : offset + 4] * POSITION_SCALE_M
            predictions.append(
                ImageEntityPrediction(
                    entity_id=entity_id,
                    visible=visible,
                    confidence=confidence,
                    relative_x=float(geometry[0]),
                    relative_y=float(geometry[1]),
                    relative_z=float(geometry[2]),
                )
            )
        return tuple(predictions)


def save_model(
    path: str | Path,
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    visibility_threshold: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an immutable model and optional JSON metadata next to it."""

    model_path = Path(path).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        model_version=np.asarray(MODEL_VERSION),
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_scale=np.asarray(feature_scale, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        bias=np.asarray(bias, dtype=np.float32),
        visibility_threshold=np.asarray(float(visibility_threshold), dtype=np.float32),
    )
    if metadata is not None:
        model_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
