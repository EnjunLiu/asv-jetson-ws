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
from typing import Any

import numpy as np
from PIL import Image


LEGACY_MODEL_VERSION = "image_entity_ridge_v1"
MODEL_VERSION = "image_entity_ridge_v2"
GRID_WIDTH = 32
GRID_HEIGHT = 18
CHANNELS = 7  # RGB plus red/blue/white/bright spatial evidence maps
BASE_FEATURE_DIM = GRID_WIDTH * GRID_HEIGHT * CHANNELS
MOMENT_MAPS = 4
MOMENT_FEATURES_PER_MAP = 8
FEATURE_DIM = BASE_FEATURE_DIM + MOMENT_MAPS * MOMENT_FEATURES_PER_MAP
ENTITY_IDS = ("target_red", "target_blue", "target_left", "target_right")
ENTITY_COUNT = len(ENTITY_IDS)
OUTPUT_DIM = ENTITY_COUNT * 4  # visible logit + relative x/y/z per slot
POSITION_SCALE_M = np.asarray((40.0, 40.0, 5.0), dtype=np.float32)
COLOR_CALIBRATED_MODEL_VERSION = "image_entity_color_calibrated_v1"
COLOR_CALIBRATION_WIDTH = 320
COLOR_CALIBRATION_HEIGHT = 180
# Fit on the available near-range S2 red masks.  The form is intentionally
# explicit so the PC calibration script can replace these values in a model
# artifact without changing the online image-only contract.
COLOR_X_COEFFICIENTS = (0.63521458, 0.15866379)
COLOR_AREA_MIN = 0.00125
COLOR_AREA_MAX = 0.0172222222


class ImageEntityPerceptionError(RuntimeError):
    """Raised when the image perception model or input is unusable."""


def _largest_red_component(mask: np.ndarray) -> tuple[int, float, float]:
    """Return pixel count and centroid of the largest 8-connected component."""

    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    best_count = 0
    best_sum_x = 0.0
    best_sum_y = 0.0
    for row, column in zip(*np.where(mask)):
        row = int(row)
        column = int(column)
        if visited[row, column]:
            continue
        visited[row, column] = True
        stack = [(row, column)]
        count = 0
        sum_x = 0.0
        sum_y = 0.0
        while stack:
            current_row, current_column = stack.pop()
            count += 1
            sum_x += current_column
            sum_y += current_row
            for delta_row in (-1, 0, 1):
                for delta_column in (-1, 0, 1):
                    if delta_row == 0 and delta_column == 0:
                        continue
                    neighbor_row = current_row + delta_row
                    neighbor_column = current_column + delta_column
                    if not (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                    ):
                        continue
                    if mask[neighbor_row, neighbor_column] and not visited[
                        neighbor_row, neighbor_column
                    ]:
                        visited[neighbor_row, neighbor_column] = True
                        stack.append((neighbor_row, neighbor_column))
        if count > best_count:
            best_count = count
            best_sum_x = sum_x
            best_sum_y = sum_y
    if best_count == 0:
        return 0, float("nan"), float("nan")
    return best_count, best_sum_x / best_count, best_sum_y / best_count


def calibrated_red_geometry(
    image: Image.Image | np.ndarray,
) -> tuple[bool, float, float, float, tuple[float, float]]:
    """Estimate red-target geometry from RGB only.

    The returned area is the fraction of the fixed 320x180 grid occupied by
    the largest red component, and the centroid is in that same grid.  A
    missing or out-of-calibration component returns ``valid=False`` with NaN
    geometry; callers must keep their existing ridge prediction in that case.
    """

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageEntityPerceptionError(
                f"expected HxWx3 image, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ImageEntityPerceptionError("image array contains non-finite values")
        if np.issubdtype(array.dtype, np.floating) and np.max(array) <= 1.0:
            array = array * 255.0
        image = Image.fromarray(
            np.clip(array, 0.0, 255.0).astype(np.uint8), mode="RGB"
        )
    if not isinstance(image, Image.Image):
        raise ImageEntityPerceptionError("image must be a PIL image or RGB array")
    grid = np.asarray(
        image.convert("RGB").resize(
            (COLOR_CALIBRATION_WIDTH, COLOR_CALIBRATION_HEIGHT),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0
    red, green, blue = grid.transpose(2, 0, 1)
    mask = (
        (red >= 0.25)
        & (red >= green * 1.25)
        & (red >= blue * 1.15)
        & ((red - green) >= 0.08)
    )
    component_pixels, centroid_x, centroid_y = _largest_red_component(mask)
    area = component_pixels / float(COLOR_CALIBRATION_WIDTH * COLOR_CALIBRATION_HEIGHT)
    if component_pixels < 8 or not (COLOR_AREA_MIN <= area <= COLOR_AREA_MAX):
        return False, float("nan"), float("nan"), float(area), (
            float(centroid_x),
            float(centroid_y),
        )
    inverse_sqrt_area = 1.0 / math.sqrt(area)
    relative_x = COLOR_X_COEFFICIENTS[0] + COLOR_X_COEFFICIENTS[1] * inverse_sqrt_area
    # A centroid left of image centre is positive +Y in the ROS base_link
    # convention.  The 0.93 factor is the frozen camera horizontal scale.
    relative_y = (
        0.93
        * relative_x
        * (0.5 - centroid_x / COLOR_CALIBRATION_WIDTH)
        * 2.0
    )
    return (
        bool(math.isfinite(relative_x) and math.isfinite(relative_y)),
        float(relative_x),
        float(relative_y),
        float(area),
        (float(centroid_x), float(centroid_y)),
    )


def _resized_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or array to a finite, resized RGB float array."""
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageEntityPerceptionError(
                f"expected HxWx3 image, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ImageEntityPerceptionError("image array contains non-finite values")
        if np.issubdtype(array.dtype, np.floating) and np.max(array) <= 1.0:
            array = array * 255.0
        image = Image.fromarray(
            np.clip(array, 0.0, 255.0).astype(np.uint8), mode="RGB"
        )
    if not isinstance(image, Image.Image):
        raise ImageEntityPerceptionError("image must be a PIL image or RGB array")
    return np.asarray(
        image.convert("RGB").resize(
            (GRID_WIDTH, GRID_HEIGHT), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0


def _extract_legacy_image_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extract the v1 RGB/evidence vector for deployed model compatibility."""

    result = _resized_rgb(image)
    red = np.maximum(
        result[..., 0] - np.maximum(result[..., 1], result[..., 2]), 0.0
    )
    blue = np.maximum(
        result[..., 2] - np.maximum(result[..., 0], result[..., 1]), 0.0
    )
    brightness = np.mean(result, axis=-1)
    saturation = np.max(result, axis=-1) - np.min(result, axis=-1)
    white = np.clip(brightness - 1.5 * saturation, 0.0, 1.0)
    bright = np.clip(brightness - 0.45, 0.0, 1.0)
    result = np.concatenate(
        [result, red[..., None], blue[..., None], white[..., None], bright[..., None]],
        axis=-1,
    )
    result = np.ascontiguousarray(result.reshape(-1), dtype=np.float32)
    if result.shape != (BASE_FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ImageEntityPerceptionError("image feature vector is invalid")
    return result


def extract_image_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extract spatial evidence plus color/area moments for v2."""

    result = _resized_rgb(image)
    red = np.maximum(
        result[..., 0] - np.maximum(result[..., 1], result[..., 2]), 0.0
    )
    blue = np.maximum(
        result[..., 2] - np.maximum(result[..., 0], result[..., 1]), 0.0
    )
    brightness = np.mean(result, axis=-1)
    saturation = np.max(result, axis=-1) - np.min(result, axis=-1)
    white = np.clip(brightness - 1.5 * saturation, 0.0, 1.0)
    bright = np.clip(brightness - 0.45, 0.0, 1.0)
    maps = (red, blue, white, bright)
    spatial = np.concatenate(
        [
            result,
            red[..., None],
            blue[..., None],
            white[..., None],
            bright[..., None],
        ],
        axis=-1,
    )

    x_coordinates = np.linspace(-1.0, 1.0, GRID_WIDTH, dtype=np.float32)[None, :]
    y_coordinates = np.linspace(0.0, 1.0, GRID_HEIGHT, dtype=np.float32)[:, None]
    moments: list[float] = []
    for evidence in maps:
        total = float(np.sum(evidence))
        denominator = max(total, 1.0e-6)
        center_x = float(np.sum(evidence * x_coordinates) / denominator)
        center_y = float(np.sum(evidence * y_coordinates) / denominator)
        moments.extend(
            (
                math.log1p(total),
                center_x,
                center_y,
                float(
                    np.sum(evidence * (x_coordinates - center_x) ** 2)
                    / denominator
                ),
                float(
                    np.sum(evidence * (y_coordinates - center_y) ** 2)
                    / denominator
                ),
                float(np.max(evidence)),
                float(np.mean(evidence)),
                float(np.mean(evidence > 0.08)),
            )
        )
    result = np.ascontiguousarray(
        np.concatenate(
            (spatial.reshape(-1), np.asarray(moments, dtype=np.float32))
        ),
        dtype=np.float32,
    )
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ImageEntityPerceptionError("image feature vector is invalid")
    return result


def _feature_dim_for_model(model_version: str) -> int:
    return BASE_FEATURE_DIM if model_version == LEGACY_MODEL_VERSION else FEATURE_DIM


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
        expected_feature_dim = _feature_dim_for_model(self.model_version)
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        scale = np.asarray(self.feature_scale, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        if mean.shape != (expected_feature_dim,) or scale.shape != (expected_feature_dim,):
            raise ImageEntityPerceptionError("invalid feature normalization shape")
        if weights.shape != (expected_feature_dim, OUTPUT_DIM) or bias.shape != (OUTPUT_DIM,):
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
        feature_extractor = (
            _extract_legacy_image_features
            if self.model_version == LEGACY_MODEL_VERSION
            else extract_image_features
        )
        features = feature_extractor(image)
        normalized = (features - self.feature_mean) / self.feature_scale
        output = normalized @ self.weights + self.bias
        if output.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(output)):
            raise ImageEntityPerceptionError("perception output is non-finite")

        # The calibrated v3 artifact keeps the trained ridge model for the
        # non-red slots, but replaces the red slot's geometry with an
        # auditable RGB-only estimate.  This is intentionally fail-closed:
        # when the red component is absent or outside the near-range
        # calibration, the target is marked invisible instead of allowing a
        # hallucinated ridge position to drive the vessel.
        calibrated_red: tuple[bool, float, float, float, tuple[float, float]] | None = None
        if self.model_version == COLOR_CALIBRATED_MODEL_VERSION:
            calibrated_red = calibrated_red_geometry(image)
        predictions: list[ImageEntityPrediction] = []
        for index, entity_id in enumerate(ENTITY_IDS):
            offset = index * 4
            visible_logit = float(output[offset])
            visible = visible_logit >= self.visibility_threshold
            confidence = float(1.0 / (1.0 + math.exp(-np.clip(visible_logit, -30.0, 30.0))))
            geometry = output[offset + 1 : offset + 4] * POSITION_SCALE_M
            if index == 0 and calibrated_red is not None:
                red_valid, red_x, red_y, _, _ = calibrated_red
                if red_valid:
                    visible = True
                    confidence = 1.0
                    geometry = np.asarray((red_x, red_y, 0.0), dtype=np.float32)
                else:
                    visible = False
                    confidence = 0.0
                    geometry = np.zeros(3, dtype=np.float32)
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
    model_version: str = MODEL_VERSION,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an immutable model and optional JSON metadata next to it."""

    model_path = Path(path).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_version.strip():
        raise ImageEntityPerceptionError("model_version must not be empty")
    np.savez_compressed(
        model_path,
        model_version=np.asarray(model_version),
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
