from dataclasses import dataclass
from collections import OrderedDict
import ast
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from asv_vla.visual_encoder import (
    FEATURE_DIM,
    TOKEN_COUNT,
    CameraProfile,
    DEFAULT_LOW_LIGHT_BRIGHTNESS,
    DEFAULT_LOW_LIGHT_CONTRAST,
    DEFAULT_LOW_LIGHT_GAMMA,
    FrozenMobileNetEncoder,
    InvalidImageError,
    TargetProjectionError,
    TargetSelectionError,
    crop_around_pixel,
    decode_camera_image,
    enhance_low_light_image,
    make_target_crop,
    project_target_to_pixel,
    select_target,
)


NODE = Path(__file__).resolve().parents[1] / "asv_vla" / "visual_encoder_node.py"


def _load_frame_sync_cache():
    tree = ast.parse(NODE.read_text(encoding="utf-8"), filename=str(NODE))
    wanted = {
        "FrameKey",
        "DEFAULT_SYNC_FRAME_TOLERANCE",
        "MAX_SYNC_FRAME_TOLERANCE",
        "FrameSyncCache",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FrameSyncCache":
            nodes.append(node)
        elif isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", "")
            if target == "FrameKey":
                nodes.append(node)
        elif isinstance(node, ast.Assign):
            if any(getattr(target, "id", "") in wanted for target in node.targets):
                nodes.append(node)
    namespace = {
        "OrderedDict": OrderedDict,
        "math": math,
        "time": time,
        "DEFAULT_SYNC_FRAME_TOLERANCE": 12,
        "MAX_SYNC_FRAME_TOLERANCE": 12,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(NODE), "exec"),
        namespace,
    )
    return namespace["FrameSyncCache"]


FrameSyncCache = _load_frame_sync_cache()


def _sync_message(run_id: str, scene_seed: int, frame_index: int):
    return SimpleNamespace(
        run_id=run_id,
        scene_seed=scene_seed,
        frame_index=frame_index,
    )


def test_frame_sync_exact_match_is_preferred_over_near_candidate():
    cache = FrameSyncCache(cache_size=4, frame_tolerance=2, ttl_sec=1.0)
    near = _sync_message("run-a", 7, 9)
    exact = _sync_message("run-a", 7, 10)
    entities = _sync_message("run-a", 7, 10)

    cache.put_frame(near, received_at=0.0)
    cache.put_frame(exact, received_at=0.0)
    cache.put_entities(entities, received_at=0.1)

    pair = cache.match_for_entities(
        cache.key_for(entities), now=0.1
    )
    assert pair == (exact, entities, 0)
    assert cache.frames[cache.key_for(near)][0] is near


def test_frame_sync_accepts_near_match_and_reports_signed_delta():
    cache = FrameSyncCache(cache_size=4, frame_tolerance=6, ttl_sec=1.0)
    frame = _sync_message("run-a", 7, 20)
    entities = _sync_message("run-a", 7, 25)

    cache.put_frame(frame, received_at=0.0)
    cache.put_entities(entities, received_at=0.1)
    pair = cache.match_for_entities(cache.key_for(entities), now=0.1)

    assert pair == (frame, entities, 5)


def test_frame_sync_rejects_frame_outside_tolerance():
    cache = FrameSyncCache(cache_size=4, frame_tolerance=6, ttl_sec=1.0)
    frame = _sync_message("run-a", 7, 20)
    entities = _sync_message("run-a", 7, 27)

    cache.put_frame(frame, received_at=0.0)
    cache.put_entities(entities, received_at=0.1)

    assert cache.match_for_entities(cache.key_for(entities), now=0.1) is None
    assert len(cache.frames) == 1
    assert len(cache.entities) == 1


def test_frame_sync_rejects_unbounded_tolerance_configuration():
    with pytest.raises(ValueError):
        FrameSyncCache(cache_size=4, frame_tolerance=13, ttl_sec=1.0)


def test_frame_sync_isolates_run_and_scene_boundaries():
    cache = FrameSyncCache(cache_size=8, frame_tolerance=6, ttl_sec=1.0)
    frame = _sync_message("run-a", 7, 10)
    wrong_run = _sync_message("run-b", 7, 10)
    wrong_scene = _sync_message("run-a", 8, 10)
    exact = _sync_message("run-a", 7, 10)

    cache.put_frame(frame, received_at=0.0)
    cache.put_entities(wrong_run, received_at=0.1)
    cache.put_entities(wrong_scene, received_at=0.1)
    assert cache.match_for_entities(cache.key_for(wrong_run), now=0.1) is None
    assert cache.match_for_entities(cache.key_for(wrong_scene), now=0.1) is None

    cache.put_entities(exact, received_at=0.1)
    assert cache.match_for_entities(cache.key_for(exact), now=0.1) == (
        frame,
        exact,
        0,
    )


def test_frame_sync_expires_entries_and_keeps_each_cache_bounded():
    cache = FrameSyncCache(cache_size=2, frame_tolerance=6, ttl_sec=1.0)
    frames = [
        _sync_message("run-a", 7, index) for index in range(3)
    ]
    for index, frame in enumerate(frames):
        evicted = cache.put_frame(frame, received_at=float(index))
        if index == 2:
            assert evicted == (frames[0],)

    assert len(cache.frames) == 2
    cache.put_entities(_sync_message("run-a", 7, 2), received_at=2.0)
    expired_frames = cache.expire(now=10.0)

    assert expired_frames == (frames[1], frames[2])
    assert len(cache.frames) == 0
    assert len(cache.entities) == 0


def test_frame_sync_rejects_and_removes_expired_source_during_match():
    cache = FrameSyncCache(cache_size=4, frame_tolerance=6, ttl_sec=1.0)
    entities = _sync_message("run-a", 7, 10)
    cache.put_entities(entities, received_at=0.0)

    assert cache.match_for_entities(cache.key_for(entities), now=1.0) is None
    assert len(cache.entities) == 0


@dataclass
class FakeEntity:
    entity_id: str
    relative_x: float
    relative_y: float
    relative_z: float
    is_target: bool = True
    visible: bool = True
    valid: bool = True


def test_projection_matches_frozen_camera_profile():
    profile = CameraProfile()
    pixel_x, pixel_y, depth = project_target_to_pixel(
        1.5, 0.0, -0.10554275, profile
    )

    assert pixel_x == pytest.approx(640.0, abs=1.0e-6)
    assert pixel_y == pytest.approx(482.05, abs=0.01)
    assert depth == pytest.approx(1.1025, abs=0.001)


def test_ros_left_and_right_project_to_expected_image_sides():
    profile = CameraProfile()
    left_x, _, _ = project_target_to_pixel(5.0, 1.0, 0.0, profile)
    right_x, _, _ = project_target_to_pixel(5.0, -1.0, 0.0, profile)

    assert left_x < profile.width / 2.0
    assert right_x > profile.width / 2.0


def test_target_behind_camera_is_rejected():
    with pytest.raises(TargetProjectionError):
        project_target_to_pixel(0.0, 0.0, 0.2, CameraProfile())


def test_target_selection_is_valid_visible_nearest_and_deterministic():
    selected = select_target([
        FakeEntity("far", 10.0, 0.0, 0.0),
        FakeEntity("hidden", 1.0, 0.0, 0.0, visible=False),
        FakeEntity("near-b", 2.0, 0.0, 0.0),
        FakeEntity("near-a", 2.0, 0.0, 0.0),
    ])
    assert selected.entity_id == "near-a"


def test_no_target_is_rejected():
    with pytest.raises(TargetSelectionError):
        select_target([
            FakeEntity("not-target", 2.0, 0.0, 0.0, is_target=False)
        ])


def test_crop_has_fixed_shape_and_black_padding_at_edge():
    image = Image.new("RGB", (1280, 720), (255, 0, 0))
    crop = crop_around_pixel(image, 1.0, 1.0, 224)

    assert crop.size == (224, 224)
    assert crop.getpixel((0, 0)) == (0, 0, 0)
    assert crop.getpixel((112, 112)) == (255, 0, 0)


def test_wrong_image_shape_fails_closed_before_crop():
    image = Image.new("RGB", (640, 480))
    target = FakeEntity("target", 5.0, 0.0, 0.0)
    with pytest.raises(InvalidImageError):
        make_target_crop(image, target, CameraProfile())


def test_empty_or_wrong_encoding_image_is_rejected():
    with pytest.raises(InvalidImageError):
        decode_camera_image(b"", "jpeg")
    with pytest.raises(InvalidImageError):
        decode_camera_image(b"not-a-jpeg", "raw")


def test_low_light_preprocess_lifts_dark_image_without_mutating_input():
    image = Image.new("RGB", (32, 24), (12, 24, 48))
    before = np.asarray(image).copy()
    enhanced = enhance_low_light_image(image)

    assert enhanced is not image
    assert enhanced.mode == "RGB"
    assert enhanced.size == image.size
    assert np.array_equal(np.asarray(image), before)
    assert np.asarray(enhanced, dtype=np.float32).mean() > before.mean()


def test_low_light_preprocess_can_be_disabled_but_still_returns_copy():
    image = Image.new("RGB", (8, 6), (12, 24, 48))
    result = enhance_low_light_image(
        image,
        enabled=False,
        gamma=DEFAULT_LOW_LIGHT_GAMMA,
        brightness=DEFAULT_LOW_LIGHT_BRIGHTNESS,
        contrast=DEFAULT_LOW_LIGHT_CONTRAST,
    )
    assert result is not image
    assert np.array_equal(np.asarray(result), np.asarray(image))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0},
        {"brightness": float("nan")},
        {"contrast": -1.0},
    ],
)
def test_low_light_preprocess_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        enhance_low_light_image(Image.new("RGB", (2, 2)), **kwargs)


def test_frozen_backbone_produces_two_fixed_normalized_tokens():
    torch = pytest.importorskip("torch")

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, FEATURE_DIM, 1)

        def forward(self, inputs):
            return self.conv(inputs)

    backbone = TinyBackbone()
    encoder = FrozenMobileNetEncoder(
        device="cpu",
        backbone=backbone,
        feature_dim=FEATURE_DIM,
    )
    image = Image.new("RGB", (1280, 720), (10, 20, 30))
    crop = Image.new("RGB", (224, 224), (30, 20, 10))
    first = encoder.encode_pair(image, crop)
    second = encoder.encode_pair(image, crop)

    assert encoder.frozen
    assert first.shape == (TOKEN_COUNT, FEATURE_DIM)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1.0e-5)
    assert np.array_equal(first, second)
