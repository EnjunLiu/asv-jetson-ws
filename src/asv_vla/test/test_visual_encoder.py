from dataclasses import dataclass

import numpy as np
from PIL import Image
import pytest

from asv_vla.visual_encoder import (
    FEATURE_DIM,
    TOKEN_COUNT,
    CameraProfile,
    FrozenMobileNetEncoder,
    InvalidImageError,
    TargetProjectionError,
    TargetSelectionError,
    crop_around_pixel,
    decode_camera_image,
    make_target_crop,
    project_target_to_pixel,
    select_target,
)


@dataclass
class FakeEntity:
    entity_id: str
    relative_x: float
    relative_y: float
    relative_z: float
    is_target: bool = True
    visible: bool = True
    valid: bool = True


def test_projection_matches_frozen_day4_camera_profile():
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
