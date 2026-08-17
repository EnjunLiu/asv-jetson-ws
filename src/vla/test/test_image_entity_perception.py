from pathlib import Path
import sys
import types

import numpy as np
import pytest
from PIL import Image
from PIL import ImageDraw

from vla.image_entity_perception import (
    BASE_FEATURE_DIM,
    COLOR_CALIBRATION_WIDTH,
    ENTITY_COUNT,
    FEATURE_DIM,
    FUSED_FEATURE_DIM,
    LEGACY_MODEL_VERSION,
    LANGUAGE_EMBEDDING_DIM,
    LOW_LIGHT_PREPROCESS_BRIGHTNESS,
    LOW_LIGHT_PREPROCESS_CONTRAST,
    LOW_LIGHT_PREPROCESS_CONTRACT,
    LOW_LIGHT_PREPROCESS_ENABLED,
    LOW_LIGHT_PREPROCESS_GAMMA,
    MODEL_VERSION,
    OUTPUT_DIM,
    ImageEntityModel,
    ImageEntityPerceptionError,
    ImageEntityPrediction,
    calibrated_color_geometry,
    extract_image_features,
    _extract_torch_image_features,
    parse_task_instruction,
    save_model,
    select_task_entities,
)


def test_low_light_preprocess_contract_is_fixed() -> None:
    assert LOW_LIGHT_PREPROCESS_ENABLED is False
    assert LOW_LIGHT_PREPROCESS_CONTRACT == (
        "ue5_capture_gamma065_brightness100_contrast100"
    )
    assert LOW_LIGHT_PREPROCESS_GAMMA == pytest.approx(0.65)
    assert LOW_LIGHT_PREPROCESS_BRIGHTNESS == pytest.approx(1.0)
    assert LOW_LIGHT_PREPROCESS_CONTRAST == pytest.approx(1.0)


def test_image_features_are_fixed_and_finite() -> None:
    image = Image.new("RGB", (1280, 720), (12, 30, 60))
    features = extract_image_features(image)
    assert features.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(features))


def test_language_conditioned_torch_features_stop_before_embedding_concat() -> None:
    torch = pytest.importorskip("torch")
    features = _extract_torch_image_features(
        Image.new("RGB", (1280, 720), (12, 30, 60)),
        torch,
        device="cpu",
        model_version=MODEL_VERSION,
    )

    assert tuple(features.shape) == (FEATURE_DIM,)
    assert bool(torch.isfinite(features).all())


def test_image_features_accept_rgb_arrays_and_include_moments() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[8:16, 10:20] = (220, 20, 20)
    features = extract_image_features(image)
    assert features.shape == (FEATURE_DIM,)
    assert FEATURE_DIM > BASE_FEATURE_DIM
    assert np.all(np.isfinite(features))


def test_legacy_model_round_trip_keeps_v1_feature_contract(tmp_path: Path) -> None:
    path = tmp_path / "legacy_perception.npz"
    save_model(
        path,
        feature_mean=np.zeros(BASE_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(BASE_FEATURE_DIM, dtype=np.float32),
        weights=np.zeros((BASE_FEATURE_DIM, OUTPUT_DIM), dtype=np.float32),
        bias=np.ones(OUTPUT_DIM, dtype=np.float32),
        model_version=LEGACY_MODEL_VERSION,
    )
    with pytest.raises(ImageEntityPerceptionError, match="MODEL_SCHEMA_MISMATCH"):
        ImageEntityModel.load(path)
    model = ImageEntityModel.load(path, allow_legacy=True)
    assert model.model_version == LEGACY_MODEL_VERSION
    assert len(model.predict(np.zeros((24, 32, 3), dtype=np.uint8))) == ENTITY_COUNT


def test_model_round_trip_and_no_velocity_output(tmp_path: Path) -> None:
    path = tmp_path / "perception.npz"
    save_model(
        path,
        feature_mean=np.zeros(FUSED_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(FUSED_FEATURE_DIM, dtype=np.float32),
        weights=np.zeros((FUSED_FEATURE_DIM, OUTPUT_DIM), dtype=np.float32),
        bias=np.ones(OUTPUT_DIM, dtype=np.float32),
        language_model_id="test-language",
        language_weights_sha256="a" * 64,
        metadata={"velocity_output": False},
    )
    model = ImageEntityModel.load(path)
    predictions = model.predict(
        Image.new("RGB", (1280, 720), (40, 50, 60)),
        task_embedding=np.ones(LANGUAGE_EMBEDDING_DIM, dtype=np.float32),
    )
    assert len(predictions) == ENTITY_COUNT
    assert all(prediction.visible for prediction in predictions)
    assert all(
        np.isfinite(
            [prediction.relative_x, prediction.relative_y, prediction.relative_z]
        ).all()
        for prediction in predictions
    )


def test_new_model_requires_a_real_task_embedding() -> None:
    model = _all_visible_model()
    with pytest.raises(ImageEntityPerceptionError, match="task embedding"):
        model.predict(Image.new("RGB", (1280, 720), (40, 50, 60)))


def test_language_features_are_fused_into_model_projection() -> None:
    weights = np.zeros((FUSED_FEATURE_DIM, OUTPUT_DIM), dtype=np.float32)
    weights[FEATURE_DIM, 0] = 2.0
    model = ImageEntityModel(
        feature_mean=np.zeros(FUSED_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(FUSED_FEATURE_DIM, dtype=np.float32),
        weights=weights,
        bias=np.zeros(OUTPUT_DIM, dtype=np.float32),
        language_model_id="test-language",
        language_weights_sha256="a" * 64,
    )
    image = Image.new("RGB", (1280, 720), (40, 50, 60))
    positive = model.predict(image, task_embedding=_embedding(1.0))
    negative = model.predict(image, task_embedding=_embedding(-1.0))
    assert positive[0].confidence > negative[0].confidence
    assert positive[0].relative_x == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("color", "rgb"),
    (("red", (220, 20, 20)), ("blue", (20, 20, 220))),
)
def test_calibrated_color_geometry_is_symmetric(
    color: str, rgb: tuple[int, int, int]
) -> None:
    left = Image.new("RGB", (1280, 720), (20, 30, 40))
    ImageDraw.Draw(left).rectangle((180, 300, 300, 390), fill=rgb)
    right = Image.new("RGB", (1280, 720), (20, 30, 40))
    ImageDraw.Draw(right).rectangle((980, 300, 1100, 390), fill=rgb)

    left_valid, left_x, left_y, left_area, left_centroid = calibrated_color_geometry(
        left, color
    )
    right_valid, right_x, right_y, right_area, right_centroid = calibrated_color_geometry(
        right, color
    )

    assert left_valid and right_valid
    assert left_area == pytest.approx(right_area, rel=0.05)
    assert left_x == pytest.approx(right_x, rel=0.05)
    assert left_y > 0.0
    assert right_y < 0.0
    assert left_centroid[0] < COLOR_CALIBRATION_WIDTH / 2.0
    assert right_centroid[0] > COLOR_CALIBRATION_WIDTH / 2.0


@pytest.mark.parametrize(
    ("color", "rgb"),
    (("red", (220, 20, 20)), ("blue", (20, 20, 220))),
)
def test_calibrated_color_geometry_keeps_near_range_component(
    color: str, rgb: tuple[int, int, int]
) -> None:
    image = Image.new("RGB", (1280, 720), (20, 30, 40))
    # This component occupies about 3.6% of the image, larger than the old
    # 1.72% cap and representative of a target after it approaches the ASV.
    ImageDraw.Draw(image).rectangle((280, 260, 520, 400), fill=rgb)

    valid, relative_x, relative_y, area, _ = calibrated_color_geometry(image, color)

    assert valid
    assert area > 0.0172222222
    assert np.isfinite((relative_x, relative_y)).all()


def test_calibrated_color_geometry_still_rejects_oversized_background_component() -> None:
    image = Image.new("RGB", (1280, 720), (20, 30, 40))
    ImageDraw.Draw(image).rectangle((0, 0, 1279, 719), fill=(220, 20, 20))

    valid, relative_x, relative_y, area, centroid = calibrated_color_geometry(image, "red")

    assert not valid
    assert area == 0.0
    assert np.isnan(relative_x)
    assert np.isnan(relative_y)
    assert np.isnan(centroid[0])
    assert np.isnan(centroid[1])


@pytest.mark.parametrize(
    "color",
    ("red", "blue"),
)
def test_calibrated_color_geometry_fails_closed_without_component(
    color: str,
) -> None:
    valid, relative_x, relative_y, area, centroid = calibrated_color_geometry(
        Image.new("RGB", (1280, 720), (20, 30, 40)), color
    )
    assert not valid
    assert np.isnan(relative_x)
    assert np.isnan(relative_y)
    assert area == 0.0
    assert np.isnan(centroid[0])
    assert np.isnan(centroid[1])


def test_blue_calibration_accepts_cyan_rgb_component() -> None:
    image = Image.new("RGB", (1280, 720), (8, 35, 75))
    ImageDraw.Draw(image).rectangle((560, 330, 680, 420), fill=(8, 58, 62))

    valid, relative_x, relative_y, area, centroid = calibrated_color_geometry(
        image, "blue"
    )

    assert valid
    assert area >= 0.00125
    assert centroid[0] == pytest.approx(620.0 / 4.0, abs=2.0)
    assert relative_x > 0.0
    assert relative_y == pytest.approx(0.0, abs=0.2)


def test_task_parser_covers_color_bearing_and_stop() -> None:
    red = parse_task_instruction("跟随红色目标船，保持3米距离")
    assert red.valid and red.is_follow
    assert red.color == "red"
    assert red.bearing == ""
    assert red.instruction_id == "follow_red"

    left = parse_task_instruction("follow the left target")
    assert left.valid and left.is_follow
    assert left.bearing == "left"
    assert left.instruction_id == "follow_left"

    stop = parse_task_instruction("STOP")
    assert stop.valid and stop.is_stop
    assert stop.instruction_id == "stop"

    assert not parse_task_instruction("unknown task").valid


def test_task_selection_only_returns_relevant_visible_entities() -> None:
    predictions = (
        ImageEntityPrediction("target_red", True, 0.9, 4.0, 1.0, 0.0),
        ImageEntityPrediction("target_blue", True, 0.9, 4.0, -1.0, 0.0),
        ImageEntityPrediction("target_left", True, 0.9, 4.0, 1.0, 0.0),
    )
    selected_red = select_task_entities(predictions, "follow red")
    assert [prediction.entity_id for prediction in selected_red] == ["target_red"]

    selected_left = select_task_entities(predictions, "follow left")
    assert [prediction.entity_id for prediction in selected_left] == ["target_left"]

    assert select_task_entities(predictions, "stop") == ()


def _all_visible_model() -> ImageEntityModel:
    return ImageEntityModel(
        feature_mean=np.zeros(FUSED_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(FUSED_FEATURE_DIM, dtype=np.float32),
        weights=np.zeros((FUSED_FEATURE_DIM, OUTPUT_DIM), dtype=np.float32),
        bias=np.ones(OUTPUT_DIM, dtype=np.float32),
        language_model_id="test-language",
        language_weights_sha256="a" * 64,
    )


def _color_ridge_model(
    *, color: str, model_version: str = MODEL_VERSION
) -> ImageEntityModel:
    bias = np.zeros(OUTPUT_DIM, dtype=np.float32)
    offset = {"red": 0, "blue": 4}[color]
    bias[offset : offset + 4] = (-1.0, 0.25, -0.5, 0.4)
    feature_dim = FEATURE_DIM if model_version != MODEL_VERSION else FUSED_FEATURE_DIM
    return ImageEntityModel(
        feature_mean=np.zeros(feature_dim, dtype=np.float32),
        feature_scale=np.ones(feature_dim, dtype=np.float32),
        weights=np.zeros((feature_dim, OUTPUT_DIM), dtype=np.float32),
        bias=bias,
        model_version=model_version,
        language_model_id="test-language",
        language_weights_sha256="a" * 64,
    )


def _embedding(value: float = 1.0) -> np.ndarray:
    result = np.zeros(LANGUAGE_EMBEDDING_DIM, dtype=np.float32)
    result[0] = value
    return result


@pytest.mark.parametrize(
    ("color", "rgb"),
    (("red", (220, 20, 20)), ("blue", (20, 20, 220))),
)
def test_prediction_uses_original_color_reference_for_red_and_blue(
    monkeypatch,
    color: str,
    rgb: tuple[int, int, int],
) -> None:
    import vla.image_entity_perception as perception

    original = Image.new("RGB", (1280, 720), (20, 30, 40))
    ImageDraw.Draw(original).rectangle((180, 300, 300, 390), fill=rgb)
    enhanced = Image.new("RGB", (1280, 720), (120, 130, 140))
    model = _color_ridge_model(color=color)
    feature_inputs = []

    monkeypatch.setattr(
        perception,
        "extract_image_features",
        lambda image: feature_inputs.append(image)
        or np.zeros(FEATURE_DIM, dtype=np.float32),
    )
    with_reference = model.predict(
        enhanced, color_image=original, task_embedding=_embedding()
    )
    without_reference = model.predict(enhanced, task_embedding=_embedding())
    target = next(
        prediction
        for prediction in with_reference
        if prediction.entity_id == f"target_{color}"
    )
    expected = calibrated_color_geometry(original, color)

    assert feature_inputs == [enhanced, enhanced]
    assert expected[0]
    assert target.visible
    assert target.confidence == 1.0
    assert target.relative_x == pytest.approx(expected[1])
    assert target.relative_y == pytest.approx(expected[2])
    assert target.relative_z == 0.0
    assert not next(
        prediction
        for prediction in without_reference
        if prediction.entity_id == f"target_{color}"
    ).visible


@pytest.mark.parametrize(
    "color",
    ("red", "blue"),
)
def test_prediction_fails_closed_when_original_color_is_missing(
    color: str,
) -> None:
    model = _color_ridge_model(color=color)
    feature_image = Image.new("RGB", (1280, 720), (120, 130, 140))
    color_image = Image.new("RGB", (1280, 720), (20, 30, 40))
    prediction = next(
        item
        for item in model.predict(
            feature_image,
            color_image=color_image,
            task_embedding=_embedding(),
        )
        if item.entity_id == f"target_{color}"
    )

    assert not prediction.visible
    assert prediction.confidence == 0.0
    assert (
        prediction.relative_x,
        prediction.relative_y,
        prediction.relative_z,
    ) == (0.0, 0.0, 0.0)


def test_predict_applies_task_condition_at_model_output_boundary() -> None:
    image = Image.new("RGB", (1280, 720), (40, 50, 60))
    model = _all_visible_model()

    unconditioned = model.predict(image, task_embedding=_embedding())
    red = model.predict(
        image,
        task=parse_task_instruction("follow red"),
        task_embedding=_embedding(),
    )
    blue = model.predict(image, task="follow blue", task_embedding=_embedding())

    assert all(prediction.visible for prediction in unconditioned)
    assert [prediction.entity_id for prediction in red if prediction.visible] == [
        "target_red"
    ]
    assert [prediction.entity_id for prediction in blue if prediction.visible] == [
        "target_blue"
    ]
    for red_prediction, blue_prediction in zip(red, blue):
        if red_prediction.entity_id in {"target_red", "target_blue"}:
            assert red_prediction.visible != blue_prediction.visible
        assert (
            red_prediction.relative_x,
            red_prediction.relative_y,
            red_prediction.relative_z,
        ) == (
            blue_prediction.relative_x,
            blue_prediction.relative_y,
            blue_prediction.relative_z,
        )


def test_cuda_path_uses_torch_feature_helper_without_numpy_fallback(
    monkeypatch,
) -> None:
    model = _all_visible_model()
    monkeypatch.setattr(
        "vla.image_entity_perception._torch_for_device",
        lambda _device: object(),
    )
    monkeypatch.setattr(
        "vla.image_entity_perception._extract_torch_image_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ImageEntityPerceptionError("CUDA_FEATURE_PATH")
        ),
    )
    monkeypatch.setattr(
        "vla.image_entity_perception.extract_image_features",
        lambda _image: pytest.fail("CUDA path used NumPy feature extraction"),
    )

    with pytest.raises(ImageEntityPerceptionError, match="CUDA_FEATURE_PATH"):
        model.predict(
            np.zeros((24, 32, 3), dtype=np.uint8),
            device="cuda",
            task_embedding=_embedding(),
        )


def test_cuda_request_fails_closed_without_silent_numpy_fallback(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(ImageEntityPerceptionError, match="CUDA perception requested"):
        ImageEntityModel.validate_device("cuda")
