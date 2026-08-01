from pathlib import Path

import numpy as np
from PIL import Image

from asv_vla.image_entity_perception import (
    ENTITY_COUNT,
    FEATURE_DIM,
    OUTPUT_DIM,
    ImageEntityModel,
    extract_image_features,
    save_model,
)


def test_image_features_are_fixed_and_finite() -> None:
    image = Image.new("RGB", (1280, 720), (12, 30, 60))
    features = extract_image_features(image)
    assert features.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(features))


def test_model_round_trip_and_no_velocity_output(tmp_path: Path) -> None:
    path = tmp_path / "perception.npz"
    save_model(
        path,
        feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(FEATURE_DIM, dtype=np.float32),
        weights=np.zeros((FEATURE_DIM, OUTPUT_DIM), dtype=np.float32),
        bias=np.ones(OUTPUT_DIM, dtype=np.float32),
        metadata={"velocity_output": False},
    )
    model = ImageEntityModel.load(path)
    predictions = model.predict(Image.new("RGB", (1280, 720), (40, 50, 60)))
    assert len(predictions) == ENTITY_COUNT
    assert all(prediction.visible for prediction in predictions)
    assert all(
        np.isfinite(
            [prediction.relative_x, prediction.relative_y, prediction.relative_z]
        ).all()
        for prediction in predictions
    )
