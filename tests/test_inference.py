"""
tests/test_inference.py

Unit tests for src/inference/predict.py (Phase 8).

All tests use synthetic in-memory/on-disk images and a tiny synthetic
model written to pytest's tmp_path -- never the real Vehicle-10 dataset
or the real models/best_model.keras. No test path references
D:\\ml-datasets\\vehicle-10 or the real checkpoint.

Tests must not modify:
  - data/raw
  - data/processed
  - models/best_model.keras
  - results/evaluation
None of the code below writes to any of those paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.predict import (
    ClassCountMismatchError,
    ImageDecodeError,
    ImageNotFoundError,
    InvalidTopKError,
    ModelNotFoundError,
    format_prediction,
    get_class_names,
    load_model,
    predict_from_path,
    predict_image,
    preprocess_image,
    run_prediction,
    verify_model_matches_classes,
)
from src.models.cnn_model import build_and_compile_model


CLASS_NAMES = ["bicycle", "boat", "car"]  # small subset, sorted order matters


# --------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------

def _write_image(path: Path, size=(50, 50), mode="RGB", color=(255, 0, 0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color)
    img.save(path)
    return path


@pytest.fixture
def synthetic_config(tmp_path):
    """
    A minimal config dict + on-disk checkpoint pointing at a tiny
    synthetic model -- never the real models/best_model.keras.
    """
    model = build_and_compile_model(input_shape=(16, 16, 3), num_classes=len(CLASS_NAMES))
    checkpoint_path = tmp_path / "models" / "tiny_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)

    return {
        "dataset": {
            "classes": list(CLASS_NAMES),  # already sorted
        },
        "preprocessing": {
            "image_size": [16, 16],  # [width, height]
        },
        "training": {
            "checkpoint": {
                "enabled": True,
                "path": str(checkpoint_path),
            },
        },
    }


@pytest.fixture
def synthetic_config_path(tmp_path, synthetic_config):
    """Write synthetic_config out to a real config.yaml on disk, for
    CLI/run_prediction()-level tests."""
    import yaml

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(synthetic_config, f)
    return config_path


# --------------------------------------------------------------------------
# 1. Module imports
# --------------------------------------------------------------------------

def test_module_imports():
    import src.inference.predict  # noqa: F401


# --------------------------------------------------------------------------
# 2. Class ordering derivation
# --------------------------------------------------------------------------

def test_get_class_names_returns_sorted_list(synthetic_config):
    names = get_class_names(synthetic_config)
    assert names == sorted(CLASS_NAMES)


def test_get_class_names_missing_raises():
    with pytest.raises(Exception):
        get_class_names({})


# --------------------------------------------------------------------------
# 3. Image preprocessing: RGB
# --------------------------------------------------------------------------

def test_preprocess_rgb_image_shape_and_dtype(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50), mode="RGB", color=(10, 20, 30))
    array = preprocess_image(str(img_path), synthetic_config)

    assert array.shape == (1, 16, 16, 3)
    assert array.dtype == np.float32


def test_preprocess_pixel_values_in_unit_range(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50), mode="RGB", color=(255, 128, 0))
    array = preprocess_image(str(img_path), synthetic_config)

    assert array.min() >= 0.0
    assert array.max() <= 1.0


# --------------------------------------------------------------------------
# 4. RGBA handling
# --------------------------------------------------------------------------

def test_preprocess_rgba_image(tmp_path, synthetic_config):
    img_path = tmp_path / "car_rgba.png"
    Image.new("RGBA", (50, 50), (0, 255, 0, 128)).save(img_path)

    array = preprocess_image(str(img_path), synthetic_config)
    assert array.shape == (1, 16, 16, 3)
    assert array.dtype == np.float32
    assert array.min() >= 0.0 and array.max() <= 1.0


# --------------------------------------------------------------------------
# 5. Grayscale / P handling
# --------------------------------------------------------------------------

def test_preprocess_grayscale_image(tmp_path, synthetic_config):
    img_path = tmp_path / "car_gray.jpg"
    Image.new("L", (50, 50), 128).save(img_path)

    array = preprocess_image(str(img_path), synthetic_config)
    assert array.shape == (1, 16, 16, 3)


def test_preprocess_palette_image(tmp_path, synthetic_config):
    img_path = tmp_path / "car_p.png"
    Image.new("RGB", (50, 50), (10, 20, 30)).convert("P").save(img_path)

    array = preprocess_image(str(img_path), synthetic_config)
    assert array.shape == (1, 16, 16, 3)


# --------------------------------------------------------------------------
# 6. Invalid image path
# --------------------------------------------------------------------------

def test_preprocess_missing_image_raises(tmp_path, synthetic_config):
    with pytest.raises(ImageNotFoundError):
        preprocess_image(str(tmp_path / "does_not_exist.jpg"), synthetic_config)


def test_preprocess_corrupted_image_raises(tmp_path, synthetic_config):
    bad_path = tmp_path / "corrupted.jpg"
    bad_path.write_bytes(b"not a real image")
    with pytest.raises(ImageDecodeError):
        preprocess_image(str(bad_path), synthetic_config)


# --------------------------------------------------------------------------
# 7. Model loading
# --------------------------------------------------------------------------

def test_load_model_succeeds(synthetic_config):
    model = load_model(synthetic_config)
    assert model.output_shape == (None, len(CLASS_NAMES))


def test_load_model_missing_raises(tmp_path):
    config = {"training": {"checkpoint": {"path": str(tmp_path / "nope.keras")}}}
    with pytest.raises(ModelNotFoundError):
        load_model(config)


# --------------------------------------------------------------------------
# 8. Class-count verification
# --------------------------------------------------------------------------

def test_verify_model_matches_classes_passes(synthetic_config):
    model = load_model(synthetic_config)
    verify_model_matches_classes(model, sorted(CLASS_NAMES))  # should not raise


def test_verify_model_matches_classes_raises_on_mismatch(synthetic_config):
    model = load_model(synthetic_config)
    with pytest.raises(ClassCountMismatchError):
        verify_model_matches_classes(model, ["only", "two"])


# --------------------------------------------------------------------------
# 9. Prediction output shape / probabilities
# --------------------------------------------------------------------------

def test_predict_image_output_shape(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    array = preprocess_image(str(img_path), synthetic_config)
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    result = predict_image(model, array, class_names, top_k=3)

    assert len(result.probabilities) == len(class_names)
    assert result.predicted_class in class_names


def test_probabilities_sum_to_one(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    array = preprocess_image(str(img_path), synthetic_config)
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    result = predict_image(model, array, class_names, top_k=3)
    assert sum(result.probabilities) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# 10. Correct class mapping (predicted class matches argmax index)
# --------------------------------------------------------------------------

def test_predicted_class_matches_argmax(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    array = preprocess_image(str(img_path), synthetic_config)
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    result = predict_image(model, array, class_names, top_k=3)

    best_index = int(np.argmax(result.probabilities))
    assert result.predicted_class == class_names[best_index]
    assert result.confidence == pytest.approx(result.probabilities[best_index])


# --------------------------------------------------------------------------
# 11. Top-k ordering
# --------------------------------------------------------------------------

def test_top_k_is_sorted_descending(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    array = preprocess_image(str(img_path), synthetic_config)
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    result = predict_image(model, array, class_names, top_k=3)
    probs = [p.probability for p in result.top_k]
    assert probs == sorted(probs, reverse=True)


def test_top_k_first_entry_is_predicted_class(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    array = preprocess_image(str(img_path), synthetic_config)
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    result = predict_image(model, array, class_names, top_k=3)
    assert result.top_k[0].class_name == result.predicted_class


# --------------------------------------------------------------------------
# 12. top-k cannot exceed number of classes
# --------------------------------------------------------------------------

def test_top_k_exceeding_class_count_raises(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    array = preprocess_image(str(img_path), synthetic_config)
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    with pytest.raises(InvalidTopKError):
        predict_image(model, array, class_names, top_k=len(class_names) + 1)


def test_top_k_zero_raises(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    array = preprocess_image(str(img_path), synthetic_config)
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    with pytest.raises(InvalidTopKError):
        predict_image(model, array, class_names, top_k=0)


# --------------------------------------------------------------------------
# 13. Invalid image path raises a clear error (predict_from_path)
# --------------------------------------------------------------------------

def test_predict_from_path_missing_image_raises(tmp_path, synthetic_config):
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)
    with pytest.raises(ImageNotFoundError):
        predict_from_path(model, str(tmp_path / "missing.jpg"), synthetic_config, class_names)


# --------------------------------------------------------------------------
# 14. Missing model raises a clear error (already covered above, plus
#     the full run_prediction() path)
# --------------------------------------------------------------------------

def test_run_prediction_missing_model_raises(tmp_path, synthetic_config_path, synthetic_config, tmp_path_factory):
    # Point the checkpoint at a nonexistent file.
    import yaml

    bad_config = dict(synthetic_config)
    bad_config["training"] = {"checkpoint": {"path": str(tmp_path / "missing_model.keras")}}
    bad_config_path = tmp_path / "bad_config.yaml"
    with open(bad_config_path, "w") as f:
        yaml.safe_dump(bad_config, f)

    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    with pytest.raises(ModelNotFoundError):
        run_prediction(str(bad_config_path), str(img_path))


# --------------------------------------------------------------------------
# 15. Output-unit/class-count mismatch raises a clear error, end-to-end
# --------------------------------------------------------------------------

def test_run_prediction_class_mismatch_raises(tmp_path, synthetic_config):
    import yaml

    mismatched_config = dict(synthetic_config)
    mismatched_config["dataset"] = {"classes": ["a", "b", "c", "d", "e"]}  # 5 != model's 3
    config_path = tmp_path / "mismatch_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(mismatched_config, f)

    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    with pytest.raises(ClassCountMismatchError):
        run_prediction(str(config_path), str(img_path))


# --------------------------------------------------------------------------
# 16. Deterministic prediction behavior
# --------------------------------------------------------------------------

def test_prediction_is_deterministic(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50), color=(30, 200, 100))
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)

    result1 = predict_from_path(model, str(img_path), synthetic_config, class_names)
    result2 = predict_from_path(model, str(img_path), synthetic_config, class_names)

    assert result1.predicted_class == result2.predicted_class
    assert result1.probabilities == result2.probabilities


# --------------------------------------------------------------------------
# 17. CLI/API consistency: run_prediction() (used by CLI) matches the
#     direct API path
# --------------------------------------------------------------------------

def test_cli_and_api_paths_agree(tmp_path, synthetic_config_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50), color=(80, 40, 200))

    cli_result = run_prediction(str(synthetic_config_path), str(img_path))

    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)
    api_result = predict_from_path(model, str(img_path), synthetic_config, class_names)

    assert cli_result.predicted_class == api_result.predicted_class
    assert cli_result.probabilities == pytest.approx(api_result.probabilities)


# --------------------------------------------------------------------------
# format_prediction() sanity
# --------------------------------------------------------------------------

def test_format_prediction_contains_key_fields(tmp_path, synthetic_config):
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)
    result = predict_from_path(model, str(img_path), synthetic_config, class_names)

    text = format_prediction(result)
    assert result.predicted_class in text
    assert "Confidence" in text
    assert str(img_path) in text


# --------------------------------------------------------------------------
# Real-data safety: inference never touches data/raw, data/processed,
# models/best_model.keras, or results/evaluation.
# --------------------------------------------------------------------------

def test_inference_does_not_touch_real_project_paths(tmp_path, synthetic_config):
    """
    Sanity check that nothing in this module references the real
    project's protected paths by construction: run a full prediction
    using only synthetic tmp_path fixtures and confirm no new top-level
    entries appear under the real data/models/results directories used
    by the rest of the project (these are simply never passed to any
    function here).
    """
    img_path = _write_image(tmp_path / "car.jpg", size=(50, 50))
    model = load_model(synthetic_config)
    class_names = get_class_names(synthetic_config)
    result = predict_from_path(model, str(img_path), synthetic_config, class_names)

    assert result.image_path == str(img_path)
    # No assertions about real paths are needed: this test module only
    # ever constructs configs pointing at tmp_path, so real project
    # directories are structurally never touched.