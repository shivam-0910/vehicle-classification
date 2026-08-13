"""
tests/test_app.py

Unit tests for the Phase 10 Streamlit deployment layer (app.py).

These tests exercise app.py's testable helper functions and its
integration with the existing src.inference.predict module directly
(no browser / no running Streamlit server required). All tests use
synthetic tmp_path fixtures and a tiny synthetic model -- never the
real Vehicle-10 dataset or the real models/best_model.keras.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import app
from src.inference.predict import InferenceError, PredictionResult
from src.models.cnn_model import build_and_compile_model


CLASS_NAMES = ["bicycle", "boat", "car"]  # small subset, sorted order matters


# --------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------

def _write_image_bytes(size=(50, 50), mode="RGB", color=(255, 0, 0)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_config(tmp_path):
    """A minimal config dict + on-disk checkpoint pointing at a tiny
    synthetic model -- never the real models/best_model.keras."""
    model = build_and_compile_model(input_shape=(16, 16, 3), num_classes=len(CLASS_NAMES))
    checkpoint_path = tmp_path / "models" / "tiny_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)

    return {
        "dataset": {"classes": list(CLASS_NAMES)},
        "preprocessing": {"image_size": [16, 16]},
        "training": {"checkpoint": {"enabled": True, "path": str(checkpoint_path)}},
    }


@pytest.fixture
def synthetic_config_path(tmp_path, synthetic_config):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(synthetic_config, f)
    return config_path


# --------------------------------------------------------------------------
# 1. app module imports successfully
# --------------------------------------------------------------------------

def test_app_module_imports():
    import app  # noqa: F401


# --------------------------------------------------------------------------
# 2. configuration can be loaded (via the existing loader, through app's
#    cached helper -- we call the inner logic directly, bypassing the
#    st.cache_resource wrapper, since that requires a Streamlit runtime)
# --------------------------------------------------------------------------

def test_config_and_model_load_via_existing_loader(synthetic_config_path):
    # get_config_and_model is decorated with st.cache_resource; call the
    # underlying function to exercise the real logic without a live
    # Streamlit script run.
    config, class_names, model = app.get_config_and_model.__wrapped__(
        str(synthetic_config_path)
    )
    assert class_names == sorted(CLASS_NAMES)
    assert model.output_shape == (None, len(CLASS_NAMES))


def test_config_loading_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        app.get_config_and_model.__wrapped__(str(tmp_path / "nope.yaml"))


# --------------------------------------------------------------------------
# 3. required deployment functions exist
# --------------------------------------------------------------------------

def test_required_functions_exist():
    assert callable(app.render_app)
    assert callable(app.get_config_and_model)
    assert callable(app.run_inference_on_bytes)
    assert callable(app.is_supported_image_filename)


# --------------------------------------------------------------------------
# 4. inference integration uses the existing inference module (not a
#    duplicate implementation)
# --------------------------------------------------------------------------

def test_app_imports_predict_module_functions_not_duplicates():
    import src.inference.predict as predict_module

    # app.preprocess_image and app.predict_image must be the SAME
    # function objects as in src.inference.predict -- proof app.py
    # does not redefine/duplicate preprocessing or prediction logic.
    assert app.preprocess_image is predict_module.preprocess_image
    assert app.predict_image is predict_module.predict_image
    assert app.get_class_names is predict_module.get_class_names
    assert app.load_model is predict_module.load_model


def test_run_inference_on_bytes_uses_existing_pipeline(synthetic_config):
    from src.inference.predict import get_class_names, load_model

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)
    image_bytes = _write_image_bytes(color=(10, 20, 30))

    result = app.run_inference_on_bytes(
        image_bytes, "car.png", synthetic_config, class_names, model, top_k=3
    )

    assert isinstance(result, PredictionResult)
    assert result.predicted_class in class_names
    assert len(result.top_k) == 3
    assert sum(result.probabilities) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# 5. supported image types are accepted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ["car.jpg", "car.jpeg", "car.png", "CAR.JPG"])
def test_supported_extensions_accepted(filename):
    assert app.is_supported_image_filename(filename) is True


@pytest.mark.parametrize("filename", ["car.gif", "car.bmp", "car.txt", "car", ""])
def test_unsupported_extensions_rejected(filename):
    assert app.is_supported_image_filename(filename) is False


# --------------------------------------------------------------------------
# 6. invalid image handling works
# --------------------------------------------------------------------------

def test_run_inference_on_corrupted_bytes_raises_inference_error(synthetic_config):
    from src.inference.predict import get_class_names, load_model

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)

    with pytest.raises(InferenceError):
        app.run_inference_on_bytes(
            b"not a real image", "bad.jpg", synthetic_config, class_names, model
        )


# --------------------------------------------------------------------------
# 7. prediction result is displayed/constructed correctly
# --------------------------------------------------------------------------

def test_prediction_result_has_expected_fields(synthetic_config):
    from src.inference.predict import get_class_names, load_model

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)
    image_bytes = _write_image_bytes(color=(200, 50, 10))

    result = app.run_inference_on_bytes(
        image_bytes, "truck.png", synthetic_config, class_names, model, top_k=3
    )

    assert result.image_path == "truck.png"
    assert 0.0 <= result.confidence <= 1.0
    top_probs = [p.probability for p in result.top_k]
    assert top_probs == sorted(top_probs, reverse=True)
    assert result.top_k[0].class_name == result.predicted_class


# --------------------------------------------------------------------------
# 8. model is not retrained by the application
# --------------------------------------------------------------------------

def test_app_never_calls_fit_or_compile(tmp_path):
    """Static check: app.py's source must not call model.fit(),
    model.compile(), or any training-pipeline functions."""
    source = inspect.getsource(app)
    assert ".fit(" not in source
    assert "build_and_compile_model" not in source
    assert "build_model(" not in source
    assert "compile_model(" not in source


def test_run_inference_does_not_change_model_weights(synthetic_config):
    from src.inference.predict import get_class_names, load_model

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)
    weights_before = [w.copy() for w in model.get_weights()]

    image_bytes = _write_image_bytes(color=(60, 90, 120))
    app.run_inference_on_bytes(
        image_bytes, "car.png", synthetic_config, class_names, model, top_k=3
    )

    weights_after = model.get_weights()
    for before, after in zip(weights_before, weights_after):
        np.testing.assert_array_equal(before, after)


# --------------------------------------------------------------------------
# 9. application does not write into data/raw
# --------------------------------------------------------------------------

def test_run_inference_does_not_write_to_data_raw(tmp_path, synthetic_config, monkeypatch):
    from src.inference.predict import get_class_names, load_model

    monkeypatch.chdir(tmp_path)
    data_raw = tmp_path / "data" / "raw"

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)
    image_bytes = _write_image_bytes()

    app.run_inference_on_bytes(
        image_bytes, "car.png", synthetic_config, class_names, model, top_k=3
    )

    assert not data_raw.exists()


# --------------------------------------------------------------------------
# 10. application does not write into data/processed
# --------------------------------------------------------------------------

def test_run_inference_does_not_write_to_data_processed(tmp_path, synthetic_config, monkeypatch):
    from src.inference.predict import get_class_names, load_model

    monkeypatch.chdir(tmp_path)
    data_processed = tmp_path / "data" / "processed"

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)
    image_bytes = _write_image_bytes()

    app.run_inference_on_bytes(
        image_bytes, "car.png", synthetic_config, class_names, model, top_k=3
    )

    assert not data_processed.exists()


# --------------------------------------------------------------------------
# 11. application does not modify models/best_model.keras (the configured
#     checkpoint file's bytes are unchanged after a prediction)
# --------------------------------------------------------------------------

def test_run_inference_does_not_modify_checkpoint_file(synthetic_config):
    from src.inference.predict import get_class_names, load_model

    checkpoint_path = Path(synthetic_config["training"]["checkpoint"]["path"])
    bytes_before = checkpoint_path.read_bytes()

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)
    image_bytes = _write_image_bytes()
    app.run_inference_on_bytes(
        image_bytes, "car.png", synthetic_config, class_names, model, top_k=3
    )

    bytes_after = checkpoint_path.read_bytes()
    assert bytes_before == bytes_after


# --------------------------------------------------------------------------
# Extra: temp files used for in-memory bridging are cleaned up
# --------------------------------------------------------------------------

def test_run_inference_cleans_up_temp_file(synthetic_config, tmp_path, monkeypatch):
    from src.inference.predict import get_class_names, load_model

    created_paths = []
    real_named_temp = tempfile_module_ref = __import__("tempfile").NamedTemporaryFile

    class _TrackingTempFile:
        def __init__(self, *args, **kwargs):
            self._inner = real_named_temp(*args, **kwargs)
            created_paths.append(self._inner.name)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    monkeypatch.setattr(app.tempfile, "NamedTemporaryFile", _TrackingTempFile)

    class_names = get_class_names(synthetic_config)
    model = load_model(synthetic_config)
    image_bytes = _write_image_bytes()
    app.run_inference_on_bytes(
        image_bytes, "car.png", synthetic_config, class_names, model, top_k=3
    )

    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists()