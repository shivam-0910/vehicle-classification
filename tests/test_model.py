"""
tests/test_model.py

Unit tests for src/models/cnn_model.py (Phase 5).

These tests use only synthetic NumPy data and never touch the real
Vehicle-10 dataset (no path to D:\\ml-datasets\\vehicle-10 is referenced
anywhere here). They are intentionally fast: a couple of tiny forward
passes and one minimal training step on a handful of random samples.
"""
from __future__ import annotations

import numpy as np
import pytest
from tensorflow import keras

from src.models.cnn_model import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    build_and_compile_model,
    build_model,
    compile_model,
    input_shape_from_config,
    num_classes_from_config,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def model():
    """A freshly built (uncompiled) model with default architecture."""
    return build_model()


@pytest.fixture
def compiled_model():
    """A freshly built and compiled model with default architecture."""
    return build_and_compile_model()


@pytest.fixture
def synthetic_batch():
    """A synthetic batch of 2 random 128x128x3 images and one-hot labels."""
    rng = np.random.default_rng(seed=0)
    x = rng.random((2, 128, 128, 3), dtype=np.float32)
    labels = rng.integers(0, DEFAULT_NUM_CLASSES, size=2)
    y = keras.utils.to_categorical(labels, num_classes=DEFAULT_NUM_CLASSES)
    return x, y


# --------------------------------------------------------------------------
# 1. Construction
# --------------------------------------------------------------------------

def test_model_builds_successfully(model):
    assert isinstance(model, keras.Model)


# --------------------------------------------------------------------------
# 2. Input shape
# --------------------------------------------------------------------------

def test_input_shape_is_correct(model):
    assert model.input_shape == (None, 128, 128, 3)


# --------------------------------------------------------------------------
# 3. Output shape
# --------------------------------------------------------------------------

def test_output_shape_is_correct(model):
    assert model.output_shape == (None, DEFAULT_NUM_CLASSES)


# --------------------------------------------------------------------------
# 4. Final activation is softmax
# --------------------------------------------------------------------------

def test_final_layer_activation_is_softmax(model):
    final_layer = model.layers[-1]
    activation = final_layer.get_config()["activation"]
    assert activation == "softmax"


# --------------------------------------------------------------------------
# 5-8. Required layer types are present
# --------------------------------------------------------------------------

def _layer_types(model):
    return [type(layer).__name__ for layer in model.layers]


def test_model_contains_conv2d_layers(model):
    assert "Conv2D" in _layer_types(model)


def test_model_contains_maxpooling2d_layers(model):
    assert "MaxPooling2D" in _layer_types(model)


def test_model_contains_dropout_layers(model):
    assert "Dropout" in _layer_types(model)


def test_model_contains_dense_layers(model):
    assert "Dense" in _layer_types(model)


# --------------------------------------------------------------------------
# 9. Compilation
# --------------------------------------------------------------------------

def test_model_compiles_with_adam_and_categorical_crossentropy(model):
    compile_model(model)
    assert model.optimizer is not None
    assert "adam" in type(model.optimizer).__name__.lower()

    loss = model.loss
    loss_name = loss if isinstance(loss, str) else getattr(loss, "name", str(loss))
    assert "categorical_crossentropy" in loss_name.replace("-", "_")

    # Metrics are stored differently across Keras versions (metrics_names
    # vs. a nested compile_metrics wrapper), so inspect the model config,
    # which reliably reflects what was passed to compile().
    compiled_config = model.get_compile_config()
    metrics_config = str(compiled_config.get("metrics"))
    assert "accuracy" in metrics_config


# --------------------------------------------------------------------------
# 10-11. Forward pass on a synthetic batch
# --------------------------------------------------------------------------

def test_forward_pass_on_synthetic_batch(compiled_model, synthetic_batch):
    x, _ = synthetic_batch
    predictions = compiled_model.predict(x, verbose=0)
    assert predictions.shape == (2, DEFAULT_NUM_CLASSES)


# --------------------------------------------------------------------------
# 12. Softmax outputs sum to ~1
# --------------------------------------------------------------------------

def test_predictions_sum_to_one(compiled_model, synthetic_batch):
    x, _ = synthetic_batch
    predictions = compiled_model.predict(x, verbose=0)
    sums = predictions.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(2), atol=1e-5)


# --------------------------------------------------------------------------
# 13. One minimal training step on synthetic data
# --------------------------------------------------------------------------

def test_minimal_training_step_does_not_crash(compiled_model, synthetic_batch):
    x, y = synthetic_batch
    history = compiled_model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    assert "loss" in history.history


# --------------------------------------------------------------------------
# Config-driven helpers
# --------------------------------------------------------------------------

def test_num_classes_from_config_uses_dataset_classes():
    config = {"dataset": {"classes": ["a", "b", "c"]}}
    assert num_classes_from_config(config) == 3


def test_num_classes_from_config_falls_back_to_default_when_missing():
    assert num_classes_from_config({}) == DEFAULT_NUM_CLASSES


def test_input_shape_from_config_uses_preprocessing_image_size():
    config = {"preprocessing": {"image_size": [128, 128]}}
    assert input_shape_from_config(config) == (128, 128, 3)


def test_input_shape_from_config_falls_back_to_default_when_missing():
    assert input_shape_from_config({}) == DEFAULT_INPUT_SHAPE


def test_build_model_respects_config_derived_arguments():
    config = {
        "dataset": {"classes": ["bicycle", "boat", "bus", "car"]},
        "preprocessing": {"image_size": [64, 64]},
    }
    m = build_model(
        input_shape=input_shape_from_config(config),
        num_classes=num_classes_from_config(config),
    )
    assert m.input_shape == (None, 64, 64, 3)
    assert m.output_shape == (None, 4)