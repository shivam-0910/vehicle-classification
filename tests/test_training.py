"""
tests/test_training.py

Unit tests for src/training/train.py (Phase 6).

All tests use tiny synthetic datasets written to pytest's tmp_path --
never the real 33,955-image Vehicle-10 dataset. No test trains for
anywhere near 30 epochs; the "smoke test" runs exactly 1 epoch on a
handful of tiny images.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.data.augmentation import AugmentationConfig
from src.training.train import (
    build_callbacks,
    build_datasets,
    build_training_pipeline,
    compute_training_class_weights,
    detect_device,
    set_seeds,
    train_model,
)


CLASS_NAMES = ["bicycle", "boat", "car"]  # small subset, sorted order matters


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _write_tiny_image(path: Path, color: tuple, size=(16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def _make_synthetic_processed_dataset(
    root: Path,
    per_class_counts: dict,
    splits: tuple = ("train", "validation"),
) -> Path:
    """
    Build a tiny data/processed/{split}/{class}/*.jpg tree.

    per_class_counts: {class_name: count_per_split}
    """
    processed_root = root / "data" / "processed"
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for split in splits:
        for i, class_name in enumerate(CLASS_NAMES):
            count = per_class_counts.get(class_name, 4)
            for n in range(count):
                _write_tiny_image(
                    processed_root / split / class_name / f"img_{n}.jpg",
                    color=colors[i % len(colors)],
                )
    return processed_root


@pytest.fixture
def synthetic_config(tmp_path):
    """A minimal config dict pointing at a tiny synthetic dataset tree."""
    processed_root = _make_synthetic_processed_dataset(
        tmp_path,
        per_class_counts={"bicycle": 6, "boat": 3, "car": 9},  # imbalanced
    )
    return {
        "dataset": {
            "processed_root": str(processed_root),
            "classes": CLASS_NAMES,
        },
        "preprocessing": {
            "image_size": [16, 16],
            "seed": 42,
        },
        "augmentation": {
            "enabled": True,
            "rotation": {"enabled": True, "range_degrees": 5.0},
            "zoom": {"enabled": False, "range_fraction": 0.1},
            "horizontal_flip": {"enabled": True, "probability": 0.5},
            "vertical_flip": {"enabled": False, "probability": 0.5},
            "brightness": {"enabled": False, "range_fraction": 0.1},
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "early_stopping": {"enabled": True, "patience": 5},
            "reduce_lr": {"enabled": True, "patience": 2, "factor": 0.5},
            "checkpoint": {
                "enabled": True,
                "path": str(tmp_path / "models" / "best_model.keras"),
            },
            "results": {
                "metrics_dir": str(tmp_path / "results" / "metrics"),
                "plots_dir": str(tmp_path / "results" / "plots"),
            },
        },
    }


# --------------------------------------------------------------------------
# 1. Module imports successfully
# --------------------------------------------------------------------------

def test_module_imports():
    import src.training.train  # noqa: F401


# --------------------------------------------------------------------------
# 2. Class-weight calculation
# --------------------------------------------------------------------------

def test_class_weights_correct_number_of_classes(synthetic_config):
    weights = compute_training_class_weights(synthetic_config, CLASS_NAMES)
    assert set(weights.keys()) == {0, 1, 2}


def test_class_weights_are_positive(synthetic_config):
    weights = compute_training_class_weights(synthetic_config, CLASS_NAMES)
    assert all(w > 0 for w in weights.values())


def test_minority_class_receives_higher_weight(synthetic_config):
    # bicycle=6, boat=3, car=9 -> boat (index 1) is the minority class
    # and must receive the largest weight under "balanced" weighting.
    weights = compute_training_class_weights(synthetic_config, CLASS_NAMES)
    boat_idx = CLASS_NAMES.index("boat")
    car_idx = CLASS_NAMES.index("car")
    assert weights[boat_idx] > weights[car_idx]


# --------------------------------------------------------------------------
# 3. Dataset loader
# --------------------------------------------------------------------------

def test_build_datasets_shapes_and_classes(synthetic_config):
    train_ds, val_ds, class_names = build_datasets(synthetic_config, batch_size=4, seed=42)
    assert class_names == sorted(CLASS_NAMES)

    image, label = next(iter(train_ds))
    assert image.shape == (16, 16, 3)
    assert label.shape == (len(CLASS_NAMES),)


def test_training_pipeline_is_batched(synthetic_config):
    train_ds, val_ds, _ = build_training_pipeline(synthetic_config, seed=42)
    images, labels = next(iter(train_ds))
    assert images.shape[0] <= synthetic_config["training"]["batch_size"]
    assert images.shape[1:] == (16, 16, 3)


def test_validation_pipeline_is_batched(synthetic_config):
    train_ds, val_ds, _ = build_training_pipeline(synthetic_config, seed=42)
    images, labels = next(iter(val_ds))
    assert images.shape[0] <= synthetic_config["training"]["batch_size"]
    assert images.shape[1:] == (16, 16, 3)


# --------------------------------------------------------------------------
# 4. Training configuration loads correctly
# --------------------------------------------------------------------------

def test_training_config_values_present(synthetic_config):
    train_cfg = synthetic_config["training"]
    assert train_cfg["epochs"] == 1
    assert train_cfg["batch_size"] == 4
    assert train_cfg["early_stopping"]["enabled"] is True
    assert train_cfg["reduce_lr"]["enabled"] is True
    assert train_cfg["checkpoint"]["enabled"] is True


# --------------------------------------------------------------------------
# 5. Callback construction
# --------------------------------------------------------------------------

def test_callbacks_are_constructed(synthetic_config):
    callbacks = build_callbacks(synthetic_config)
    types = [type(cb).__name__ for cb in callbacks]
    assert "ModelCheckpoint" in types
    assert "EarlyStopping" in types
    assert "ReduceLROnPlateau" in types


def test_callbacks_respect_disabled_flags(synthetic_config):
    synthetic_config["training"]["early_stopping"]["enabled"] = False
    callbacks = build_callbacks(synthetic_config)
    types = [type(cb).__name__ for cb in callbacks]
    assert "EarlyStopping" not in types


# --------------------------------------------------------------------------
# 6. Tiny one-epoch training smoke test
# --------------------------------------------------------------------------

def test_one_epoch_training_smoke_test(synthetic_config):
    history = train_model(synthetic_config, epochs=1, save_artifacts=True)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1

    # Artifacts were written where configured.
    metrics_dir = Path(synthetic_config["training"]["results"]["metrics_dir"])
    history_path = metrics_dir / "training_history.json"
    assert history_path.exists()
    with open(history_path) as f:
        saved = json.load(f)
    assert "loss" in saved

    plots_dir = Path(synthetic_config["training"]["results"]["plots_dir"])
    assert (plots_dir / "training_accuracy.png").exists()
    assert (plots_dir / "training_loss.png").exists()

    checkpoint_path = Path(synthetic_config["training"]["checkpoint"]["path"])
    assert checkpoint_path.exists()


# --------------------------------------------------------------------------
# 7. Test split is never used by train.py
# --------------------------------------------------------------------------

def test_test_split_directory_is_never_loaded(tmp_path, synthetic_config):
    """
    Add a 'test' split directory containing a class folder with a
    single, obviously-out-of-place large image and a bogus class name
    that isn't in CLASS_NAMES. If train.py ever touched the test split,
    building the datasets would either error (unknown class folder) or
    the extra images would show up in the pipeline. Neither happens
    because build_datasets() and build_training_pipeline() only ever
    look at 'train' and 'validation'.
    """
    processed_root = Path(synthetic_config["dataset"]["processed_root"])
    _write_tiny_image(
        processed_root / "test" / "not_a_real_class" / "poison.jpg",
        color=(1, 2, 3),
    )

    # Should not raise despite the malformed 'test' directory, and
    # should not pick up the bogus class.
    train_ds, val_ds, class_names = build_datasets(synthetic_config, batch_size=4, seed=42)
    assert "not_a_real_class" not in class_names
    assert class_names == sorted(CLASS_NAMES)


# --------------------------------------------------------------------------
# 8. Class ordering matches model output ordering
# --------------------------------------------------------------------------

def test_class_ordering_matches_class_weight_indices(synthetic_config):
    train_ds, val_ds, class_names = build_datasets(synthetic_config, batch_size=4, seed=42)
    weights = compute_training_class_weights(synthetic_config, class_names)

    # class_names is alphabetically sorted by image_dataset_from_directory,
    # which is also how the model's softmax output is ordered (index i
    # corresponds to class_names[i]). Class weight keys must be indices
    # into this exact list.
    assert list(range(len(class_names))) == sorted(weights.keys())


def test_model_output_units_match_class_count(synthetic_config):
    from src.models.cnn_model import build_model

    _, _, class_names = build_datasets(synthetic_config, batch_size=4, seed=42)
    model = build_model(input_shape=(16, 16, 3), num_classes=len(class_names))
    assert model.output_shape == (None, len(class_names))


# --------------------------------------------------------------------------
# Misc: device detection, seeding
# --------------------------------------------------------------------------

def test_detect_device_returns_cpu_or_gpu():
    assert detect_device() in ("CPU", "GPU")


def test_set_seeds_does_not_raise():
    set_seeds(42)