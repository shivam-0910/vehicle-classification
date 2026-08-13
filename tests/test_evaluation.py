"""
tests/test_evaluation.py

Unit tests for src/evaluation/evaluate.py (Phase 7).

All tests use tiny synthetic datasets and a tiny synthetic model
written to pytest's tmp_path -- never the real Vehicle-10 dataset or
the real trained checkpoint. No test path references
D:\\ml-datasets\\vehicle-10 or the real models/best_model.keras.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.evaluation.evaluate import (
    build_test_dataset,
    collect_predictions,
    compute_metrics,
    evaluate_model,
    load_trained_model,
    save_classification_report_csv,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    save_per_class_metrics_csv,
    save_test_metrics_json,
    verify_class_ordering,
)
from src.models.cnn_model import build_and_compile_model


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
    splits: tuple = ("train", "validation", "test"),
) -> Path:
    """Build a tiny data/processed/{split}/{class}/*.jpg tree."""
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
    """A minimal config dict pointing at a tiny synthetic dataset tree
    and a tiny freshly-trained (not the real) checkpoint."""
    processed_root = _make_synthetic_processed_dataset(
        tmp_path,
        per_class_counts={"bicycle": 5, "boat": 4, "car": 6},
    )

    # Build + save a tiny synthetic "trained" model -- never touches the
    # real models/best_model.keras.
    model = build_and_compile_model(input_shape=(16, 16, 3), num_classes=len(CLASS_NAMES))
    checkpoint_path = tmp_path / "models" / "best_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)

    return {
        "dataset": {
            "processed_root": str(processed_root),
            "classes": CLASS_NAMES,
        },
        "preprocessing": {
            "image_size": [16, 16],
            "seed": 42,
        },
        "training": {
            "batch_size": 4,
            "checkpoint": {
                "enabled": True,
                "path": str(checkpoint_path),
            },
        },
        "evaluation": {
            "output_dir": str(tmp_path / "results" / "evaluation"),
        },
    }


# --------------------------------------------------------------------------
# 1. Module imports successfully
# --------------------------------------------------------------------------

def test_module_imports():
    import src.evaluation.evaluate  # noqa: F401


# --------------------------------------------------------------------------
# 2. Test dataset can be loaded, and ONLY the test split
# --------------------------------------------------------------------------

def test_build_test_dataset_shapes_and_classes(synthetic_config):
    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    assert class_names == sorted(CLASS_NAMES)

    images, labels = next(iter(test_ds))
    assert images.shape[1:] == (16, 16, 3)
    assert labels.shape[1] == len(CLASS_NAMES)


def test_build_test_dataset_does_not_use_train_or_validation(tmp_path, synthetic_config):
    """
    Poison the train/ and validation/ dirs with an extra bogus class
    directory. If evaluate.py ever read those splits, the bogus class
    would appear in class_names or images would be over-counted.
    """
    processed_root = Path(synthetic_config["dataset"]["processed_root"])
    _write_tiny_image(
        processed_root / "train" / "not_a_real_class" / "poison.jpg",
        color=(1, 2, 3),
    )
    _write_tiny_image(
        processed_root / "validation" / "also_not_real" / "poison.jpg",
        color=(4, 5, 6),
    )

    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    assert class_names == sorted(CLASS_NAMES)
    assert "not_a_real_class" not in class_names
    assert "also_not_real" not in class_names


def test_build_test_dataset_missing_dir_raises(tmp_path):
    config = {
        "dataset": {"processed_root": str(tmp_path / "nonexistent"), "classes": CLASS_NAMES},
        "preprocessing": {"image_size": [16, 16]},
        "training": {"batch_size": 4},
    }
    with pytest.raises(FileNotFoundError):
        build_test_dataset(config)


def test_test_dataset_images_are_not_shuffled_or_augmented(synthetic_config):
    """Calling build_test_dataset twice must yield identical label order
    (no shuffling), which augmentation/shuffling would break."""
    ds1, _ = build_test_dataset(synthetic_config, batch_size=1000)
    ds2, _ = build_test_dataset(synthetic_config, batch_size=1000)
    labels1 = np.concatenate([y.numpy() for _, y in ds1], axis=0)
    labels2 = np.concatenate([y.numpy() for _, y in ds2], axis=0)
    np.testing.assert_array_equal(labels1, labels2)


# --------------------------------------------------------------------------
# 3. Model can be loaded (not retrained)
# --------------------------------------------------------------------------

def test_load_trained_model(synthetic_config):
    model = load_trained_model(synthetic_config)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, len(CLASS_NAMES))


def test_load_trained_model_missing_raises(tmp_path):
    config = {"training": {"checkpoint": {"path": str(tmp_path / "nope.keras")}}}
    with pytest.raises(FileNotFoundError):
        load_trained_model(config)


# --------------------------------------------------------------------------
# 4. Class ordering verification
# --------------------------------------------------------------------------

def test_verify_class_ordering_passes_for_matching_model(synthetic_config):
    model = load_trained_model(synthetic_config)
    # Should not raise.
    verify_class_ordering(model, sorted(CLASS_NAMES), synthetic_config)


def test_verify_class_ordering_raises_on_unit_mismatch(synthetic_config):
    model = load_trained_model(synthetic_config)
    with pytest.raises(ValueError):
        verify_class_ordering(model, ["only", "two"], synthetic_config)


# --------------------------------------------------------------------------
# 5-6. Predictions have expected shape; probabilities sum to ~1
# --------------------------------------------------------------------------

def test_predictions_have_expected_shape(synthetic_config):
    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    model = load_trained_model(synthetic_config)
    y_true, y_pred, y_proba = collect_predictions(model, test_ds)

    num_test_images = sum(1 for _ in Path(synthetic_config["dataset"]["processed_root"], "test").rglob("*.jpg"))
    assert y_true.shape == (num_test_images,)
    assert y_pred.shape == (num_test_images,)
    assert y_proba.shape == (num_test_images, len(class_names))


def test_probabilities_sum_to_one(synthetic_config):
    test_ds, _ = build_test_dataset(synthetic_config, batch_size=4)
    model = load_trained_model(synthetic_config)
    _, _, y_proba = collect_predictions(model, test_ds)
    sums = y_proba.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(len(sums)), atol=1e-4)


# --------------------------------------------------------------------------
# 7. Metrics are generated correctly
# --------------------------------------------------------------------------

def test_compute_metrics_contains_all_classes(synthetic_config):
    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    model = load_trained_model(synthetic_config)
    y_true, y_pred, _ = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names)

    assert set(metrics["per_class_metrics"].keys()) == set(class_names)
    for cls_metrics in metrics["per_class_metrics"].values():
        assert set(cls_metrics.keys()) == {"precision", "recall", "f1_score", "support"}


def test_classification_report_contains_all_classes_and_averages(synthetic_config):
    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    model = load_trained_model(synthetic_config)
    y_true, y_pred, _ = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names)

    report = metrics["report_dict"]
    for cls in class_names:
        assert cls in report
    assert "macro avg" in report
    assert "weighted avg" in report
    assert "accuracy" in report


def test_confusion_matrix_shape(synthetic_config):
    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    model = load_trained_model(synthetic_config)
    y_true, y_pred, _ = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names)

    cm = metrics["confusion_matrix"]
    assert cm.shape == (len(class_names), len(class_names))
    # Every test example is counted exactly once.
    assert cm.sum() == len(y_true)


def test_support_counts_match_class_counts(synthetic_config):
    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    model = load_trained_model(synthetic_config)
    y_true, y_pred, _ = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names)

    expected_counts = {"bicycle": 5, "boat": 4, "car": 6}
    for cls, expected in expected_counts.items():
        assert metrics["per_class_metrics"][cls]["support"] == expected


# --------------------------------------------------------------------------
# 8. Full pipeline + output files are created
# --------------------------------------------------------------------------

def test_evaluate_model_end_to_end_creates_outputs(synthetic_config):
    result = evaluate_model(synthetic_config, save_artifacts=True)

    assert "test_loss" in result
    assert "test_accuracy" in result
    assert result["class_names"] == sorted(CLASS_NAMES)

    output_dir = Path(synthetic_config["evaluation"]["output_dir"])
    assert (output_dir / "test_metrics.json").exists()
    assert (output_dir / "classification_report.csv").exists()
    assert (output_dir / "confusion_matrix.csv").exists()
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "per_class_metrics.csv").exists()

    with open(output_dir / "test_metrics.json") as f:
        saved = json.load(f)
    assert saved["class_names"] == sorted(CLASS_NAMES)
    assert saved["num_classes"] == len(CLASS_NAMES)
    assert saved["num_test_images"] == 15  # 5 + 4 + 6
    for key in (
        "test_loss", "test_accuracy", "per_class_metrics",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    ):
        assert key in saved


def test_evaluate_model_without_save_artifacts_writes_nothing(synthetic_config):
    evaluate_model(synthetic_config, save_artifacts=False)
    output_dir = Path(synthetic_config["evaluation"]["output_dir"])
    assert not output_dir.exists()


# --------------------------------------------------------------------------
# 9. Individual save_* helpers work standalone
# --------------------------------------------------------------------------

def test_save_helpers_write_expected_files(tmp_path, synthetic_config):
    test_ds, class_names = build_test_dataset(synthetic_config, batch_size=4)
    model = load_trained_model(synthetic_config)
    y_true, y_pred, _ = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names)

    out_dir = tmp_path / "custom_eval_dir"
    p1 = save_test_metrics_json(0.5, 0.8, class_names, metrics, out_dir)
    p2 = save_classification_report_csv(metrics, out_dir)
    p3 = save_per_class_metrics_csv(metrics, out_dir)
    p4 = save_confusion_matrix_csv(metrics, class_names, out_dir)
    p5 = save_confusion_matrix_plot(metrics, class_names, out_dir)

    for p in (p1, p2, p3, p4, p5):
        assert p.exists()


# --------------------------------------------------------------------------
# 10. Determinism: repeated evaluation on the same model/data agrees
# --------------------------------------------------------------------------

def test_evaluation_is_deterministic(synthetic_config):
    result1 = evaluate_model(synthetic_config, save_artifacts=False)
    result2 = evaluate_model(synthetic_config, save_artifacts=False)

    assert result1["test_accuracy"] == pytest.approx(result2["test_accuracy"])
    np.testing.assert_array_equal(result1["y_pred"], result2["y_pred"])
    np.testing.assert_array_equal(result1["y_true"], result2["y_true"])


# --------------------------------------------------------------------------
# 11. Evaluation does not modify the test dataset on disk
# --------------------------------------------------------------------------

def test_evaluation_does_not_modify_test_images(synthetic_config):
    test_dir = Path(synthetic_config["dataset"]["processed_root"]) / "test"
    before = {
        f: (f.stat().st_size, f.stat().st_mtime)
        for f in test_dir.rglob("*.jpg")
    }

    evaluate_model(synthetic_config, save_artifacts=True)

    after = {
        f: (f.stat().st_size, f.stat().st_mtime)
        for f in test_dir.rglob("*.jpg")
    }
    assert before == after