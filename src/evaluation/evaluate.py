"""
src/evaluation/evaluate.py

Model evaluation pipeline for the Vehicle-10 CNN (Phase 7).

This module evaluates the ALREADY-TRAINED model checkpoint
(models/best_model.keras, from Phase 6) on the held-out test split
ONLY:

    data/processed/test/

It does NOT retrain, redesign the CNN, modify preprocessing, or
modify augmentation. It reuses:

  - src/data/loader.py        -> load_config()
  - src/models/cnn_model.py   -> input_shape_from_config(),
                                  num_classes_from_config()
  - src/config/config.yaml    -> dataset/preprocessing/training config
                                  (no values duplicated here)

High-level flow (see evaluate_model()):

  1. Load config.yaml.
  2. Build a tf.data pipeline over data/processed/test using
     image_dataset_from_directory -- the SAME function and image-size
     convention train.py uses for train/validation, so preprocessing
     (resize/normalize) is identical. NOT shuffled, NOT augmented.
  3. Load the trained model from models/best_model.keras (no
     compile-time side effects on weights).
  4. Verify the model's output class ordering matches the test
     dataset's inferred class ordering (both come from sorted
     subfolder names -- the same convention used throughout this
     project) before computing anything.
  5. Run model.evaluate() for loss/accuracy, and model.predict() for
     per-class metrics, confusion matrix, and classification report
     (via sklearn).
  6. Save results to results/evaluation/.

Nothing here mutates data/processed/test, data/processed/train,
data/processed/validation, or the raw dataset. Only files under
results/evaluation/ are written.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data.loader import load_config
from src.models.cnn_model import input_shape_from_config, num_classes_from_config

logger = logging.getLogger("vehicle10.evaluation")


# --------------------------------------------------------------------------
# Paths (derived from config, never hardcoded)
# --------------------------------------------------------------------------

def _processed_test_dir(config: dict) -> Path:
    """Path to data/processed/test (never train/ or validation/)."""
    processed_root = Path(config["dataset"]["processed_root"])
    return processed_root / "test"


def _model_path(config: dict) -> Path:
    """Path to the trained checkpoint, from training.checkpoint.path."""
    checkpoint_cfg = config.get("training", {}).get("checkpoint", {})
    return Path(checkpoint_cfg.get("path", "models/best_model.keras"))


def _evaluation_output_dir(config: dict) -> Path:
    """
    Where evaluation artifacts are written. Mirrors the
    training.results.{metrics_dir,plots_dir} convention but is its own
    section so Phase 6 outputs are never overwritten; falls back to
    'results/evaluation' if not present in config.
    """
    eval_cfg = config.get("evaluation", {})
    return Path(eval_cfg.get("output_dir", "results/evaluation"))


# --------------------------------------------------------------------------
# Dataset loading (test split ONLY)
# --------------------------------------------------------------------------

def build_test_dataset(
    config: dict,
    batch_size: Optional[int] = None,
) -> Tuple[tf.data.Dataset, List[str]]:
    """
    Build the test tf.data pipeline from data/processed/test ONLY.

    Uses image_dataset_from_directory with the same image_size
    convention as train.py's build_datasets() (preprocessing.image_size
    is [width, height] in config; Keras wants (height, width)), so the
    same resize/normalization is applied. NOT shuffled, NOT augmented
    -- this module never imports src.data.augmentation.

    Returns:
        (test_ds, class_names)

        - test_ds: batched, normalized to float32 [0, 1], NOT shuffled,
          NOT augmented, prefetched.
        - class_names: sorted list of class subfolder names under
          data/processed/test, i.e. the ordering Keras assigned integer
          labels 0..N-1 with. This is the authoritative class ordering
          for this evaluation run.
    """
    pp_cfg = config["preprocessing"]
    train_cfg = config.get("training", {})

    image_size = tuple(pp_cfg["image_size"])  # (width, height) in config
    target_size = (image_size[1], image_size[0])
    batch_size = batch_size or train_cfg.get("batch_size", 32)

    test_dir = _processed_test_dir(config)
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test split not found at {test_dir}. Run preprocessing "
            f"(src/data/preprocess.py) before evaluation."
        )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=target_size,
        batch_size=None,  # batch after normalization, for consistency
        shuffle=False,
    )

    class_names = test_ds.class_names

    def _normalize(image: tf.Tensor, label: tf.Tensor):
        return tf.cast(image, tf.float32) / 255.0, label

    test_ds = test_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return test_ds, class_names


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_trained_model(config: dict) -> tf.keras.Model:
    """
    Load the already-trained model checkpoint (models/best_model.keras
    by default, or training.checkpoint.path from config). Does NOT
    retrain or recompile with different settings -- compile=True so
    the saved optimizer/loss/metrics state is preserved for
    model.evaluate().
    """
    model_path = _model_path(config)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at {model_path}. "
            f"Run training (src/training/train.py) before evaluation."
        )
    return tf.keras.models.load_model(model_path)


def verify_class_ordering(model: tf.keras.Model, class_names: List[str], config: dict) -> None:
    """
    Verify the model's output layer size matches the test dataset's
    number of classes. Class ORDER for a Keras Dense-softmax model is
    implicit (label index -> output unit), so the concrete guarantee
    we can check is that the unit count agrees with len(class_names);
    the ordering itself is guaranteed by both train.py and this module
    using the identical convention (image_dataset_from_directory's
    sorted subfolder order). Raises if the unit count disagrees, since
    that would silently misalign every prediction.
    """
    output_units = model.output_shape[-1]
    if output_units != len(class_names):
        raise ValueError(
            f"Model output has {output_units} units but the test dataset "
            f"has {len(class_names)} classes ({class_names}). Class "
            f"ordering/count mismatch would silently corrupt evaluation "
            f"results."
        )

    config_classes = config.get("dataset", {}).get("classes")
    if config_classes and sorted(config_classes) != sorted(class_names):
        logger.warning(
            "config.yaml dataset.classes (%s) does not match the class "
            "folders found under data/processed/test (%s); using the "
            "on-disk test class list as authoritative, matching "
            "train.py's convention.",
            sorted(config_classes), sorted(class_names),
        )


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

def collect_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run predictions across the full test dataset.

    Returns:
        (y_true, y_pred, y_proba)
        - y_true: (N,) integer class indices (from one-hot labels).
        - y_pred: (N,) integer predicted class indices (argmax).
        - y_proba: (N, num_classes) predicted probabilities.
    """
    y_true_parts: List[np.ndarray] = []
    y_proba_parts: List[np.ndarray] = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_proba_parts.append(probs)
        y_true_parts.append(labels.numpy())

    y_true_onehot = np.concatenate(y_true_parts, axis=0)
    y_proba = np.concatenate(y_proba_parts, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_proba, axis=1)

    return y_true, y_pred, y_proba


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute per-class precision/recall/F1/support, macro and weighted
    averages, overall accuracy, the classification report (as a dict,
    for CSV export), and the confusion matrix.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class_metrics = {
        class_name: {
            "precision": report_dict[class_name]["precision"],
            "recall": report_dict[class_name]["recall"],
            "f1_score": report_dict[class_name]["f1-score"],
            "support": report_dict[class_name]["support"],
        }
        for class_name in class_names
    }

    return {
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "accuracy": report_dict["accuracy"],
    }


# --------------------------------------------------------------------------
# Output artifacts
# --------------------------------------------------------------------------

def save_test_metrics_json(
    test_loss: float,
    test_accuracy: float,
    class_names: List[str],
    metrics: Dict,
    output_dir: Path,
) -> Path:
    """Save the machine-readable metrics JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_metrics.json"

    payload = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "num_test_images": int(sum(m["support"] for m in metrics["per_class_metrics"].values())),
        "num_classes": len(class_names),
        "class_names": class_names,
        "per_class_metrics": metrics["per_class_metrics"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_classification_report_csv(metrics: Dict, output_dir: Path) -> Path:
    """Save the full sklearn classification report as a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "classification_report.csv"
    df = pd.DataFrame(metrics["report_dict"]).transpose()
    df.to_csv(path)
    return path


def save_per_class_metrics_csv(metrics: Dict, output_dir: Path) -> Path:
    """Save a compact per-class precision/recall/F1/support CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "per_class_metrics.csv"
    df = pd.DataFrame(metrics["per_class_metrics"]).transpose()
    df.index.name = "class"
    df.to_csv(path)
    return path


def save_confusion_matrix_csv(
    metrics: Dict,
    class_names: List[str],
    output_dir: Path,
) -> Path:
    """Save the confusion matrix as a labeled CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "confusion_matrix.csv"
    df = pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names)
    df.to_csv(path)
    return path


def save_confusion_matrix_plot(
    metrics: Dict,
    class_names: List[str],
    output_dir: Path,
) -> Path:
    """
    Save a confusion matrix heatmap PNG, axis-labeled with the actual
    class names. Matplotlib is imported inside this function (not at
    module level), matching train.py's save_training_plots() convention,
    so importing this module never requires a display backend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "confusion_matrix.png"

    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.7)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def evaluate_model(config: dict, save_artifacts: bool = True) -> Dict:
    """
    Full Phase 7 evaluation orchestration:

      1. Build the test tf.data pipeline from data/processed/test ONLY.
      2. Load the trained model (models/best_model.keras), no retrain.
      3. Verify class ordering/count agreement between model and test
         dataset.
      4. model.evaluate() for test loss/accuracy.
      5. Collect predictions/probabilities across the full test set.
      6. Compute per-class precision/recall/F1/support, confusion
         matrix, classification report, macro/weighted averages.
      7. If save_artifacts: write all outputs under
         results/evaluation/ (or evaluation.output_dir from config).

    Returns:
        A dict with test_loss, test_accuracy, class_names, y_true,
        y_pred, y_proba, and the full `metrics` dict from
        compute_metrics(), for use by tests or the CLI report.
    """
    test_ds, class_names = build_test_dataset(config)
    logger.info("Test classes (%d): %s", len(class_names), class_names)

    model = load_trained_model(config)
    verify_class_ordering(model, class_names, config)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    logger.info("Test loss: %.4f | Test accuracy: %.4f", test_loss, test_accuracy)

    y_true, y_pred, y_proba = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names)

    result = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }

    if save_artifacts:
        output_dir = _evaluation_output_dir(config)
        save_test_metrics_json(test_loss, test_accuracy, class_names, metrics, output_dir)
        save_classification_report_csv(metrics, output_dir)
        save_per_class_metrics_csv(metrics, output_dir)
        save_confusion_matrix_csv(metrics, class_names, output_dir)
        save_confusion_matrix_plot(metrics, class_names, output_dir)
        logger.info("Saved evaluation artifacts to %s", output_dir)

    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate the trained Vehicle-10 CNN on the test split.")
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        help="Path to config.yaml (default: src/config/config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_model(config)


if __name__ == "__main__":
    main()