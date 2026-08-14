"""
src/training/train.py

Training pipeline for the Vehicle-10 CNN (Phase 6).

This module wires together the already-completed pieces of the
project:

  - src/data/loader.py        -> load_config()
  - src/data/augmentation.py  -> TrainingAugmentor / AugmentationConfig
  - src/models/cnn_model.py   -> build_and_compile_model(), config helpers

It does NOT redefine the CNN architecture, does NOT duplicate the
augmentation transforms, and does NOT touch the raw dataset or the
test split. Only data/processed/train/ and data/processed/validation/
are read.

High-level flow (see train_model()):

  1. Build a tf.data pipeline over data/processed/train and
     data/processed/validation using image_dataset_from_directory,
     which infers class names from the subfolder names (sorted
     alphabetically) -- the SAME ordering convention Keras uses
     everywhere else, so this is what "class ordering" means
     throughout this module.
  2. Apply the existing TrainingAugmentor to the training pipeline
     ONLY, via tf.numpy_function (the augmentor works on PIL images,
     not tensors). Validation images pass through untouched.
  3. Compute class weights from the training directory's file counts
     (not by iterating the tf.data pipeline) using
     sklearn.utils.class_weight.compute_class_weight, indexed to
     match the same sorted class ordering used above.
  4. Build + compile the model via cnn_model.build_and_compile_model().
  5. Train with ModelCheckpoint / EarlyStopping / ReduceLROnPlateau,
     passing class_weight into model.fit().
  6. Save training history (accuracy/loss curves) as JSON and PNG
     plots under results/.

No precision/recall/F1 or other evaluation metrics are computed here
-- that is Phase 7 (src/evaluation/evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

from src.data.augmentation import (
    AugmentationConfig,
    Split,
    TrainingAugmentor,
    augmentation_config_from_dict,
)
from src.data.loader import load_config
from src.models.cnn_model import (
    build_and_compile_model,
    dense_units_from_config,
    input_shape_from_config,
    num_classes_from_config,
)

logger = logging.getLogger("vehicle10.training")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """
    Seed Python's random module, NumPy, and TensorFlow.

    This makes model initialization and most of the training pipeline
    reproducible. It does NOT force full bitwise TensorFlow
    determinism (e.g. via TF_DETERMINISTIC_OPS) -- some GPU ops and
    tf.data's own buffering/interleaving are not made bit-exact by
    seeding alone, and forcing full determinism can noticeably slow
    training on a normal laptop, which the project explicitly wants
    to avoid. Class-weight computation and the train/validation split
    on disk (from Phase 3's preprocessing) are already fully
    deterministic for a given seed independent of this function.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------
# Device detection
# --------------------------------------------------------------------------

def detect_device() -> str:
    """
    Detect whether TensorFlow sees a GPU. Returns "GPU" or "CPU".
    Never raises even if no GPU / no CUDA is present -- CPU training
    is an expected, supported path for this project.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:  # pragma: no cover - defensive only
        gpus = []
    return "GPU" if gpus else "CPU"


# --------------------------------------------------------------------------
# Dataset loading (tf.data)
# --------------------------------------------------------------------------

def _processed_split_dir(config: dict, split: str) -> Path:
    """Path to data/processed/<split> (e.g. 'train', 'validation')."""
    processed_root = Path(config["dataset"]["processed_root"])
    return processed_root / split


def build_datasets(
    config: dict,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Build the training and validation tf.data pipelines from
    data/processed/train and data/processed/validation.

    Only these two directories are read. The test split
    (data/processed/test) is never referenced by this function or
    anywhere else in this module.

    Returns:
        (train_ds, val_ds, class_names)

        - train_ds: shuffled, augmented (see apply_training_augmentation),
          batched, prefetched. Labels are one-hot (categorical), matching
          cnn_model's categorical_crossentropy loss.
        - val_ds: NOT shuffled, NOT augmented, batched, prefetched.
        - class_names: the sorted list of class subfolder names, i.e.
          the ordering Keras used to assign integer labels 0..N-1.
          This is the authoritative "class ordering" for class weights,
          predictions, etc.
    """
    pp_cfg = config["preprocessing"]
    train_cfg = config.get("training", {})

    image_size = tuple(pp_cfg["image_size"])  # (width, height) in config
    # image_dataset_from_directory expects image_size as (height, width).
    target_size = (image_size[1], image_size[0])
    batch_size = batch_size or train_cfg.get("batch_size", 32)

    train_dir = _processed_split_dir(config, "train")
    val_dir = _processed_split_dir(config, "validation")

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training split not found at {train_dir}. Run preprocessing "
            f"(src/data/preprocess.py) before training."
        )
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Validation split not found at {val_dir}. Run preprocessing "
            f"(src/data/preprocess.py) before training."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=target_size,
        batch_size=None,  # batch AFTER augmentation, see below
        shuffle=True,
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=target_size,
        batch_size=None,
        shuffle=False,
    )

    class_names = train_ds.class_names
    val_class_names = val_ds.class_names
    if class_names != val_class_names:
        raise ValueError(
            "Train/validation class ordering mismatch: "
            f"train={class_names} validation={val_class_names}. "
            "This would silently corrupt label alignment."
        )

    return train_ds, val_ds, class_names


def apply_training_augmentation(
    train_ds: tf.data.Dataset,
    augmentor: TrainingAugmentor,
) -> tf.data.Dataset:
    """
    Apply the existing TrainingAugmentor to every image in `train_ds`.

    TrainingAugmentor.augment() operates on PIL Images (see
    src/data/augmentation.py), not tensors, so each image is routed
    through tf.numpy_function. This keeps augmentation logic entirely
    inside augmentation.py -- no transform is reimplemented here.

    This function must only ever be called on the TRAINING dataset.
    Validation/test datasets must never be passed here.
    """

    def _augment_numpy(image_uint8: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image_uint8, mode="RGB")
        augmented = augmentor.augment(pil_img)
        return np.array(augmented, dtype=np.uint8)

    def _augment_map_fn(image: tf.Tensor, label: tf.Tensor):
        image_uint8 = tf.cast(image, tf.uint8)
        augmented = tf.numpy_function(
            func=_augment_numpy, inp=[image_uint8], Tout=tf.uint8
        )
        augmented.set_shape(image.shape)
        return augmented, label

    return train_ds.map(_augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _normalize(image: tf.Tensor, label: tf.Tensor):
    """Scale uint8 [0, 255] images to float32 [0, 1]."""
    return tf.cast(image, tf.float32) / 255.0, label


def finalize_dataset(
    ds: tf.data.Dataset,
    batch_size: int,
    shuffle_buffer: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Normalize, optionally shuffle, batch, and prefetch a per-image
    tf.data.Dataset. Shared by both train and validation so batching/
    prefetching behavior stays consistent; only `shuffle_buffer`
    differs (validation should pass None).
    """
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_training_pipeline(
    config: dict,
    seed: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Full input-pipeline assembly: build_datasets() + training-only
    augmentation + normalization/shuffle/batch/prefetch for both
    splits.

    This is the function train_model() actually calls; build_datasets()
    and apply_training_augmentation() are exposed separately mainly so
    tests can exercise each stage independently.
    """
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)

    train_ds, val_ds, class_names = build_datasets(config, batch_size=batch_size, seed=seed)

    aug_cfg = augmentation_config_from_dict(config.get("augmentation", {}))
    augmentor = TrainingAugmentor(aug_cfg, seed=seed)
    train_ds = apply_training_augmentation(train_ds, augmentor)

    # Shuffle buffer: modest fixed size rather than the full training-set
    # count, so we never need to hold a large fraction of the dataset in
    # memory at once (see project's memory-safety requirements).
    train_ds = finalize_dataset(train_ds, batch_size, shuffle_buffer=1000)
    val_ds = finalize_dataset(val_ds, batch_size, shuffle_buffer=None)

    return train_ds, val_ds, class_names


# --------------------------------------------------------------------------
# Class weights
# --------------------------------------------------------------------------

def compute_training_class_weights(
    config: dict, class_names: List[str]
) -> Dict[int, float]:
    """
    Compute class weights from the TRAINING split's on-disk file counts
    only (data/processed/train/<class>/*), using
    sklearn.utils.class_weight.compute_class_weight(class_weight="balanced").

    Args:
        config: loaded config.yaml dict (used to locate
            data/processed/train).
        class_names: the sorted class-name ordering returned by
            build_datasets(), i.e. the ordering the model's output
            layer uses. Weight dict keys are integer indices into
            THIS list, so they line up with the label encoding
            image_dataset_from_directory produced.

    Returns:
        {class_index: weight}, suitable for model.fit(class_weight=...).

    Validation/test files are never counted here -- using validation
    or test distributions to weight the training loss would leak
    information about those splits into training.
    """
    train_dir = _processed_split_dir(config, "train")

    # One label per training file, encoded using the SAME class_names
    # ordering as the tf.data pipeline, so weight indices line up with
    # the model's output classes.
    labels: List[int] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = train_dir / class_name
        if not class_dir.is_dir():
            continue
        count = sum(1 for f in class_dir.iterdir() if f.is_file())
        labels.extend([class_index] * count)

    if not labels:
        raise ValueError(f"No training images found under {train_dir}")

    unique_classes = np.arange(len(class_names))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=np.array(labels),
    )
    return {int(idx): float(w) for idx, w in zip(unique_classes, weights)}


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------

def build_callbacks(config: dict) -> List[tf.keras.callbacks.Callback]:
    """
    Build the training callbacks (ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau) from config.yaml's training: section. Each is
    only included if its `enabled` flag is true (defaulting to True
    if the sub-section is present at all).
    """
    train_cfg = config.get("training", {})
    callbacks: List[tf.keras.callbacks.Callback] = []

    checkpoint_cfg = train_cfg.get("checkpoint", {})
    if checkpoint_cfg.get("enabled", True):
        checkpoint_path = Path(checkpoint_cfg.get("path", "models/best_model.keras"))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            )
        )

    early_stopping_cfg = train_cfg.get("early_stopping", {})
    if early_stopping_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_cfg.get("patience", 5),
                restore_best_weights=True,
            )
        )

    reduce_lr_cfg = train_cfg.get("reduce_lr", {})
    if reduce_lr_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=reduce_lr_cfg.get("patience", 2),
                factor=reduce_lr_cfg.get("factor", 0.5),
                verbose=1,
            )
        )

    return callbacks


# --------------------------------------------------------------------------
# Results (history JSON + accuracy/loss plots)
# --------------------------------------------------------------------------

def save_training_history(history: tf.keras.callbacks.History, config: dict) -> Path:
    """
    Save `history.history` (per-epoch train/val accuracy and loss) as
    JSON under results/metrics/training_history.json (path configurable
    via training.results.metrics_dir).
    """
    results_cfg = config.get("training", {}).get("results", {})
    metrics_dir = Path(results_cfg.get("metrics_dir", "results/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history_path = metrics_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    return history_path


def save_training_plots(history: tf.keras.callbacks.History, config: dict) -> Tuple[Path, Path]:
    """
    Save training/validation accuracy and loss curves as PNGs under
    results/plots/ (path configurable via training.results.plots_dir).

    Matplotlib is imported inside this function (not at module level)
    so importing train.py never requires a display backend or fails
    in headless/test environments that don't need plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_cfg = config.get("training", {}).get("results", {})
    plots_dir = Path(results_cfg.get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    h = history.history
    epochs_range = range(1, len(h.get("loss", [])) + 1)

    acc_path = plots_dir / "training_accuracy.png"
    fig, ax = plt.subplots()
    if "accuracy" in h:
        ax.plot(epochs_range, h["accuracy"], label="train")
    if "val_accuracy" in h:
        ax.plot(epochs_range, h["val_accuracy"], label="validation")
    ax.set_title("Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig(acc_path)
    plt.close(fig)

    loss_path = plots_dir / "training_loss.png"
    fig, ax = plt.subplots()
    ax.plot(epochs_range, h.get("loss", []), label="train")
    if "val_loss" in h:
        ax.plot(epochs_range, h["val_loss"], label="validation")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(loss_path)
    plt.close(fig)

    return acc_path, loss_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def train_model(
    config: dict,
    epochs: Optional[int] = None,
    save_artifacts: bool = True,
) -> tf.keras.callbacks.History:
    """
    Full Phase 6 training orchestration:

      1. Seed Python/NumPy/TensorFlow from config.preprocessing.seed.
      2. Build the train/validation tf.data pipelines (train_ds is
         augmented via the existing TrainingAugmentor; val_ds is not).
      3. Compute class weights from the training split only.
      4. Build + compile the existing CNN (src/models/cnn_model.py).
      5. Train with ModelCheckpoint / EarlyStopping / ReduceLROnPlateau,
         passing class_weight into model.fit().
      6. If save_artifacts: write training history JSON + accuracy/loss
         plots under results/.

    Only data/processed/train and data/processed/validation are ever
    read. data/processed/test is never referenced.

    Args:
        config: loaded config.yaml dict.
        epochs: override for training.epochs (mainly for tests / smoke
            runs); defaults to the configured value.
        save_artifacts: if False, skip writing history/plots to disk
            (used by fast tests that only care about the fit() call
            succeeding).

    Returns:
        The Keras History object returned by model.fit().
    """
    seed = config.get("preprocessing", {}).get("seed", 42)
    set_seeds(seed)

    device = detect_device()
    logger.info("Device: %s", device)

    train_ds, val_ds, class_names = build_training_pipeline(config, seed=seed)
    logger.info("Classes (%d): %s", len(class_names), class_names)

    class_weights = compute_training_class_weights(config, class_names)
    logger.info("Class weights: %s", class_weights)

    input_shape = input_shape_from_config(config)
    num_classes = num_classes_from_config(config)
    if num_classes != len(class_names):
        logger.warning(
            "config.dataset.classes has %d entries but %d class folders "
            "were found on disk (%s); using the on-disk class list.",
            num_classes, len(class_names), class_names,
        )

    train_cfg = config.get("training", {})
    model = build_and_compile_model(
        input_shape=input_shape,
        num_classes=len(class_names),
        dense_units=dense_units_from_config(config),
        learning_rate=train_cfg.get("learning_rate"),
    )
    logger.info("Model parameters: %d", model.count_params())
    logger.info("Batch size: %d", train_cfg.get("batch_size", 32))

    total_epochs = epochs if epochs is not None else train_cfg.get("epochs", 30)
    logger.info("Epochs: %d", total_epochs)

    callbacks = build_callbacks(config)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    if history.history.get("val_accuracy"):
        best_val_acc = max(history.history["val_accuracy"])
        logger.info("Best validation accuracy: %.4f", best_val_acc)
    logger.info("Training complete.")

    if save_artifacts:
        history_path = save_training_history(history, config)
        acc_path, loss_path = save_training_plots(history, config)
        logger.info("Saved history to %s", history_path)
        logger.info("Saved plots to %s and %s", acc_path, loss_path)

    return history


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train the Vehicle-10 CNN.")
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        help="Path to config.yaml (default: src/config/config.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config, epochs=args.epochs)


if __name__ == "__main__":
    main()