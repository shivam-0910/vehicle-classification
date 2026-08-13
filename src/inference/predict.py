"""
src/inference/predict.py

Single-image inference for the Vehicle-10 CNN (Phase 8).

This module loads the ALREADY-TRAINED model checkpoint
(models/best_model.keras by default, or training.checkpoint.path from
config -- same convention as src/evaluation/evaluate.py) and runs a
single image through it to produce a class prediction with confidence
and top-k breakdown.

It does NOT retrain, redesign the model, or duplicate preprocessing
logic. It reuses:

  - src/data/loader.py        -> load_config()
  - src/models/cnn_model.py   -> input_shape_from_config(),
                                  num_classes_from_config()
  - src/config/config.yaml    -> dataset.classes (class ordering),
                                  preprocessing.image_size,
                                  training.checkpoint.path

--------------------------------------------------------------------
CLASS ORDERING
--------------------------------------------------------------------
The model's softmax output units are ordered the same way
train.py/evaluate.py order them: `tf.keras.utils.image_dataset_from_
directory` assigns label indices from the *sorted* list of class
subfolder names. src/config/config.yaml's `dataset.classes` list is
already stored in that same sorted order (see config.yaml), so this
module derives class names from `sorted(config["dataset"]["classes"])`
rather than inventing a new ordering mechanism. This mirrors
evaluate.py's `verify_class_ordering()`, which treats the on-disk
sorted folder order as authoritative and cross-checks it against
config.

--------------------------------------------------------------------
PREPROCESSING
--------------------------------------------------------------------
Preprocessing here matches preprocess.py/evaluate.py exactly:
  - Convert to RGB (mirrors preprocess.convert_to_rgb: RGBA/P/L/etc.
    all safely become RGB; alpha is composited onto white).
  - Resize to preprocessing.image_size ([width, height] in config).
  - Cast to float32 and divide by 255.0 (mirrors evaluate.py's
    build_test_dataset._normalize).
  - Add a batch dimension.

No augmentation is applied -- this module never imports
src.data.augmentation, and inference is fully deterministic.

--------------------------------------------------------------------
READ-ONLY / NO DATA LEAKAGE
--------------------------------------------------------------------
This module never writes to data/raw, data/processed, or
models/best_model.keras, and never reads training/validation/test
labels to influence a prediction. It is a pure
INPUT IMAGE -> PREPROCESS -> TRAINED MODEL -> SOFTMAX -> CLASS +
CONFIDENCE + TOP-K pipeline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from src.data.loader import load_config
from src.models.cnn_model import input_shape_from_config, num_classes_from_config


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class InferenceError(Exception):
    """Base class for user-facing inference errors."""


class ImageNotFoundError(InferenceError):
    """Raised when the requested image path does not exist."""


class ImageDecodeError(InferenceError):
    """Raised when the image file exists but cannot be opened/decoded."""


class ModelNotFoundError(InferenceError):
    """Raised when the trained model checkpoint file does not exist."""


class ClassCountMismatchError(InferenceError):
    """Raised when the model's output units don't match the configured class count."""


class InvalidTopKError(InferenceError):
    """Raised when top_k is not a valid value for the number of classes."""


class InvalidPredictionError(InferenceError):
    """Raised when the model's output probabilities are malformed (non-finite, wrong shape, etc.)."""


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassPrediction:
    """One entry in a top-k prediction list."""
    class_name: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    """Full result of predicting a single image."""
    image_path: str
    predicted_class: str
    confidence: float
    probabilities: List[float]           # full distribution, aligned to class_names order
    class_names: List[str]
    top_k: List[ClassPrediction]


# --------------------------------------------------------------------------
# Class ordering (reuses config.yaml's dataset.classes, sorted -- the
# same convention image_dataset_from_directory / evaluate.py use)
# --------------------------------------------------------------------------

def get_class_names(config: dict) -> List[str]:
    """
    Derive the authoritative, softmax-index-aligned class name list
    from config["dataset"]["classes"], sorted alphabetically to match
    the ordering `tf.keras.utils.image_dataset_from_directory` used
    during training/evaluation (sorted subfolder names).
    """
    classes = (config or {}).get("dataset", {}).get("classes")
    if not classes:
        raise InferenceError(
            "config.yaml is missing dataset.classes; cannot determine "
            "class ordering for inference."
        )
    return sorted(classes)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def _model_path_from_config(config: dict) -> Path:
    """Path to the trained checkpoint, from training.checkpoint.path
    (same lookup evaluate.py uses), defaulting to models/best_model.keras."""
    checkpoint_cfg = config.get("training", {}).get("checkpoint", {})
    return Path(checkpoint_cfg.get("path", "models/best_model.keras"))


def load_model(config: dict) -> keras.Model:
    """
    Load the already-trained model checkpoint. Does NOT retrain,
    fine-tune, or modify the checkpoint file in any way.
    """
    model_path = _model_path_from_config(config)
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"Trained model checkpoint not found at '{model_path}'. "
            f"Run training (src/training/train.py) before inference."
        )
    return keras.models.load_model(model_path)


def verify_model_matches_classes(model: keras.Model, class_names: List[str]) -> None:
    """
    Verify model output-unit count matches the configured class count.
    Raises ClassCountMismatchError with a clear message otherwise.
    """
    output_units = model.output_shape[-1]
    if output_units != len(class_names):
        raise ClassCountMismatchError(
            f"Model output has {output_units} units but {len(class_names)} "
            f"classes are configured ({class_names}). This would silently "
            f"misalign predictions, so inference is refusing to continue."
        )


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def _convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Safely convert any PIL image mode to RGB. Mirrors
    preprocess.convert_to_rgb(): RGBA/LA/palette-with-transparency
    images are alpha-composited onto a white background so transparency
    doesn't turn black; everything else (grayscale L, P without
    transparency, etc.) uses PIL's standard convert("RGB").
    """
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def preprocess_image(image_path: str, config: dict) -> np.ndarray:
    """
    Load an image from disk and preprocess it identically to
    training/evaluation:

      1. Verify the file exists.
      2. Open safely with PIL (raises ImageDecodeError on corruption).
      3. Convert to RGB (handles RGB/RGBA/P/grayscale).
      4. Resize to preprocessing.image_size from config.
      5. Cast to float32, normalize [0,255] -> [0,1].
      6. Add a batch dimension: (H,W,3) -> (1,H,W,3).

    No augmentation is applied. Deterministic.

    Returns:
        np.ndarray of shape (1, height, width, 3), dtype float32,
        values in [0, 1].
    """
    path = Path(image_path)
    if not path.is_file():
        raise ImageNotFoundError(f"Image not found at '{image_path}'.")

    try:
        with Image.open(path) as img:
            img.load()
            rgb_img = _convert_to_rgb(img)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError(
            f"Could not open or decode image at '{image_path}': {exc}"
        ) from exc

    height, width, _ = input_shape_from_config(config)
    resized = rgb_img.resize((width, height), resample=Image.Resampling.BILINEAR)

    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)  # (1, H, W, 3)

    return array


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def _validate_probabilities(probs: np.ndarray, num_classes: int) -> None:
    if probs.ndim != 1 or probs.shape[0] != num_classes:
        raise InvalidPredictionError(
            f"Expected a probability vector of shape ({num_classes},), "
            f"got shape {probs.shape}."
        )
    if not np.all(np.isfinite(probs)):
        raise InvalidPredictionError(
            "Model produced non-finite probability values (NaN or Inf)."
        )
    total = float(probs.sum())
    if not np.isclose(total, 1.0, atol=1e-2):
        raise InvalidPredictionError(
            f"Predicted probabilities do not sum to ~1.0 (got {total:.4f})."
        )


def predict_image(
    model: keras.Model,
    image_array: np.ndarray,
    class_names: List[str],
    top_k: int = 3,
) -> PredictionResult:
    """
    Run the model on a preprocessed image batch of shape (1,H,W,3) and
    build a PredictionResult with the predicted class, confidence,
    full probability distribution, and top-k predictions.

    Raises InvalidTopKError if top_k is not in [1, len(class_names)].
    """
    if not (1 <= top_k <= len(class_names)):
        raise InvalidTopKError(
            f"top_k must be between 1 and {len(class_names)} "
            f"(the number of classes), got {top_k}."
        )

    raw_output = model.predict(image_array, verbose=0)
    probs = np.asarray(raw_output[0], dtype=np.float64)

    _validate_probabilities(probs, len(class_names))

    predicted_index = int(np.argmax(probs))
    if not (0 <= predicted_index < len(class_names)):
        raise InvalidPredictionError(
            f"Predicted index {predicted_index} is out of range for "
            f"{len(class_names)} classes."
        )

    order = np.argsort(probs)[::-1][:top_k]
    top_k_predictions = [
        ClassPrediction(class_name=class_names[i], probability=float(probs[i]))
        for i in order
    ]

    return PredictionResult(
        image_path="",  # filled in by predict_from_path()
        predicted_class=class_names[predicted_index],
        confidence=float(probs[predicted_index]),
        probabilities=[float(p) for p in probs],
        class_names=class_names,
        top_k=top_k_predictions,
    )


def predict_from_path(
    model: keras.Model,
    image_path: str,
    config: dict,
    class_names: List[str],
    top_k: int = 3,
) -> PredictionResult:
    """Convenience wrapper: preprocess_image() + predict_image(), with
    image_path recorded on the result."""
    image_array = preprocess_image(image_path, config)
    result = predict_image(model, image_array, class_names, top_k=top_k)
    return PredictionResult(
        image_path=image_path,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
        class_names=result.class_names,
        top_k=result.top_k,
    )


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def format_prediction(result: PredictionResult) -> str:
    """Render a PredictionResult as a clean, human-readable report."""
    lines = []
    width = 40
    lines.append("=" * width)
    lines.append("VEHICLE CLASSIFICATION RESULT")
    lines.append("=" * width)
    lines.append("")
    lines.append(f"Image: {result.image_path}")
    lines.append("")
    lines.append(f"Predicted class: {result.predicted_class}")
    lines.append(f"Confidence: {result.confidence * 100:.2f}%")
    lines.append("")
    lines.append(f"Top {len(result.top_k)} predictions:")
    lines.append("-" * width)
    for i, pred in enumerate(result.top_k, start=1):
        lines.append(f"{i}. {pred.class_name:<14}{pred.probability * 100:.2f}%")
    lines.append("-" * width)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_prediction(config_path: str, image_path: str, top_k: int = 3) -> PredictionResult:
    """
    Full end-to-end single-image inference:
      config -> class names -> model -> preprocess -> predict.
    Used by both the CLI and tests, so CLI/API behavior stays
    identical.
    """
    config = load_config(config_path)
    class_names = get_class_names(config)

    model = load_model(config)
    verify_model_matches_classes(model, class_names)

    return predict_from_path(model, image_path, config, class_names, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-image inference with the trained Vehicle-10 CNN."
    )
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        help="Path to config.yaml (default: src/config/config.yaml)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image to classify.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    args = parser.parse_args()

    try:
        result = run_prediction(args.config, args.image, top_k=args.top_k)
    except InferenceError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(format_prediction(result))


if __name__ == "__main__":
    main()