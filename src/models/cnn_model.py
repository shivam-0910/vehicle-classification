"""
src/models/cnn_model.py

CNN architecture for the Vehicle-10 classification project (Phase 5).

This module defines a from-scratch Convolutional Neural Network for
128x128 RGB vehicle images, per the Bright Hub internship handbook
requirements:

    Convolution Layer -> Max Pooling -> Dropout -> Dense Layer -> Softmax

Architecture:

    Input (128, 128, 3)
      -> Conv2D(32, 3x3, relu)  -> MaxPooling2D(2x2)
      -> Conv2D(64, 3x3, relu)  -> MaxPooling2D(2x2)
      -> Conv2D(128, 3x3, relu) -> MaxPooling2D(2x2)
      -> Dropout(0.3)
      -> Flatten
      -> Dense(128, relu)
      -> Dropout(0.4)
      -> Dense(num_classes, softmax)

Design notes:
  - Three conv/pool blocks with increasing filter counts (32 -> 64 -> 128)
    is a standard, well-understood pattern for a from-scratch CNN at
    128x128 resolution: enough capacity to learn vehicle shapes/textures
    without being an oversized architecture for a laptop-trainable
    internship project.
  - No BatchNormalization: keeps the architecture simple and easy to
    explain layer-by-layer, per project constraints.
  - No pretrained/transfer-learning backbones (no ResNet/MobileNet/etc.)
    -- this is intentionally a from-scratch CNN.
  - Dropout is used twice: once after the conv stack (0.3, regularizes
    the flattened feature map) and once after the dense layer (0.4,
    a slightly stronger rate right before the classification head,
    since the dense layer has the most parameters and is most prone
    to overfitting).
  - The number of output classes is derived from configuration
    (config.yaml's `dataset.classes` list) rather than hardcoded,
    though it defaults to 10 to match the current Vehicle-10 dataset.

This module is training-agnostic: it only builds and (optionally)
compiles the model. Epochs, batch size, callbacks, and class-weight
handling belong to the future src/training/train.py.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from tensorflow import keras
from tensorflow.keras import layers


# Defaults mirror src/config/config.yaml (preprocessing.image_size + the
# 10-entry dataset.classes list) so the model "just works" if built with
# no arguments, while still allowing config-driven overrides.
DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (128, 128, 3)
DEFAULT_NUM_CLASSES: int = 10
DEFAULT_CONV_DROPOUT_RATE: float = 0.3
DEFAULT_DENSE_DROPOUT_RATE: float = 0.4
DEFAULT_DENSE_UNITS: int = 128


def num_classes_from_config(config: dict) -> int:
    """
    Derive the number of output classes from a loaded config.yaml dict,
    using dataset.classes (the authoritative class list already used by
    validator.py / loader.py). Falls back to DEFAULT_NUM_CLASSES if the
    section is missing, so this stays usable in isolation (e.g. tests).
    """
    classes = (config or {}).get("dataset", {}).get("classes")
    if not classes:
        return DEFAULT_NUM_CLASSES
    return len(classes)


def input_shape_from_config(config: dict) -> Tuple[int, int, int]:
    """
    Derive the (height, width, channels) input shape from config.yaml's
    preprocessing.image_size ([width, height]), assuming RGB (3 channels)
    as produced by preprocess.py's convert_to_rgb(). Falls back to
    DEFAULT_INPUT_SHAPE if unavailable.
    """
    image_size = (config or {}).get("preprocessing", {}).get("image_size")
    if not image_size or len(image_size) != 2:
        return DEFAULT_INPUT_SHAPE
    width, height = image_size
    return (height, width, 3)


def build_model(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    conv_dropout_rate: float = DEFAULT_CONV_DROPOUT_RATE,
    dense_dropout_rate: float = DEFAULT_DENSE_DROPOUT_RATE,
    dense_units: int = DEFAULT_DENSE_UNITS,
) -> keras.Model:
    """
    Build (but do not compile) the Vehicle-10 CNN.

    Args:
        input_shape: (height, width, channels) of input images.
            Defaults to (128, 128, 3), matching the processed dataset.
        num_classes: number of output classes for the final softmax
            layer. Defaults to 10 (the current Vehicle-10 class count);
            prefer deriving this from config via num_classes_from_config()
            when building the model for real training.
        conv_dropout_rate: dropout applied once after the conv/pool
            stack, before flattening.
        dense_dropout_rate: dropout applied after the dense hidden
            layer, before the final classification layer.
        dense_units: number of units in the dense hidden layer.

    Returns:
        An uncompiled keras.Model. Use compile_model() to compile it,
        or compile it yourself (e.g. with different training settings).
    """
    inputs = keras.Input(shape=input_shape, name="vehicle_image")

    # --- Convolutional feature extraction ---
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # --- Regularization before the classification head ---
    x = layers.Dropout(conv_dropout_rate, name="conv_dropout")(x)
    x = layers.Flatten(name="flatten")(x)

    # --- Classification head ---
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense_dropout_rate, name="dense_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="vehicle10_cnn")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: Optional[float] = None,
) -> keras.Model:
    """
    Compile a model built by build_model() with the internship-mandated
    training settings: Adam optimizer, categorical crossentropy loss,
    accuracy metric.

    Training-specific configuration (epochs, batch size, callbacks,
    class weights) intentionally does NOT live here -- that belongs to
    the future training phase (src/training/train.py).

    Args:
        model: an uncompiled (or already-compiled) keras.Model, typically
            from build_model().
        learning_rate: optional override for Adam's learning rate. If
            None, Keras' Adam default (0.001) is used.

    Returns:
        The same model instance, compiled in place (also returned for
        convenient chaining, e.g. `model = compile_model(build_model())`).
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate) if learning_rate else "adam"
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_and_compile_model(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    conv_dropout_rate: float = DEFAULT_CONV_DROPOUT_RATE,
    dense_dropout_rate: float = DEFAULT_DENSE_DROPOUT_RATE,
    dense_units: int = DEFAULT_DENSE_UNITS,
    learning_rate: Optional[float] = None,
) -> keras.Model:
    """Convenience wrapper: build_model() followed by compile_model()."""
    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_dropout_rate=conv_dropout_rate,
        dense_dropout_rate=dense_dropout_rate,
        dense_units=dense_units,
    )
    return compile_model(model, learning_rate=learning_rate)