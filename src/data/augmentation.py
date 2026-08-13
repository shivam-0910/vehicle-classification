"""
src/data/augmentation.py

On-the-fly data augmentation for the Vehicle-10 dataset (Phase 4).

This module does NOT write an augmented copy of the dataset to disk.
It provides a small, composable API that takes an already-processed
128x128 RGB PIL Image (as produced by preprocess.py) and returns a
new, randomly-augmented PIL Image of the same size and mode. The
input image object is never mutated in place.

Supported augmentations (all optional / independently configurable):
  - rotation            (small random rotation, degrees)
  - zoom                (small random zoom in/out, center-cropped or
                          padded back to the original size)
  - horizontal_flip      (random left-right flip)
  - vertical_flip         (random top-bottom flip; OFF by default —
                          vehicles are normally upright, see config)
  - brightness           (moderate random brightness scaling)

--------------------------------------------------------------------
TRAINING-ONLY / DATA LEAKAGE SAFEGUARD
--------------------------------------------------------------------
Augmentation must only ever be applied to the "train" split. Rather
than relying on a path substring check (fragile — e.g. a class named
"train" would false-positive), this module exposes:

  - Split, an enum of the three known splits.
  - augment_image(image, config, split=Split.TRAIN, ...)  requires the
    caller to explicitly state which split the image belongs to, and
    raises SplitNotAugmentableError if split != Split.TRAIN.
  - TrainingAugmentor, a thin wrapper bound to Split.TRAIN at
    construction time. It has no way to accept a validation/test
    image at all -- there is no split argument to get wrong, so
    misuse would require deliberately instantiating a second
    augmentor and mislabeling it.

train.py (Phase 6) is expected to build its train/validation/test
image pipelines from separate DatasetEntry lists (already split
by loader.py / preprocess.py), and only ever pass the "train" list
through a TrainingAugmentor. Validation/test loading code should
never import this module at all.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PIL import Image, ImageEnhance


class Split(str, Enum):
    """The three dataset splits produced by preprocess.py."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitNotAugmentableError(ValueError):
    """Raised when augmentation is attempted on a non-training split."""


# --------------------------------------------------------------------------
# Config dataclasses (mirror src/config/config.yaml's `augmentation:` block)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RotationConfig:
    enabled: bool = True
    range_degrees: float = 10.0  # rotation sampled from [-range, +range]


@dataclass(frozen=True)
class ZoomConfig:
    enabled: bool = True
    range_fraction: float = 0.10  # sampled from [1 - range, 1 + range]


@dataclass(frozen=True)
class FlipConfig:
    enabled: bool = True
    probability: float = 0.5


@dataclass(frozen=True)
class BrightnessConfig:
    enabled: bool = True
    range_fraction: float = 0.20  # factor sampled from [1 - range, 1 + range]


@dataclass(frozen=True)
class AugmentationConfig:
    """Full augmentation configuration for one call to augment_image()."""
    enabled: bool = True
    rotation: RotationConfig = RotationConfig()
    zoom: ZoomConfig = ZoomConfig()
    horizontal_flip: FlipConfig = FlipConfig(enabled=True, probability=0.5)
    # Vertical flip defaults OFF: vehicles are normally upright, and a
    # vertically-flipped car/truck is not a realistic training example.
    # It remains fully configurable per the internship spec.
    vertical_flip: FlipConfig = FlipConfig(enabled=False, probability=0.5)
    brightness: BrightnessConfig = BrightnessConfig()


def augmentation_config_from_dict(cfg: dict) -> AugmentationConfig:
    """
    Build an AugmentationConfig from the `augmentation:` section of
    config.yaml (a plain dict, as returned by loader.load_config()).
    Missing keys fall back to the dataclass defaults above.
    """
    cfg = cfg or {}

    def _get(section: str, key: str, default):
        return (cfg.get(section) or {}).get(key, default)

    rotation = RotationConfig(
        enabled=_get("rotation", "enabled", RotationConfig().enabled),
        range_degrees=_get("rotation", "range_degrees", RotationConfig().range_degrees),
    )
    zoom = ZoomConfig(
        enabled=_get("zoom", "enabled", ZoomConfig().enabled),
        range_fraction=_get("zoom", "range_fraction", ZoomConfig().range_fraction),
    )
    horizontal_flip = FlipConfig(
        enabled=_get("horizontal_flip", "enabled", FlipConfig(enabled=True).enabled),
        probability=_get("horizontal_flip", "probability", FlipConfig(enabled=True).probability),
    )
    vertical_flip = FlipConfig(
        enabled=_get("vertical_flip", "enabled", FlipConfig(enabled=False).enabled),
        probability=_get("vertical_flip", "probability", FlipConfig(enabled=False).probability),
    )
    brightness = BrightnessConfig(
        enabled=_get("brightness", "enabled", BrightnessConfig().enabled),
        range_fraction=_get("brightness", "range_fraction", BrightnessConfig().range_fraction),
    )

    return AugmentationConfig(
        enabled=cfg.get("enabled", True),
        rotation=rotation,
        zoom=zoom,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness=brightness,
    )


# --------------------------------------------------------------------------
# Individual transforms
# --------------------------------------------------------------------------

def _apply_rotation(img: Image.Image, cfg: RotationConfig, rng: random.Random) -> Image.Image:
    if not cfg.enabled or cfg.range_degrees <= 0:
        return img
    angle = rng.uniform(-cfg.range_degrees, cfg.range_degrees)
    # fillcolor=white keeps corners consistent with the white background
    # used elsewhere in the pipeline (see preprocess.convert_to_rgb);
    # resample=Resampling.BICUBIC for smoother edges than the default NEAREST.
    return img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def _apply_zoom(img: Image.Image, cfg: ZoomConfig, rng: random.Random) -> Image.Image:
    if not cfg.enabled or cfg.range_fraction <= 0:
        return img
    factor = rng.uniform(1 - cfg.range_fraction, 1 + cfg.range_fraction)
    factor = max(factor, 1e-3)
    orig_w, orig_h = img.size
    new_w = max(1, round(orig_w * factor))
    new_h = max(1, round(orig_h * factor))
    resized = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    if factor >= 1.0:
        # Zoomed in: crop back down to the original size from the center.
        left = (new_w - orig_w) // 2
        top = (new_h - orig_h) // 2
        return resized.crop((left, top, left + orig_w, top + orig_h))

    # Zoomed out: paste onto a white canvas of the original size, centered.
    canvas = Image.new(img.mode, (orig_w, orig_h), (255, 255, 255))
    left = (orig_w - new_w) // 2
    top = (orig_h - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _apply_horizontal_flip(img: Image.Image, cfg: FlipConfig, rng: random.Random) -> Image.Image:
    if not cfg.enabled:
        return img
    if rng.random() < cfg.probability:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img


def _apply_vertical_flip(img: Image.Image, cfg: FlipConfig, rng: random.Random) -> Image.Image:
    if not cfg.enabled:
        return img
    if rng.random() < cfg.probability:
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img


def _apply_brightness(img: Image.Image, cfg: BrightnessConfig, rng: random.Random) -> Image.Image:
    if not cfg.enabled or cfg.range_fraction <= 0:
        return img
    factor = rng.uniform(1 - cfg.range_fraction, 1 + cfg.range_fraction)
    factor = max(factor, 0.0)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def augment_image(
    image: Image.Image,
    config: AugmentationConfig,
    split: Split,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Apply the configured augmentations to `image` and return a NEW
    PIL Image of the same size and mode. `image` is never modified
    in place.

    Args:
        image:  a PIL Image (expected 128x128 RGB, matching
                data/processed output, but any size/mode is accepted
                and preserved).
        config: an AugmentationConfig describing which transforms are
                enabled and their ranges.
        split:  MUST be Split.TRAIN. Any other value raises
                SplitNotAugmentableError -- this is the structural
                guard against accidentally augmenting validation/test
                data.
        seed:   optional seed for deterministic output (same seed +
                same image + same config -> same result). If omitted,
                a fresh, unseeded Random() is used (non-deterministic,
                appropriate for real training).

    Returns:
        A new PIL Image, same size and mode as the input.
    """
    if split != Split.TRAIN:
        raise SplitNotAugmentableError(
            f"augment_image() may only be called for split={Split.TRAIN!r}, "
            f"got {split!r}. Validation and test images must never be "
            f"augmented."
        )

    original_size = image.size
    original_mode = image.mode

    rng = random.Random(seed) if seed is not None else random.Random()

    out = image.copy()

    if not config.enabled:
        return out

    out = _apply_rotation(out, config.rotation, rng)
    out = _apply_zoom(out, config.zoom, rng)
    out = _apply_horizontal_flip(out, config.horizontal_flip, rng)
    out = _apply_vertical_flip(out, config.vertical_flip, rng)
    out = _apply_brightness(out, config.brightness, rng)

    # Defensive: augmentation must never change dimensions or mode.
    if out.size != original_size:
        out = out.resize(original_size, resample=Image.Resampling.BICUBIC)
    if out.mode != original_mode:
        out = out.convert(original_mode)

    return out


class TrainingAugmentor:
    """
    Convenience wrapper bound to Split.TRAIN at construction time.

    This is the API train.py (Phase 6) is expected to use: build one
    TrainingAugmentor from config, then call .augment(image) for every
    image drawn from the TRAINING split's file list only. There is no
    split argument to pass incorrectly, so a validation/test image can
    only be augmented by deliberately misusing loader-level split
    bookkeeping to feed it in -- augmentation.py itself provides no
    path that accepts a non-train split.
    """

    def __init__(self, config: AugmentationConfig, seed: Optional[int] = None):
        self._config = config
        self._base_seed = seed
        # Always create an RNG instance so static checkers know `_rng`
        # is a `random.Random`. When `seed` is None this behaves like
        # an unseeded RNG and is not used (we only derive per-call
        # seeds when `_base_seed` is not None).
        self._rng = random.Random(seed)

    def augment(self, image: Image.Image) -> Image.Image:
        """Augment one image using this augmentor's config. Non-deterministic
        unless a seed was supplied at construction, in which case calls are
        deterministic in call-order (each call advances the shared RNG)."""
        if self._base_seed is not None:
            # Derive a fresh per-call seed from the shared RNG so repeated
            # calls with a fixed base seed are reproducible as a sequence,
            # rather than every call returning an identical augmentation.
            per_call_seed = self._rng.randint(0, 2**31 - 1)
            return augment_image(image, self._config, Split.TRAIN, seed=per_call_seed)
        return augment_image(image, self._config, Split.TRAIN, seed=None)