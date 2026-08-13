"""
Tests for src/data/augmentation.py.

All tests use synthetic in-memory PIL images (128x128 RGB, matching
the real processed dataset's output format) — none depend on the
real Vehicle-10 dataset or on data/processed/ existing on disk.
"""
import pytest
from PIL import Image
from dataclasses import replace
from typing import Iterable, cast

from src.data.augmentation import (
    AugmentationConfig,
    BrightnessConfig,
    FlipConfig,
    RotationConfig,
    Split,
    SplitNotAugmentableError,
    TrainingAugmentor,
    ZoomConfig,
    augment_image,
    augmentation_config_from_dict,
)


def _make_image(size=(128, 128), mode="RGB", color=(200, 50, 50)):
    return Image.new(mode, size, color)


def _all_enabled_config(**overrides) -> AugmentationConfig:
    # Construct explicitly to satisfy static type-checkers.
    cfg = AugmentationConfig(
        enabled=True,
        rotation=RotationConfig(enabled=True, range_degrees=10.0),
        zoom=ZoomConfig(enabled=True, range_fraction=0.10),
        horizontal_flip=FlipConfig(enabled=True, probability=1.0),
        vertical_flip=FlipConfig(enabled=True, probability=1.0),
        brightness=BrightnessConfig(enabled=True, range_fraction=0.20),
    )
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def _all_disabled_config() -> AugmentationConfig:
    return AugmentationConfig(
        enabled=True,
        rotation=RotationConfig(enabled=False),
        zoom=ZoomConfig(enabled=False),
        horizontal_flip=FlipConfig(enabled=False),
        vertical_flip=FlipConfig(enabled=False),
        brightness=BrightnessConfig(enabled=False),
    )


# --------------------------------------------------------------------
# 1. Module import
# --------------------------------------------------------------------

def test_module_imports_successfully():
    import importlib
    importlib.import_module("src.data.augmentation")


# --------------------------------------------------------------------
# 2 & 3. Mode / size preserved
# --------------------------------------------------------------------

def test_output_remains_rgb():
    img = _make_image(mode="RGB")
    out = augment_image(img, _all_enabled_config(), Split.TRAIN, seed=1)
    assert out.mode == "RGB"


def test_output_remains_128x128():
    img = _make_image(size=(128, 128))
    out = augment_image(img, _all_enabled_config(), Split.TRAIN, seed=1)
    assert out.size == (128, 128)


# --------------------------------------------------------------------
# 4-8. Individual augmentations actually change the image
# --------------------------------------------------------------------

def test_rotation_changes_pixels():
    img = _make_image(color=(255, 0, 0))
    # Draw a distinct corner so rotation is detectable.
    for x in range(20):
        for y in range(20):
            img.putpixel((x, y), (0, 255, 0))

    cfg = _all_disabled_config()
    cfg = replace(cfg, rotation=RotationConfig(enabled=True, range_degrees=45.0))
    out = augment_image(img, cfg, Split.TRAIN, seed=1)
    assert list(cast(Iterable, out.getdata())) != list(cast(Iterable, img.getdata()))


def test_zoom_changes_pixels():
    img = _make_image(color=(255, 0, 0))
    for x in range(10):
        for y in range(10):
            img.putpixel((x, y), (0, 255, 0))

    cfg = replace(_all_disabled_config(), zoom=ZoomConfig(enabled=True, range_fraction=0.3))
    out = augment_image(img, cfg, Split.TRAIN, seed=2)
    assert list(cast(Iterable, out.getdata())) != list(cast(Iterable, img.getdata()))


def test_horizontal_flip_works():
    img = _make_image(color=(255, 0, 0))
    img.putpixel((0, 0), (0, 255, 0))  # top-left marker

    cfg = replace(_all_disabled_config(), horizontal_flip=FlipConfig(enabled=True, probability=1.0))
    out = augment_image(img, cfg, Split.TRAIN, seed=3)
    assert out.getpixel((0, 0)) != (0, 255, 0)
    assert out.getpixel((127, 0)) == (0, 255, 0)


def test_vertical_flip_works():
    img = _make_image(color=(255, 0, 0))
    img.putpixel((0, 0), (0, 255, 0))  # top-left marker

    cfg = replace(_all_disabled_config(), vertical_flip=FlipConfig(enabled=True, probability=1.0))
    out = augment_image(img, cfg, Split.TRAIN, seed=4)
    assert out.getpixel((0, 0)) != (0, 255, 0)
    assert out.getpixel((0, 127)) == (0, 255, 0)


def test_brightness_changes_pixels():
    img = _make_image(color=(100, 100, 100))
    cfg = replace(_all_disabled_config(), brightness=BrightnessConfig(enabled=True, range_fraction=0.5))
    out = augment_image(img, cfg, Split.TRAIN, seed=5)
    assert list(cast(Iterable, out.getdata())) != list(cast(Iterable, img.getdata()))


# --------------------------------------------------------------------
# 9. Disabled augmentations leave the image unchanged
# --------------------------------------------------------------------

def test_all_disabled_leaves_image_unchanged():
    img = _make_image(color=(123, 45, 67))
    out = augment_image(img, _all_disabled_config(), Split.TRAIN, seed=1)
    assert list(cast(Iterable, out.getdata())) == list(cast(Iterable, img.getdata()))
    assert out.size == img.size
    assert out.mode == img.mode


def test_globally_disabled_leaves_image_unchanged():
    img = _make_image(color=(123, 45, 67))
    cfg = AugmentationConfig(enabled=False)
    out = augment_image(img, cfg, Split.TRAIN, seed=1)
    assert list(cast(Iterable, out.getdata())) == list(cast(Iterable, img.getdata()))


# --------------------------------------------------------------------
# 10. Deterministic behavior with a seed
# --------------------------------------------------------------------

def test_seeded_augmentation_is_deterministic():
    img = _make_image(color=(80, 160, 200))
    cfg = _all_enabled_config()
    out_a = augment_image(img, cfg, Split.TRAIN, seed=99)
    out_b = augment_image(img, cfg, Split.TRAIN, seed=99)
    assert list(cast(Iterable, out_a.getdata())) == list(cast(Iterable, out_b.getdata()))


def test_different_seeds_can_differ():
    img = _make_image(color=(80, 160, 200))
    for x in range(15):
        for y in range(15):
            img.putpixel((x, y), (0, 0, 255))
    cfg = _all_enabled_config()
    out_a = augment_image(img, cfg, Split.TRAIN, seed=1)
    out_b = augment_image(img, cfg, Split.TRAIN, seed=2)
    assert list(cast(Iterable, out_a.getdata())) != list(cast(Iterable, out_b.getdata()))


# --------------------------------------------------------------------
# 11. Validation/test cannot go through the training augmentation path
# --------------------------------------------------------------------

def test_validation_split_raises():
    img = _make_image()
    with pytest.raises(SplitNotAugmentableError):
        augment_image(img, _all_enabled_config(), Split.VALIDATION, seed=1)


def test_test_split_raises():
    img = _make_image()
    with pytest.raises(SplitNotAugmentableError):
        augment_image(img, _all_enabled_config(), Split.TEST, seed=1)


def test_training_augmentor_has_no_split_argument():
    """TrainingAugmentor.augment() takes only an image -- there is no
    split parameter that calling code could set incorrectly."""
    import inspect
    sig = inspect.signature(TrainingAugmentor.augment)
    assert "split" not in sig.parameters


# --------------------------------------------------------------------
# 12. Original source image is not modified
# --------------------------------------------------------------------

def test_original_image_not_mutated():
    img = _make_image(color=(10, 20, 30))
    original_pixels = list(cast(Iterable, img.getdata()))
    augment_image(img, _all_enabled_config(), Split.TRAIN, seed=1)
    assert list(cast(Iterable, img.getdata())) == original_pixels


# --------------------------------------------------------------------
# 13. All enabled augmentations can be composed
# --------------------------------------------------------------------

def test_all_augmentations_compose_without_error():
    img = _make_image(mode="RGB", color=(90, 90, 90))
    out = augment_image(img, _all_enabled_config(), Split.TRAIN, seed=7)
    assert out.size == (128, 128)
    assert out.mode == "RGB"


def test_training_augmentor_end_to_end():
    img = _make_image()
    augmentor = TrainingAugmentor(_all_enabled_config(), seed=42)
    out1 = augmentor.augment(img)
    out2 = augmentor.augment(img)
    assert out1.size == (128, 128)
    assert out2.size == (128, 128)
    # Sequential calls with a fixed base seed should be reproducible
    # as a sequence: re-running from the same base seed reproduces
    # the same first-call output.
    augmentor_replay = TrainingAugmentor(_all_enabled_config(), seed=42)
    out1_replay = augmentor_replay.augment(img)
    assert list(cast(Iterable, out1.getdata())) == list(cast(Iterable, out1_replay.getdata()))


# --------------------------------------------------------------------
# 14. Configuration values are respected
# --------------------------------------------------------------------

def test_config_from_dict_respects_values():
    raw = {
        "enabled": True,
        "rotation": {"enabled": False, "range_degrees": 99},
        "zoom": {"enabled": True, "range_fraction": 0.5},
        "horizontal_flip": {"enabled": False, "probability": 0.9},
        "vertical_flip": {"enabled": True, "probability": 0.7},
        "brightness": {"enabled": False, "range_fraction": 0.9},
    }
    cfg = augmentation_config_from_dict(raw)
    assert cfg.rotation.enabled is False
    assert cfg.zoom.enabled is True
    assert cfg.zoom.range_fraction == 0.5
    assert cfg.horizontal_flip.enabled is False
    assert cfg.vertical_flip.enabled is True
    assert cfg.vertical_flip.probability == 0.7
    assert cfg.brightness.enabled is False


def test_config_from_dict_defaults_vertical_flip_off():
    cfg = augmentation_config_from_dict({})
    assert cfg.vertical_flip.enabled is False


def test_disabled_rotation_flag_prevents_rotation_even_with_seed():
    img = _make_image(color=(255, 0, 0))
    for x in range(20):
        for y in range(20):
            img.putpixel((x, y), (0, 255, 0))

    cfg = replace(_all_disabled_config(), rotation=RotationConfig(enabled=False, range_degrees=45.0))
    out = augment_image(img, cfg, Split.TRAIN, seed=1)
    assert list(cast(Iterable, out.getdata())) == list(cast(Iterable, img.getdata()))


# --------------------------------------------------------------------
# Real 128x128 RGB image, as produced by the actual preprocessing
# pipeline, sanity-checks the whole module end-to-end.
# --------------------------------------------------------------------

def test_realistic_processed_image_roundtrip(tmp_path):
    src_path = tmp_path / "sample.jpg"
    Image.new("RGB", (128, 128), (30, 60, 90)).save(src_path)

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        out = augment_image(img, _all_enabled_config(), Split.TRAIN, seed=11)

    assert out.size == (128, 128)
    assert out.mode == "RGB"