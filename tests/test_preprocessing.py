"""
Tests for src/data/preprocess.py and src/data/loader.py.

All tests use synthetic images built in pytest tmp_path fixtures — none
depend on the real 36,006-image Vehicle-10 dataset. This keeps the
suite fast and runnable without the dataset present on disk.
"""
import csv
import json
import os

import pytest
from PIL import Image

from src.data.loader import build_dataset_index, index_by_split
from src.data.preprocess import (
    remove_exact_duplicates,
    filter_tiny_images,
    carve_test_split,
    build_split_assignments,
    convert_to_rgb,
    run_preprocessing,
)


def _make_image(path, size=(100, 100), mode="RGB", color=(255, 0, 0), unique_marker=None):
    """
    Save a synthetic source image on disk. RGBA and P images can't be
    saved as JPEG directly (PIL limitation), so those are saved as PNG
    instead, matching the .png handling that also occurs in the real
    Vehicle-10 dataset (which has a small number of RGBA/P PNGs).

    unique_marker draws one pixel a different color so otherwise
    solid-color synthetic images don't accidentally hash-collide with
    each other (only genuine, intentional duplicates should collide).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mode == "RGBA":
        img = Image.new("RGBA", size, color)
    elif mode == "P":
        img = Image.new("RGB", size, color[:3] if len(color) == 4 else color).convert("P")
    else:
        img = Image.new(mode, size, color)

    if unique_marker is not None:
        # JPEG is lossy, so a single-pixel marker can get compressed away
        # and cause unrelated same-color images to hash identically.
        # Draw a solid block in the corner instead so it survives
        # JPEG compression and produces genuinely distinct file bytes.
        block = 20
        for x in range(min(block, size[0])):
            for y in range(min(block, size[1])):
                if mode == "P":
                    idx = img.getpixel((x, y))
                    img.putpixel((x, y), (idx + unique_marker[0] + 1) % 256)
                else:
                    img.putpixel((x, y), unique_marker)

    if mode in ("RGBA", "P"):
        png_path = os.path.splitext(path)[0] + ".png"
        img.save(png_path)
        return png_path

    img.save(path)
    return path


@pytest.fixture
def synthetic_dataset(tmp_path):
    """
    Builds a synthetic Vehicle-10-like dataset with:
      - 2 classes: car (label 3), truck (label 9)
      - a mix of RGB, RGBA, P images
      - one tiny image (<64px)
      - one exact duplicate pair
      - official train_meta.json / valid_meta.json
    """
    root = tmp_path / "dataset"

    # --- car class: 6 normal RGB train images, 2 valid images ---
    car_train_paths = []
    for i in range(6):
        p = root / "car" / f"car_{i}.jpg"
        _make_image(str(p), size=(100, 100), mode="RGB", color=(255, 0, 0), unique_marker=(i * 40 % 256, 0, 0))
        car_train_paths.append(f"car/car_{i}.jpg")

    car_valid_paths = []
    for i in range(6, 8):
        p = root / "car" / f"car_{i}.jpg"
        _make_image(str(p), size=(100, 100), mode="RGB", color=(255, 0, 0), unique_marker=(i * 40 % 256, 0, 0))
        car_valid_paths.append(f"car/car_{i}.jpg")

    # RGBA car image (train)
    p = root / "car" / "car_rgba.jpg"
    _make_image(str(p), size=(100, 100), mode="RGBA", color=(0, 255, 0, 128))
    car_train_paths.append("car/car_rgba.png")

    # P-mode car image (train)
    p = root / "car" / "car_p.jpg"
    _make_image(str(p), size=(100, 100), mode="P", color=(10, 20, 30))
    car_train_paths.append("car/car_p.png")

    # tiny car image (train) - should be excluded
    p = root / "car" / "car_tiny.jpg"
    _make_image(str(p), size=(32, 32), mode="RGB", color=(255, 0, 0), unique_marker=(99, 0, 0))
    car_train_paths.append("car/car_tiny.jpg")

    # --- truck class: 6 normal RGB train images, 2 valid images ---
    truck_train_paths = []
    for i in range(6):
        p = root / "truck" / f"truck_{i}.jpg"
        _make_image(str(p), size=(100, 100), mode="RGB", color=(0, 0, 255), unique_marker=(0, 0, i * 40 % 256))
        truck_train_paths.append(f"truck/truck_{i}.jpg")

    truck_valid_paths = []
    for i in range(6, 8):
        p = root / "truck" / f"truck_{i}.jpg"
        _make_image(str(p), size=(100, 100), mode="RGB", color=(0, 0, 255), unique_marker=(0, 0, i * 40 % 256))
        truck_valid_paths.append(f"truck/truck_{i}.jpg")

    # exact duplicate: truck_0_dup.jpg is a byte-identical copy of truck_0.jpg
    src_bytes = (root / "truck" / "truck_0.jpg").read_bytes()
    dup_path = root / "truck" / "truck_0_dup.jpg"
    dup_path.write_bytes(src_bytes)
    truck_train_paths.append("truck/truck_0_dup.jpg")

    train_meta = {
        "path": car_train_paths + truck_train_paths,
        "label": [3] * len(car_train_paths) + [9] * len(truck_train_paths),
    }
    valid_meta = {
        "path": car_valid_paths + truck_valid_paths,
        "label": [3] * len(car_valid_paths) + [9] * len(truck_valid_paths),
    }

    (root / "train_meta.json").write_text(json.dumps(train_meta))
    (root / "valid_meta.json").write_text(json.dumps(valid_meta))

    return {
        "root": str(root),
        "car_train_count_before_filtering": len(car_train_paths),
        "truck_train_count_before_filtering": len(truck_train_paths),
    }


@pytest.fixture
def base_config(tmp_path):
    return {
        "dataset": {
            "raw_root": "unused-overridden-in-tests",
            "processed_root": str(tmp_path / "processed"),
            "classes": [
                "bicycle", "boat", "bus", "car", "helicopter",
                "minibus", "motorcycle", "taxi", "train", "truck",
            ],
        },
        "preprocessing": {
            "image_size": [32, 32],
            "min_image_size": 64,
            "test_split_fraction": 0.25,
            "seed": 42,
            "output_format": "jpg",
            "jpeg_quality": 90,
        },
        "manifest": {"filename": "manifest.csv"},
    }


# --------------------------------------------------------------------
# RGB / RGBA / P conversion
# --------------------------------------------------------------------

def test_convert_rgb_passthrough():
    img = Image.new("RGB", (10, 10), (1, 2, 3))
    out = convert_to_rgb(img)
    assert out.mode == "RGB"


def test_convert_rgba_to_rgb():
    img = Image.new("RGBA", (10, 10), (0, 255, 0, 128))
    out = convert_to_rgb(img)
    assert out.mode == "RGB"
    assert out.size == (10, 10)


def test_convert_p_to_rgb():
    img = Image.new("RGB", (10, 10), (10, 20, 30)).convert("P")
    out = convert_to_rgb(img)
    assert out.mode == "RGB"


# --------------------------------------------------------------------
# Duplicate exclusion
# --------------------------------------------------------------------

def test_remove_exact_duplicates(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    before = len(index)
    deduped = remove_exact_duplicates(index)
    # exactly one duplicate pair -> exactly one entry removed
    assert len(deduped) == before - 1

    rel_paths = {e.rel_path for e in deduped}
    # only one of truck_0.jpg / truck_0_dup.jpg should remain
    assert not ("truck/truck_0.jpg" in rel_paths and "truck/truck_0_dup.jpg" in rel_paths)


# --------------------------------------------------------------------
# Tiny image exclusion
# --------------------------------------------------------------------

def test_filter_tiny_images_excludes_small(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    kept, dims = filter_tiny_images(index, min_size=64)
    rel_paths = {e.rel_path for e in kept}
    assert "car/car_tiny.jpg" not in rel_paths
    # normal images survive
    assert "car/car_0.jpg" in rel_paths
    assert dims["car/car_0.jpg"] == (100, 100)


def test_filter_tiny_images_threshold_is_configurable(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    # with a very low threshold, even the tiny image survives
    kept, _ = filter_tiny_images(index, min_size=16)
    rel_paths = {e.rel_path for e in kept}
    assert "car/car_tiny.jpg" in rel_paths


# --------------------------------------------------------------------
# Split determinism & no-overlap
# --------------------------------------------------------------------

def test_split_determinism_same_seed_same_result(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    index = remove_exact_duplicates(index)
    index, _ = filter_tiny_images(index, min_size=64)

    split_a = build_split_assignments(index, test_fraction=0.25, seed=42)
    split_b = build_split_assignments(index, test_fraction=0.25, seed=42)

    paths_a = {name: sorted(e.rel_path for e in entries) for name, entries in split_a.items()}
    paths_b = {name: sorted(e.rel_path for e in entries) for name, entries in split_b.items()}
    assert paths_a == paths_b


def test_split_different_seed_can_differ(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    index = remove_exact_duplicates(index)
    index, _ = filter_tiny_images(index, min_size=64)

    split_a = build_split_assignments(index, test_fraction=0.25, seed=1)
    split_b = build_split_assignments(index, test_fraction=0.25, seed=2)

    test_a = sorted(e.rel_path for e in split_a["test"])
    test_b = sorted(e.rel_path for e in split_b["test"])
    assert test_a != test_b


def test_official_validation_split_untouched(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    index = remove_exact_duplicates(index)
    index, _ = filter_tiny_images(index, min_size=64)

    by_split = index_by_split(index)
    official_valid_paths = sorted(e.rel_path for e in by_split["valid"])

    splits = build_split_assignments(index, test_fraction=0.25, seed=42)
    new_valid_paths = sorted(e.rel_path for e in splits["validation"])

    assert official_valid_paths == new_valid_paths


def test_test_set_only_carved_from_official_train(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    index = remove_exact_duplicates(index)
    index, _ = filter_tiny_images(index, min_size=64)

    by_split = index_by_split(index)
    official_train_paths = {e.rel_path for e in by_split["train"]}

    splits = build_split_assignments(index, test_fraction=0.25, seed=42)
    test_paths = {e.rel_path for e in splits["test"]}

    assert test_paths.issubset(official_train_paths)


def test_no_overlap_across_splits(synthetic_dataset):
    index = build_dataset_index(synthetic_dataset["root"])
    index = remove_exact_duplicates(index)
    index, _ = filter_tiny_images(index, min_size=64)

    splits = build_split_assignments(index, test_fraction=0.25, seed=42)
    train_paths = {e.rel_path for e in splits["train"]}
    valid_paths = {e.rel_path for e in splits["validation"]}
    test_paths = {e.rel_path for e in splits["test"]}

    assert train_paths.isdisjoint(valid_paths)
    assert train_paths.isdisjoint(test_paths)
    assert valid_paths.isdisjoint(test_paths)


def test_carve_test_split_is_stratified_per_class():
    from src.data.loader import DatasetEntry

    entries = [
        DatasetEntry(f"/x/car_{i}.jpg", f"car/car_{i}.jpg", "car", "train")
        for i in range(8)
    ] + [
        DatasetEntry(f"/x/truck_{i}.jpg", f"truck/truck_{i}.jpg", "truck", "train")
        for i in range(4)
    ]
    train, test = carve_test_split(entries, test_fraction=0.25, seed=7)

    test_car_count = sum(1 for e in test if e.class_name == "car")
    test_truck_count = sum(1 for e in test if e.class_name == "truck")
    assert test_car_count == 2  # 25% of 8
    assert test_truck_count == 1  # 25% of 4 (rounded)


# --------------------------------------------------------------------
# Full pipeline + manifest correctness
# --------------------------------------------------------------------

def test_run_preprocessing_end_to_end_and_manifest(synthetic_dataset, base_config):
    manifest_path = run_preprocessing(base_config, dataset_root=synthetic_dataset["root"])

    assert manifest_path.exists()

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0

    # manifest schema
    expected_fields = {
        "rel_output_path", "class_name", "split",
        "original_width", "original_height",
        "processed_width", "processed_height", "source_rel_path",
    }
    assert expected_fields.issubset(set(rows[0].keys()))

    # tiny image must not appear anywhere in the manifest
    assert not any(r["source_rel_path"] == "car/car_tiny.jpg" for r in rows)

    # duplicate: only one of truck_0.jpg / truck_0_dup.jpg present
    dup_sources = {r["source_rel_path"] for r in rows} & {
        "truck/truck_0.jpg", "truck/truck_0_dup.jpg"
    }
    assert len(dup_sources) == 1

    # processed images actually exist on disk at the recorded path, at
    # the configured output size
    processed_root = manifest_path.parent
    for row in rows:
        out_file = processed_root / row["rel_output_path"]
        assert out_file.exists()
        with Image.open(out_file) as img:
            assert img.mode == "RGB"
            assert img.size == (32, 32)  # configured image_size

    # splits present are exactly train/validation/test
    assert set(r["split"] for r in rows).issubset({"train", "validation", "test"})


def test_run_preprocessing_is_deterministic(synthetic_dataset, base_config, tmp_path):
    manifest_path_1 = run_preprocessing(base_config, dataset_root=synthetic_dataset["root"])
    with open(manifest_path_1, newline="") as f:
        rows_1 = sorted(
            (r["source_rel_path"], r["split"]) for r in csv.DictReader(f)
        )

    # run again into a fresh processed_root with the same seed
    base_config["dataset"]["processed_root"] = str(tmp_path / "processed_2")
    manifest_path_2 = run_preprocessing(base_config, dataset_root=synthetic_dataset["root"])
    with open(manifest_path_2, newline="") as f:
        rows_2 = sorted(
            (r["source_rel_path"], r["split"]) for r in csv.DictReader(f)
        )

    assert rows_1 == rows_2


def test_raw_dataset_untouched_by_preprocessing(synthetic_dataset, base_config):
    before = sorted(os.listdir(os.path.join(synthetic_dataset["root"], "car")))
    run_preprocessing(base_config, dataset_root=synthetic_dataset["root"])
    after = sorted(os.listdir(os.path.join(synthetic_dataset["root"], "car")))
    assert before == after