"""
tests/test_phase9_gap_coverage.py

Phase 9 audit: targeted tests for functions that had zero coverage in
the existing suite, discovered by cross-referencing every public
function in src/data/loader.py and src/data/validator.py against what
the existing test files actually exercise.

Gaps filled here:
  - src/data/loader.py: load_config() (YAML file reading) was never
    tested directly -- run_preprocessing() in test_preprocessing.py
    always builds config dicts in Python, bypassing this function
    entirely.
  - src/data/validator.py: run_full_validation() (the orchestrator),
    check_image_integrity(), and find_exact_duplicates() had no tests
    of their own; only their lower-level building blocks
    (discover_class_folders, load_metadata, check_*) were tested.

All tests use synthetic tmp_path fixtures. No real dataset, model, or
project path is touched.
"""
import json
import os

import pytest
import yaml
from PIL import Image

from src.data.loader import load_config
from src.data.validator import (
    check_image_integrity,
    discover_class_folders,
    find_exact_duplicates,
    run_full_validation,
)


# --------------------------------------------------------------------
# loader.load_config()
# --------------------------------------------------------------------

def test_load_config_reads_yaml_correctly(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "dataset:\n"
        "  raw_root: /tmp/x\n"
        "  classes: [car, truck]\n"
        "preprocessing:\n"
        "  image_size: [64, 64]\n"
    )
    config = load_config(str(cfg_path))
    assert config["dataset"]["raw_root"] == "/tmp/x"
    assert config["dataset"]["classes"] == ["car", "truck"]
    assert config["preprocessing"]["image_size"] == [64, 64]


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.yaml"))


def test_load_config_invalid_yaml_raises(tmp_path):
    bad_path = tmp_path / "bad.yaml"
    # Unbalanced flow-mapping brackets: invalid YAML syntax.
    bad_path.write_text("dataset: [unterminated\n  nested: {a: b\n")
    with pytest.raises(yaml.YAMLError):
        load_config(str(bad_path))


# --------------------------------------------------------------------
# validator.check_image_integrity()
# --------------------------------------------------------------------

def _make_image(path, size=(64, 64), mode="RGB"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new(mode, size).save(path)


def test_check_image_integrity_reports_modes_and_dims(tmp_path):
    root = tmp_path / "dataset"
    _make_image(str(root / "car" / "a.jpg"), size=(80, 60), mode="RGB")
    _make_image(str(root / "car" / "b.png"), size=(40, 40), mode="RGBA")

    class_files = discover_class_folders(str(root))
    corrupted, mode_counter, dims = check_image_integrity(class_files)

    assert corrupted == []
    assert mode_counter["RGB"] == 1
    assert mode_counter["RGBA"] == 1
    dims_by_name = {os.path.basename(p): (w, h) for p, w, h in dims}
    assert dims_by_name["a.jpg"] == (80, 60)
    assert dims_by_name["b.png"] == (40, 40)


def test_check_image_integrity_flags_corrupted_file(tmp_path):
    root = tmp_path / "dataset"
    class_dir = root / "car"
    os.makedirs(class_dir, exist_ok=True)
    # Valid image alongside one corrupted "image" (not real image bytes).
    _make_image(str(class_dir / "good.jpg"), size=(64, 64))
    bad_path = class_dir / "bad.jpg"
    bad_path.write_bytes(b"this is not a real image file")

    class_files = discover_class_folders(str(root))
    corrupted, mode_counter, dims = check_image_integrity(class_files)

    corrupted_paths = {p for p, _err in corrupted}
    assert str(bad_path) in corrupted_paths
    # The corrupted file must not appear in dims (only survivors do).
    assert not any(os.path.basename(p) == "bad.jpg" for p, _, _ in dims)


# --------------------------------------------------------------------
# validator.find_exact_duplicates()
# --------------------------------------------------------------------

def test_find_exact_duplicates_detects_byte_identical_files(tmp_path):
    root = tmp_path / "dataset"
    class_dir = root / "car"
    os.makedirs(class_dir, exist_ok=True)

    _make_image(str(class_dir / "orig.jpg"), size=(64, 64), mode="RGB")
    # Byte-identical copy.
    src_bytes = (class_dir / "orig.jpg").read_bytes()
    (class_dir / "dup.jpg").write_bytes(src_bytes)
    # A distinct, non-duplicate image.
    _make_image(str(class_dir / "other.jpg"), size=(32, 32), mode="RGB")

    class_files = discover_class_folders(str(root))
    dup_groups = find_exact_duplicates(class_files)

    assert len(dup_groups) == 1
    group = next(iter(dup_groups.values()))
    basenames = {os.path.basename(p) for p in group}
    assert basenames == {"orig.jpg", "dup.jpg"}


def test_find_exact_duplicates_empty_when_all_unique(tmp_path):
    root = tmp_path / "dataset"
    class_dir = root / "car"
    os.makedirs(class_dir, exist_ok=True)
    _make_image(str(class_dir / "a.jpg"), size=(64, 64), mode="RGB")
    _make_image(str(class_dir / "b.jpg"), size=(32, 32), mode="RGB")

    class_files = discover_class_folders(str(root))
    dup_groups = find_exact_duplicates(class_files)
    assert dup_groups == {}


# --------------------------------------------------------------------
# validator.run_full_validation() -- the orchestrator
# --------------------------------------------------------------------

@pytest.fixture
def mini_validation_dataset(tmp_path):
    """A tiny 2-class dataset with one corrupted file, one duplicate
    pair, one tiny image, and a train/valid metadata mismatch, so a
    single run_full_validation() call exercises every report section."""
    root = tmp_path / "dataset"

    _make_image(str(root / "car" / "car_0.jpg"), size=(100, 100))
    _make_image(str(root / "car" / "car_1.jpg"), size=(100, 100))
    _make_image(str(root / "car" / "car_tiny.jpg"), size=(10, 10))

    # Corrupted file.
    corrupt_path = root / "car" / "car_bad.jpg"
    os.makedirs(corrupt_path.parent, exist_ok=True)
    corrupt_path.write_bytes(b"not an image")

    # Duplicate of car_0.
    dup_bytes = (root / "car" / "car_0.jpg").read_bytes()
    (root / "car" / "car_0_dup.jpg").write_bytes(dup_bytes)

    _make_image(str(root / "truck" / "truck_0.jpg"), size=(100, 100))

    train_meta = {
        "path": ["car/car_0.jpg", "car/car_1.jpg", "truck/truck_0.jpg"],
        "label": [3, 3, 9],
    }
    # Validation references a file that doesn't exist on disk, and
    # overlaps with a train path -- both should surface in the report.
    valid_meta = {
        "path": ["car/car_0.jpg", "car/missing.jpg"],
        "label": [3, 3],
    }

    train_meta_path = root / "train_meta.json"
    valid_meta_path = root / "valid_meta.json"
    train_meta_path.write_text(json.dumps(train_meta))
    valid_meta_path.write_text(json.dumps(valid_meta))

    return {
        "root": str(root),
        "train_meta_path": str(train_meta_path),
        "valid_meta_path": str(valid_meta_path),
    }


def test_run_full_validation_produces_complete_report(mini_validation_dataset):
    report = run_full_validation(
        mini_validation_dataset["root"],
        mini_validation_dataset["train_meta_path"],
        mini_validation_dataset["valid_meta_path"],
    )

    expected_keys = {
        "folder_counts", "extensions", "corrupted_images", "image_modes",
        "dimension_summary", "exact_duplicates", "train_missing_files",
        "valid_missing_files", "train_label_mismatches",
        "valid_label_mismatches", "train_valid_overlap",
        "train_class_distribution", "valid_class_distribution",
        "metadata_vs_folder_count_diff",
    }
    assert expected_keys.issubset(report.keys())


def test_run_full_validation_detects_corruption(mini_validation_dataset):
    report = run_full_validation(
        mini_validation_dataset["root"],
        mini_validation_dataset["train_meta_path"],
        mini_validation_dataset["valid_meta_path"],
    )
    corrupted_names = {os.path.basename(p) for p, _err in report["corrupted_images"]}
    assert "car_bad.jpg" in corrupted_names


def test_run_full_validation_detects_duplicates(mini_validation_dataset):
    report = run_full_validation(
        mini_validation_dataset["root"],
        mini_validation_dataset["train_meta_path"],
        mini_validation_dataset["valid_meta_path"],
    )
    assert len(report["exact_duplicates"]) == 1


def test_run_full_validation_detects_tiny_images(mini_validation_dataset):
    report = run_full_validation(
        mini_validation_dataset["root"],
        mini_validation_dataset["train_meta_path"],
        mini_validation_dataset["valid_meta_path"],
    )
    tiny_names = {os.path.basename(p) for p, _w, _h in report["dimension_summary"]["tiny_images_under_64px"]}
    assert "car_tiny.jpg" in tiny_names


def test_run_full_validation_detects_missing_metadata_file(mini_validation_dataset):
    report = run_full_validation(
        mini_validation_dataset["root"],
        mini_validation_dataset["train_meta_path"],
        mini_validation_dataset["valid_meta_path"],
    )
    valid_missing_paths = {p for p, _label in report["valid_missing_files"]}
    assert "car/missing.jpg" in valid_missing_paths


def test_run_full_validation_detects_train_valid_overlap(mini_validation_dataset):
    report = run_full_validation(
        mini_validation_dataset["root"],
        mini_validation_dataset["train_meta_path"],
        mini_validation_dataset["valid_meta_path"],
    )
    assert "car/car_0.jpg" in report["train_valid_overlap"]


def test_run_full_validation_is_read_only(mini_validation_dataset):
    """run_full_validation() must never modify, move, or delete files
    under dataset_root -- confirmed by comparing directory listings
    before and after."""
    root = mini_validation_dataset["root"]
    before = sorted(
        os.path.relpath(os.path.join(dp, f), root)
        for dp, _, files in os.walk(root)
        for f in files
    )

    run_full_validation(
        root,
        mini_validation_dataset["train_meta_path"],
        mini_validation_dataset["valid_meta_path"],
    )

    after = sorted(
        os.path.relpath(os.path.join(dp, f), root)
        for dp, _, files in os.walk(root)
        for f in files
    )
    assert before == after