"""
Tests for src/data/validator.py — dataset discovery and metadata loading.

These tests build small synthetic datasets in a pytest tmp_path fixture,
so they do NOT depend on the real Vehicle-10 dataset being present on
disk. This keeps the test suite fast and runnable in CI or on a machine
that hasn't downloaded the dataset yet.
"""
import json
import os
import pytest
from PIL import Image

from src.data.validator import (
    discover_class_folders,
    check_extensions,
    load_metadata,
    check_metadata_paths_exist,
    check_label_folder_consistency,
    check_train_valid_overlap,
    class_distribution,
    LABEL_TO_CLASS,
)


def _make_image(path, size=(64, 64), mode="RGB"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new(mode, size).save(path)


@pytest.fixture
def mini_dataset(tmp_path):
    """
    Builds a tiny 2-class dataset (car, truck) with 3 images each,
    plus train/valid metadata files, mirroring Vehicle-10's structure.
    """
    root = tmp_path / "dataset"
    car_dir = root / "car"
    truck_dir = root / "truck"

    for i in range(3):
        _make_image(str(car_dir / f"car_{i}.jpg"))
    for i in range(3):
        _make_image(str(truck_dir / f"truck_{i}.jpg"))

    train_meta = {
        "path": ["car/car_0.jpg", "car/car_1.jpg", "truck/truck_0.jpg", "truck/truck_1.jpg"],
        "label": [3, 3, 9, 9],  # matches Vehicle-10's official label indices
    }
    valid_meta = {
        "path": ["car/car_2.jpg", "truck/truck_2.jpg"],
        "label": [3, 9],
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


def test_discover_class_folders_finds_all_classes(mini_dataset):
    class_files = discover_class_folders(mini_dataset["root"])
    assert set(class_files.keys()) == {"car", "truck"}
    assert len(class_files["car"]) == 3
    assert len(class_files["truck"]) == 3


def test_discover_class_folders_ignores_non_directory_files(mini_dataset, tmp_path):
    # a stray file at the dataset root (e.g. a README) should not be treated as a class
    stray_file = os.path.join(mini_dataset["root"], "notes.txt")
    with open(stray_file, "w") as f:
        f.write("not a class folder")

    class_files = discover_class_folders(mini_dataset["root"])
    assert "notes.txt" not in class_files
    assert set(class_files.keys()) == {"car", "truck"}


def test_check_extensions_counts_correctly(mini_dataset):
    class_files = discover_class_folders(mini_dataset["root"])
    ext_counts = check_extensions(class_files)
    assert ext_counts[".jpg"] == 6


def test_load_metadata_returns_matching_pairs(mini_dataset):
    entries = load_metadata(mini_dataset["train_meta_path"])
    assert len(entries) == 4
    assert ("car/car_0.jpg", 3) in entries
    assert ("truck/truck_0.jpg", 9) in entries


def test_load_metadata_raises_on_missing_keys(tmp_path):
    bad_meta = tmp_path / "bad_meta.json"
    bad_meta.write_text(json.dumps({"path": ["a.jpg"]}))  # missing 'label'
    with pytest.raises(ValueError):
        load_metadata(str(bad_meta))


def test_load_metadata_raises_on_length_mismatch(tmp_path):
    bad_meta = tmp_path / "bad_meta.json"
    bad_meta.write_text(json.dumps({"path": ["a.jpg", "b.jpg"], "label": [0]}))
    with pytest.raises(ValueError):
        load_metadata(str(bad_meta))


def test_check_metadata_paths_exist_detects_missing_file(mini_dataset):
    entries = load_metadata(mini_dataset["train_meta_path"])
    entries.append(("car/does_not_exist.jpg", 3))
    missing = check_metadata_paths_exist(mini_dataset["root"], entries)
    assert ("car/does_not_exist.jpg", 3) in missing
    assert len(missing) == 1


def test_check_label_folder_consistency_flags_mismatch(mini_dataset):
    entries = load_metadata(mini_dataset["train_meta_path"])
    # Inject an entry where the label (9 = truck) doesn't match the folder (car)
    entries.append(("car/car_0.jpg", 9))
    mismatches = check_label_folder_consistency(entries, LABEL_TO_CLASS)
    assert len(mismatches) == 1
    assert mismatches[0][0] == "car/car_0.jpg"
    assert mismatches[0][2] == "truck"  # expected class for label 9
    assert mismatches[0][3] == "car"    # actual folder


def test_check_label_folder_consistency_no_false_positives(mini_dataset):
    entries = load_metadata(mini_dataset["train_meta_path"])
    mismatches = check_label_folder_consistency(entries, LABEL_TO_CLASS)
    assert mismatches == []


def test_check_train_valid_overlap_detects_leakage(mini_dataset):
    train_entries = load_metadata(mini_dataset["train_meta_path"])
    valid_entries = load_metadata(mini_dataset["valid_meta_path"])
    # No overlap expected in the clean fixture
    overlap = check_train_valid_overlap(train_entries, valid_entries)
    assert overlap == set()

    # Now inject a leaked path into validation
    valid_entries.append(("car/car_0.jpg", 3))
    overlap = check_train_valid_overlap(train_entries, valid_entries)
    assert "car/car_0.jpg" in overlap


def test_class_distribution_counts_by_class_name(mini_dataset):
    entries = load_metadata(mini_dataset["train_meta_path"])
    dist = class_distribution(entries, LABEL_TO_CLASS)
    assert dist["car"] == 2
    assert dist["truck"] == 2