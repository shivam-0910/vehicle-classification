"""
src/data/validator.py

Dataset validation for the Vehicle-10 dataset.

This module performs READ-ONLY inspection of the dataset — it never
modifies, moves, deletes, or renames any file. It is meant to be run
once (as a script) to produce a validation report BEFORE any
preprocessing, splitting, or training code is written.

Usage (from repo root, after activating .venv):

    python -m src.data.validator --dataset-root "D:\\ml-datasets\\vehicle-10"

Or import the functions directly for use in tests / notebooks:

    from src.data.validator import run_full_validation
    report = run_full_validation(root, train_meta_path, valid_meta_path)
"""
import os
import json
import hashlib
import argparse
from collections import Counter, defaultdict
from PIL import Image

# Vehicle-10 official class -> label mapping (from dataset README / metadata)
EXPECTED_CLASSES = [
    "bicycle", "boat", "bus", "car", "helicopter",
    "minibus", "motorcycle", "taxi", "train", "truck",
]
LABEL_TO_CLASS = {i: c for i, c in enumerate(EXPECTED_CLASSES)}

# Official reported counts from the Vehicle-10 README, used only as a
# reference to compare against what we actually find on disk.
REPORTED_FOLDER_COUNTS = {
    "bicycle": 1618,
    "boat": 8897,
    "bus": 4064,
    "car": 8540,
    "helicopter": 668,
    "minibus": 1477,
    "motorcycle": 4438,
    "taxi": 908,
    "train": 1682,
    "truck": 3714,
}


def discover_class_folders(dataset_root):
    """Return {class_name: [absolute file paths]} for every subfolder found."""
    result = {}
    for entry in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, entry)
        if os.path.isdir(full):
            files = [
                os.path.join(full, f)
                for f in os.listdir(full)
                if os.path.isfile(os.path.join(full, f))
            ]
            result[entry] = files
    return result


def check_extensions(class_files):
    """Return Counter of file extensions found across all classes."""
    ext_counter = Counter()
    for files in class_files.values():
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            ext_counter[ext] += 1
    return ext_counter


def check_image_integrity(class_files):
    """
    Attempt to open + verify every image with PIL.

    Returns:
        corrupted: list of (path, error_message)
        mode_counter: Counter of PIL image modes (RGB, L, RGBA, ...)
        dims: list of (path, width, height)
    """
    corrupted = []
    mode_counter = Counter()
    dims = []

    for files in class_files.values():
        for f in files:
            try:
                with Image.open(f) as img:
                    img.verify()  # cheap corruption check, invalidates handle
                with Image.open(f) as img:  # reopen to safely read mode/size
                    mode_counter[img.mode] += 1
                    dims.append((f, img.width, img.height))
            except Exception as e:
                corrupted.append((f, str(e)))

    return corrupted, mode_counter, dims


def compute_file_hash(path, chunk_size=65536):
    """MD5 hash of raw file bytes — used for exact-duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(class_files):
    """Return {hash: [paths]} restricted to hashes that appear more than once."""
    hash_map = defaultdict(list)
    for files in class_files.values():
        for f in files:
            try:
                h = compute_file_hash(f)
                hash_map[h].append(f)
            except Exception:
                continue  # unreadable files are already reported by the integrity check
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def load_metadata(meta_path):
    """Load a Vehicle-10 metadata JSON file (train_meta.json / valid_meta.json)."""
    with open(meta_path, "r") as f:
        data = json.load(f)
    if "path" not in data or "label" not in data:
        raise ValueError(f"{meta_path} missing expected 'path'/'label' keys")
    if len(data["path"]) != len(data["label"]):
        raise ValueError(f"{meta_path} path/label length mismatch")
    return list(zip(data["path"], data["label"]))


def check_metadata_paths_exist(dataset_root, meta_entries):
    """Return list of (path, label) entries whose file is missing on disk."""
    missing = []
    for rel_path, label in meta_entries:
        full = os.path.join(dataset_root, rel_path)
        if not os.path.isfile(full):
            missing.append((rel_path, label))
    return missing


def check_label_folder_consistency(meta_entries, label_to_class):
    """
    Return list of (path, label, expected_class, actual_folder) for entries
    where the metadata label's expected class doesn't match the path's
    actual parent folder name.
    """
    mismatches = []
    for rel_path, label in meta_entries:
        expected_class = label_to_class.get(label)
        normalized = rel_path.replace("\\", "/")
        actual_folder = normalized.split("/")[0]
        if expected_class != actual_folder:
            mismatches.append((rel_path, label, expected_class, actual_folder))
    return mismatches


def check_train_valid_overlap(train_entries, valid_entries):
    """Return the set of relative paths present in BOTH train and validation metadata."""
    train_paths = {p for p, _ in train_entries}
    valid_paths = {p for p, _ in valid_entries}
    return train_paths & valid_paths


def class_distribution(meta_entries, label_to_class):
    """Return Counter of class name -> count, derived from metadata labels."""
    counter = Counter()
    for _, label in meta_entries:
        counter[label_to_class.get(label, f"UNKNOWN_LABEL_{label}")] += 1
    return counter


def run_full_validation(dataset_root, train_meta_path, valid_meta_path):
    """
    Orchestrates every validation check and returns a single structured
    report dict. This function is READ-ONLY: it never writes, moves, or
    deletes anything in dataset_root.
    """
    report = {}

    # --- Folder scan, extensions, integrity, modes, dimensions ---
    class_files = discover_class_folders(dataset_root)
    report["folder_counts"] = {c: len(f) for c, f in class_files.items()}
    report["extensions"] = dict(check_extensions(class_files))

    corrupted, modes, dims = check_image_integrity(class_files)
    report["corrupted_images"] = corrupted
    report["image_modes"] = dict(modes)

    widths = [w for _, w, h in dims]
    heights = [h for _, w, h in dims]
    report["dimension_summary"] = {
        "min_width": min(widths) if widths else None,
        "max_width": max(widths) if widths else None,
        "min_height": min(heights) if heights else None,
        "max_height": max(heights) if heights else None,
        "tiny_images_under_64px": [
            (p, w, h) for p, w, h in dims if w < 64 or h < 64
        ],
    }

    # --- Exact duplicate detection ---
    report["exact_duplicates"] = find_exact_duplicates(class_files)

    # --- Metadata checks ---
    train_entries = load_metadata(train_meta_path)
    valid_entries = load_metadata(valid_meta_path)

    report["train_missing_files"] = check_metadata_paths_exist(dataset_root, train_entries)
    report["valid_missing_files"] = check_metadata_paths_exist(dataset_root, valid_entries)

    report["train_label_mismatches"] = check_label_folder_consistency(train_entries, LABEL_TO_CLASS)
    report["valid_label_mismatches"] = check_label_folder_consistency(valid_entries, LABEL_TO_CLASS)

    report["train_valid_overlap"] = sorted(check_train_valid_overlap(train_entries, valid_entries))

    report["train_class_distribution"] = dict(class_distribution(train_entries, LABEL_TO_CLASS))
    report["valid_class_distribution"] = dict(class_distribution(valid_entries, LABEL_TO_CLASS))

    # --- Metadata totals vs. actual folder counts ---
    combined_meta_counts = Counter()
    for cls, count in report["train_class_distribution"].items():
        combined_meta_counts[cls] += count
    for cls, count in report["valid_class_distribution"].items():
        combined_meta_counts[cls] += count

    report["metadata_vs_folder_count_diff"] = {
        cls: {
            "folder_count": report["folder_counts"].get(cls, 0),
            "metadata_count": combined_meta_counts.get(cls, 0),
            "reported_count": REPORTED_FOLDER_COUNTS.get(cls),
            "difference_folder_minus_metadata": (
                report["folder_counts"].get(cls, 0) - combined_meta_counts.get(cls, 0)
            ),
        }
        for cls in EXPECTED_CLASSES
    }

    return report


def print_report_summary(report):
    """Human-readable console summary of a validation report."""
    print("=== FOLDER COUNTS ===")
    for c, n in report["folder_counts"].items():
        print(f"  {c}: {n}")

    print("\n=== EXTENSIONS FOUND ===")
    print(" ", report["extensions"])

    print("\n=== IMAGE MODES (RGB / L / RGBA / ...) ===")
    print(" ", report["image_modes"])

    print("\n=== DIMENSIONS ===")
    ds = report["dimension_summary"]
    print(f"  width range:  {ds['min_width']} - {ds['max_width']}")
    print(f"  height range: {ds['min_height']} - {ds['max_height']}")
    print(f"  tiny images (<64px on either side): {len(ds['tiny_images_under_64px'])}")

    print("\n=== CORRUPTED / UNREADABLE IMAGES ===")
    print(f"  count: {len(report['corrupted_images'])}")
    for p, err in report["corrupted_images"][:20]:
        print(f"    {p} -> {err}")
    if len(report["corrupted_images"]) > 20:
        print(f"    ... and {len(report['corrupted_images']) - 20} more")

    print("\n=== EXACT DUPLICATE IMAGES (hash-based) ===")
    print(f"  duplicate hash groups: {len(report['exact_duplicates'])}")
    for h, paths in list(report["exact_duplicates"].items())[:10]:
        print(f"    {h[:10]}...: {len(paths)} files")
    if len(report["exact_duplicates"]) > 10:
        print(f"    ... and {len(report['exact_duplicates']) - 10} more groups")

    print("\n=== METADATA: PATHS MISSING ON DISK ===")
    print(f"  train missing: {len(report['train_missing_files'])}")
    print(f"  valid missing: {len(report['valid_missing_files'])}")

    print("\n=== METADATA: LABEL <-> FOLDER MISMATCHES ===")
    print(f"  train mismatches: {len(report['train_label_mismatches'])}")
    print(f"  valid mismatches: {len(report['valid_label_mismatches'])}")

    print("\n=== TRAIN / VALIDATION OVERLAP (LEAKAGE CHECK) ===")
    print(f"  overlapping paths: {len(report['train_valid_overlap'])}")
    for p in report["train_valid_overlap"][:20]:
        print(f"    {p}")

    print("\n=== CLASS DISTRIBUTION — TRAIN (from train_meta.json) ===")
    print(" ", report["train_class_distribution"])

    print("\n=== CLASS DISTRIBUTION — VALIDATION (from valid_meta.json) ===")
    print(" ", report["valid_class_distribution"])

    print("\n=== METADATA TOTAL vs FOLDER COUNT vs README-REPORTED COUNT ===")
    for cls, d in report["metadata_vs_folder_count_diff"].items():
        flag = "  <-- CHECK THIS" if d["difference_folder_minus_metadata"] != 0 else ""
        print(
            f"  {cls}: folder={d['folder_count']} "
            f"metadata_total={d['metadata_count']} "
            f"readme_reported={d['reported_count']} "
            f"diff(folder-metadata)={d['difference_folder_minus_metadata']}{flag}"
        )


def main():
    parser = argparse.ArgumentParser(description="Validate the Vehicle-10 dataset.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the Vehicle-10 dataset root "
             "(the folder containing bus/, car/, ..., train_meta.json, valid_meta.json).",
    )
    args = parser.parse_args()

    train_meta_path = os.path.join(args.dataset_root, "train_meta.json")
    valid_meta_path = os.path.join(args.dataset_root, "valid_meta.json")

    report = run_full_validation(args.dataset_root, train_meta_path, valid_meta_path)
    print_report_summary(report)


if __name__ == "__main__":
    main()