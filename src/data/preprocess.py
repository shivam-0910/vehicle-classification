"""
src/data/preprocess.py

Preprocessing pipeline for the Vehicle-10 dataset.

Pipeline stages (in order):
  1. Build the raw dataset index (via loader.build_dataset_index).
  2. Drop exact duplicate files (keep one copy per duplicate group),
     BEFORE any splitting, so a duplicate can never land in two
     different splits.
  3. Drop tiny images (width < min_image_size OR height < min_image_size),
     checked from actual on-disk image dimensions.
  4. Preserve the official train/valid split from the dataset metadata.
     Carve a new "test" split out of the official *training* entries
     only (stratified per class, fixed seed). The official validation
     split is left untouched and becomes the new "validation" split.
  5. For every surviving entry: open the image, convert mode safely to
     RGB (handles RGB / RGBA / P and any other PIL mode), resize to the
     configured target size, and save it under
     data/processed/{train,validation,test}/{class}/.
     Images are processed and written ONE AT A TIME — the full dataset
     is never held in memory at once.
  6. Write a manifest (CSV) describing every processed image: path,
     class, split, original dimensions, processed dimensions.

Determinism: splitting uses Python's `random.Random(seed)` with a
fixed seed from config, so re-running preprocessing on an unchanged
raw dataset always reproduces the exact same train/validation/test
membership.

The raw dataset is never modified, moved, or deleted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from src.data.loader import DatasetEntry, build_dataset_index, load_config, index_by_split


# --------------------------------------------------------------------------
# Stage: exact duplicate removal
# --------------------------------------------------------------------------

def _file_hash(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_exact_duplicates(entries: List[DatasetEntry]) -> List[DatasetEntry]:
    """
    Keep only the first occurrence (by stable input order) of each
    exact-duplicate file (identical byte content). This must run
    BEFORE splitting so a duplicate pair can never straddle two splits.
    """
    seen_hashes = set()
    kept: List[DatasetEntry] = []
    for entry in entries:
        h = _file_hash(entry.abs_path)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        kept.append(entry)
    return kept


# --------------------------------------------------------------------------
# Stage: tiny image filtering
# --------------------------------------------------------------------------

def filter_tiny_images(
    entries: List[DatasetEntry], min_size: int
) -> Tuple[List[DatasetEntry], Dict[str, Tuple[int, int]]]:
    """
    Drop entries whose width or height (as stored on disk) is below
    min_size. Returns the surviving entries plus a dict mapping
    rel_path -> (orig_width, orig_height) for every SURVIVING entry
    (used later so we don't have to reopen the file to record original
    dimensions in the manifest).
    """
    kept: List[DatasetEntry] = []
    dims: Dict[str, Tuple[int, int]] = {}
    for entry in entries:
        try:
            with Image.open(entry.abs_path) as img:
                w, h = img.width, img.height
        except Exception:
            # Unreadable files are out of scope here (validator.py already
            # reports corruption); skip defensively rather than crash the run.
            continue
        if w < min_size or h < min_size:
            continue
        kept.append(entry)
        dims[entry.rel_path] = (w, h)
    return kept, dims


# --------------------------------------------------------------------------
# Stage: deterministic split (preserve official train/valid, carve test)
# --------------------------------------------------------------------------

def carve_test_split(
    train_entries: List[DatasetEntry],
    test_fraction: float,
    seed: int,
) -> Tuple[List[DatasetEntry], List[DatasetEntry]]:
    """
    Stratified, deterministic carve of a test set out of the official
    training entries only. Validation entries are never touched by
    this function.

    Returns:
        (new_train_entries, new_test_entries)
    """
    by_class: Dict[str, List[DatasetEntry]] = {}
    for entry in train_entries:
        by_class.setdefault(entry.class_name, []).append(entry)

    new_train: List[DatasetEntry] = []
    new_test: List[DatasetEntry] = []

    for class_name in sorted(by_class.keys()):
        class_entries = sorted(by_class[class_name], key=lambda e: e.rel_path)
        rng = random.Random(f"{seed}-{class_name}")
        shuffled = class_entries[:]
        rng.shuffle(shuffled)

        n_test = round(len(shuffled) * test_fraction)
        test_part = shuffled[:n_test]
        train_part = shuffled[n_test:]

        new_test.extend(test_part)
        new_train.extend(train_part)

    return new_train, new_test


def build_split_assignments(
    index: List[DatasetEntry], test_fraction: float, seed: int
) -> Dict[str, List[DatasetEntry]]:
    """
    Full split-assignment stage: preserve official validation, carve
    test out of official training, keep remainder as new training.

    Returns {"train": [...], "validation": [...], "test": [...]}.
    """
    by_split = index_by_split(index)
    official_train = by_split.get("train", [])
    official_valid = by_split.get("valid", [])

    new_train, new_test = carve_test_split(official_train, test_fraction, seed)

    return {
        "train": new_train,
        "validation": official_valid,
        "test": new_test,
    }


# --------------------------------------------------------------------------
# Stage: image conversion + resize + incremental write
# --------------------------------------------------------------------------

def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Safely convert any PIL image mode (RGB, RGBA, P, L, etc.) to RGB.
    RGBA and P (palette) images are alpha-composited onto a white
    background before conversion so transparency doesn't turn black.
    """
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def process_and_save_image(
    entry: DatasetEntry,
    split_name: str,
    processed_root: Path,
    image_size: Tuple[int, int],
    output_format: str,
    jpeg_quality: int,
) -> Tuple[int, int]:
    """
    Open one image, convert to RGB, resize, save under
    processed_root/split_name/class_name/. Returns (out_width, out_height).

    Processes exactly one image at a time — never holds more than one
    decoded image in memory.
    """
    out_dir = processed_root / split_name / entry.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(entry.rel_path).stem
    out_path = out_dir / f"{stem}.{output_format}"

    with Image.open(entry.abs_path) as img:
        rgb_img = convert_to_rgb(img)
        # Pillow 9+ exposes resampling filters under Image.Resampling; older
        # versions provide them directly on Image. Handle both for compatibility.
        try:
            resample_filter = Image.Resampling.BILINEAR
        except AttributeError:
            # Older Pillow versions expose filters directly on Image. Use getattr
            # to avoid static analysis errors when the attribute may not exist.
            # Fallback to older-style constants if Resampling isn't available.
            # Use getattr for NEAREST too to avoid attribute errors in static analysis.
            resample_filter = getattr(Image, "BILINEAR", getattr(Image, "NEAREST", 0))
        resized = rgb_img.resize(tuple(image_size), resample=resample_filter)
        save_kwargs = {}
        if output_format.lower() in ("jpg", "jpeg"):
            save_kwargs["quality"] = jpeg_quality
        resized.save(out_path, **save_kwargs)

    return resized.width, resized.height


# --------------------------------------------------------------------------
# Stage: manifest
# --------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "rel_output_path",
    "class_name",
    "split",
    "original_width",
    "original_height",
    "processed_width",
    "processed_height",
    "source_rel_path",
]


def write_manifest(rows: List[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_preprocessing(config: dict, dataset_root: str | None = None) -> Path:
    """
    Run the full preprocessing pipeline end-to-end using the given
    config dict (as loaded from config.yaml). Returns the path to the
    written manifest.

    dataset_root, if given, overrides config["dataset"]["raw_root"]
    (used by tests to point at a synthetic tmp_path dataset instead of
    the real Vehicle-10 dataset).
    """
    ds_cfg = config["dataset"]
    pp_cfg = config["preprocessing"]
    manifest_cfg = config.get("manifest", {"filename": "manifest.csv"})

    root = dataset_root or ds_cfg["raw_root"]
    processed_root = Path(ds_cfg["processed_root"])
    image_size = tuple(pp_cfg["image_size"])
    min_image_size = pp_cfg["min_image_size"]
    test_fraction = pp_cfg["test_split_fraction"]
    seed = pp_cfg["seed"]
    output_format = pp_cfg.get("output_format", "jpg")
    jpeg_quality = pp_cfg.get("jpeg_quality", 95)

    # 1. Build raw index (metadata + on-disk intersection)
    index = build_dataset_index(root)

    # 2. Remove exact duplicates before any splitting
    index = remove_exact_duplicates(index)

    # 3. Drop tiny images
    index, orig_dims = filter_tiny_images(index, min_image_size)

    # 4. Deterministic split: preserve official valid, carve test from train
    splits = build_split_assignments(index, test_fraction, seed)

    # 5 + 6. Process each split incrementally, collect manifest rows
    manifest_rows: List[dict] = []
    for split_name, entries in splits.items():
        for entry in sorted(entries, key=lambda e: e.rel_path):
            out_w, out_h = process_and_save_image(
                entry, split_name, processed_root, image_size, output_format, jpeg_quality
            )
            orig_w, orig_h = orig_dims.get(entry.rel_path, (None, None))
            rel_out = f"{split_name}/{entry.class_name}/{Path(entry.rel_path).stem}.{output_format}"
            manifest_rows.append(
                {
                    "rel_output_path": rel_out,
                    "class_name": entry.class_name,
                    "split": split_name,
                    "original_width": orig_w,
                    "original_height": orig_h,
                    "processed_width": out_w,
                    "processed_height": out_h,
                    "source_rel_path": entry.rel_path,
                }
            )

    manifest_path = processed_root / manifest_cfg.get("filename", "manifest.csv")
    write_manifest(manifest_rows, manifest_path)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess the Vehicle-10 dataset.")
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        help="Path to config.yaml (default: src/config/config.yaml)",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Override the dataset root from config.yaml (optional).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = run_preprocessing(config, dataset_root=args.dataset_root)
    print(f"Preprocessing complete. Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()