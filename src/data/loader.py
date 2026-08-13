"""
src/data/loader.py

Thin index-building layer on top of validator.py.

This module does NOT decode or touch image pixels — it only builds an
in-memory index of (absolute_path, class_name, original_split) tuples
by combining:

  - validator.discover_class_folders()  -> which files exist on disk
  - validator.load_metadata()           -> the official train/valid
                                            split from train_meta.json /
                                            valid_meta.json

The resulting index is what preprocess.py consumes to do the actual
work (dedup, filtering, resizing, splitting, manifest generation).

No image decoding happens here, so building the index is cheap even
for tens of thousands of files.
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.data.validator import (
    discover_class_folders,
    load_metadata,
    LABEL_TO_CLASS,
)


@dataclass(frozen=True)
class DatasetEntry:
    """One image's identity within the raw dataset."""
    abs_path: str          # absolute path on disk
    rel_path: str          # path relative to dataset_root, forward-slash normalized
    class_name: str        # e.g. "car"
    original_split: str    # "train" or "valid" (from official metadata)


def load_config(config_path: str) -> dict:
    """Load and return the YAML config as a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _normalize_rel_path(rel_path: str) -> str:
    return rel_path.replace("\\", "/")


def build_dataset_index(dataset_root: str) -> List[DatasetEntry]:
    """
    Build the full dataset index by combining validator.py's folder
    discovery with the official train/valid metadata files.

    Only files that:
      (a) physically exist under a class folder in dataset_root, AND
      (b) are referenced by train_meta.json or valid_meta.json

    are included. This naturally excludes stray files not covered by
    the official metadata, and skips metadata entries whose file is
    missing on disk (validator.py's check_metadata_paths_exist can be
    used separately to audit that case).

    Returns:
        List[DatasetEntry], one per valid (on-disk AND in-metadata) image.
    """
    dataset_root = str(dataset_root)
    train_meta_path = os.path.join(dataset_root, "train_meta.json")
    valid_meta_path = os.path.join(dataset_root, "valid_meta.json")

    # (a) what's actually on disk, keyed by class folder
    class_files = discover_class_folders(dataset_root)
    on_disk_rel_paths = set()
    for class_name, files in class_files.items():
        for f in files:
            rel = os.path.relpath(f, dataset_root)
            on_disk_rel_paths.add(_normalize_rel_path(rel))

    # (b) what the official metadata says exists, per split
    train_entries = load_metadata(train_meta_path)
    valid_entries = load_metadata(valid_meta_path)

    index: List[DatasetEntry] = []

    for rel_path, label in train_entries:
        norm = _normalize_rel_path(rel_path)
        if norm not in on_disk_rel_paths:
            continue  # metadata references a file that isn't on disk; skip
        class_name = LABEL_TO_CLASS.get(label)
        if class_name is None:
            continue
        index.append(
            DatasetEntry(
                abs_path=os.path.join(dataset_root, rel_path),
                rel_path=norm,
                class_name=class_name,
                original_split="train",
            )
        )

    for rel_path, label in valid_entries:
        norm = _normalize_rel_path(rel_path)
        if norm not in on_disk_rel_paths:
            continue
        class_name = LABEL_TO_CLASS.get(label)
        if class_name is None:
            continue
        index.append(
            DatasetEntry(
                abs_path=os.path.join(dataset_root, rel_path),
                rel_path=norm,
                class_name=class_name,
                original_split="valid",
            )
        )

    return index


def index_by_class(index: List[DatasetEntry]) -> Dict[str, List[DatasetEntry]]:
    """Group an index into {class_name: [DatasetEntry, ...]}."""
    grouped: Dict[str, List[DatasetEntry]] = {}
    for entry in index:
        grouped.setdefault(entry.class_name, []).append(entry)
    return grouped


def index_by_split(index: List[DatasetEntry]) -> Dict[str, List[DatasetEntry]]:
    """Group an index into {"train": [...], "valid": [...]}."""
    grouped: Dict[str, List[DatasetEntry]] = {"train": [], "valid": []}
    for entry in index:
        grouped[entry.original_split].append(entry)
    return grouped