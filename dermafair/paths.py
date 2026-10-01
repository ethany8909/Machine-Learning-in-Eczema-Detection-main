"""Filesystem layout for the study.

Data and outputs never live in version control. Two environment variables point the
code at them, so the same scripts run on any machine:

    DERMAFAIR_DATA_DIR     raw cohorts and the manifests/splits derived from them
                           (default: <repo>/data)
    DERMAFAIR_RESULTS_DIR  checkpoints, predictions, reports and figures
                           (default: <repo>/results)

Expected contents of DERMAFAIR_DATA_DIR are documented in docs/DATA.md.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DERMAFAIR_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
RESULTS_DIR = Path(os.environ.get("DERMAFAIR_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()

# --- internal development cohort (DermaCon-IN) ---------------------------------------
DERMACON_METADATA = DATA_DIR / "Skin_Metadata-1.csv"
DERMACON_IMAGE_DIRS = (DATA_DIR / "DATASET_0", DATA_DIR / "DATASET_1")
SPLIT_DIR = DATA_DIR / "dataset_split"  # single stratified split (scripts/data/split_dataset.py)
SPLIT_CLEAN_DIR = DATA_DIR / "dataset_split_clean"  # after de-duplication (scripts/data/clean_dataset.py)
MANIFEST_CLEAN = DATA_DIR / "manifest_clean.csv"  # one row per image, with CV fold
INTERNAL_CANONICAL = DATA_DIR / "internal_canonical_manifest.csv"

# --- external cohorts ------------------------------------------------------------------
SCIN_METADATA_DIR = DATA_DIR / "scin_metadata"  # scin_cases.csv, scin_labels.csv
SCIN_IMAGES = DATA_DIR / "scin_images"
SCIN_MANIFEST = DATA_DIR / "external_scin_manifest.csv"
FITZ17K_DIR = DATA_DIR / "fitzpatrick17k"


def output_dir(*parts: str) -> Path:
    """A results sub-directory, created on first use."""
    path = RESULTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
