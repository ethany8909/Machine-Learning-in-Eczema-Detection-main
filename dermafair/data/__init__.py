"""Dataset, stratified splitting, and the leakage audit.

The leakage audit is a first-class deliverable: it writes LEAKAGE_AUDIT.md, which
becomes Methods-section text and pre-empts the most common reviewer concern for
small-N medical-imaging papers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Stratified, patient-aware splitting
# --------------------------------------------------------------------------- #
def make_splits(
    df: pd.DataFrame,
    label_col: str,
    stratify_cols: list[str],
    patient_col: str | None = None,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return dict of split -> row-index arrays.

    Joint stratification on ``stratify_cols`` (e.g. [label, fitzpatrick]). If
    ``patient_col`` is given, splitting is done at the patient level so no
    patient appears in more than one split (prevents the most insidious leak).
    """
    assert abs(train + val + test - 1.0) < 1e-6, "splits must sum to 1"

    strat_key = df[stratify_cols].astype(str).agg("_".join, axis=1)

    if patient_col and patient_col in df.columns:
        # split unique patients, carrying their (majority) strat key
        patients = df.groupby(patient_col)[stratify_cols[0]].first().index.to_numpy()
        pat_strat = (
            df.groupby(patient_col)[stratify_cols]
            .first()
            .astype(str)
            .agg("_".join, axis=1)
            .to_numpy()
        )
        p_train, p_tmp, s_train, s_tmp = train_test_split(
            patients, pat_strat, train_size=train, stratify=_safe_strat(pat_strat), random_state=seed
        )
        rel = val / (val + test)
        p_val, p_test = train_test_split(
            p_tmp, train_size=rel, stratify=_safe_strat(s_tmp), random_state=seed
        )
        idx = {
            "train": df.index[df[patient_col].isin(p_train)].to_numpy(),
            "val": df.index[df[patient_col].isin(p_val)].to_numpy(),
            "test": df.index[df[patient_col].isin(p_test)].to_numpy(),
        }
        return idx

    # image-level fallback
    all_idx = df.index.to_numpy()
    i_train, i_tmp = train_test_split(
        all_idx, train_size=train, stratify=_safe_strat(strat_key.to_numpy()), random_state=seed
    )
    rel = val / (val + test)
    tmp_strat = strat_key.loc[i_tmp].to_numpy()
    i_val, i_test = train_test_split(
        i_tmp, train_size=rel, stratify=_safe_strat(tmp_strat), random_state=seed
    )
    return {"train": i_train, "val": i_val, "test": i_test}


def _safe_strat(arr: np.ndarray):
    """Disable stratification if any class has <2 members (sklearn requirement)."""
    _, counts = np.unique(arr, return_counts=True)
    return arr if counts.min() >= 2 else None


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
def leakage_audit(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    image_col: str,
    patient_col: str | None,
    image_root: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    """Run duplicate + split-overlap checks; write LEAKAGE_AUDIT.md.

    Checks:
      1. Patient overlap across splits (if patient_col available).
      2. Exact path duplicates across splits.
      3. Perceptual near-duplicates across splits (if images + imagehash present).
    """
    findings: dict = {}

    # 1. patient overlap
    if patient_col and patient_col in df.columns:
        sets = {k: set(df.loc[idx, patient_col]) for k, idx in splits.items()}
        overlap = (
            (sets["train"] & sets["val"])
            | (sets["train"] & sets["test"])
            | (sets["val"] & sets["test"])
        )
        findings["patient_overlap_count"] = len(overlap)
    else:
        findings["patient_overlap_count"] = "n/a (no patient_col)"

    # 2. exact path duplicates across splits
    path_sets = {k: set(df.loc[idx, image_col]) for k, idx in splits.items()}
    exact = (
        (path_sets["train"] & path_sets["val"])
        | (path_sets["train"] & path_sets["test"])
        | (path_sets["val"] & path_sets["test"])
    )
    findings["exact_path_overlap_count"] = len(exact)

    # 3. perceptual near-duplicates
    findings["perceptual_near_duplicates"] = _perceptual_check(
        df, splits, image_col, image_root
    )

    if out_path is not None:
        _write_audit_md(findings, out_path)
    return findings


def _perceptual_check(df, splits, image_col, image_root):
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return "skipped (imagehash/PIL not installed)"
    if image_root is None:
        return "skipped (no image_root provided)"

    def hash_split(idx):
        hashes = {}
        for i in idx:
            p = Path(image_root) / df.loc[i, image_col]
            if p.exists():
                try:
                    hashes[i] = imagehash.phash(Image.open(p).convert("RGB"))
                except Exception:
                    continue
        return hashes

    h_train = hash_split(splits["train"])
    near = 0
    for other in ("val", "test"):
        h_other = hash_split(splits[other])
        for ho in h_other.values():
            for ht in h_train.values():
                if ho - ht <= 5:  # Hamming distance threshold
                    near += 1
                    break
    return near


def _write_audit_md(findings: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Leakage Audit",
        "",
        "Automated integrity checks run before model training. A clean audit is a "
        "precondition for trusting downstream accuracy and fairness results.",
        "",
        f"- **Patient overlap across splits:** {findings['patient_overlap_count']}",
        f"- **Exact image-path overlap across splits:** {findings['exact_path_overlap_count']}",
        f"- **Perceptual near-duplicates (train vs. val/test, Hamming ≤5):** "
        f"{findings['perceptual_near_duplicates']}",
        "",
        "All counts should be **0** (or 'n/a'). Any non-zero value must be resolved "
        "by removing or reassigning the offending samples before proceeding.",
    ]
    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Composition table (Table S1)
# --------------------------------------------------------------------------- #
def composition_table(df: pd.DataFrame, label_col: str, sensitive_attr: str) -> pd.DataFrame:
    """Class × sensitive-attribute contingency table (becomes Table S1)."""
    return pd.crosstab(df[sensitive_attr], df[label_col], margins=True, margins_name="Total")
