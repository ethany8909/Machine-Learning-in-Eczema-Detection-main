"""Shared data loading, label definitions, and evaluation for the improvement experiments.

Every experiment in experiments/improvements/ uses:
  * the same subject-level, class x Fitzpatrick-stratified 5 folds (manifest_clean.csv),
  * the same external test set (SCIN, 1,128 cases), predicted by the ensemble of the 5 fold models,
  * the same skin-tone breakdown (Fitzpatrick I-IV vs V-VI) with bootstrap intervals.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dermafair.data.labels import harmonized_label  # noqa: F401  (re-exported)
from dermafair.paths import DATA_DIR, RESULTS_DIR

ROOT = DATA_DIR  # cohort data (DERMAFAIR_DATA_DIR)
IMP = RESULTS_DIR / "improvements"
FEAT = IMP / "features"
CACHE = IMP / "cache"
RESULTS = IMP / "results"
for d in (FEAT, CACHE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

K = 5
AGES = ["under_20", "20_39", "40_59", "60_plus"]
SEXES = ["male", "female"]
SITES = ["head_neck", "arm", "hand", "torso_front", "torso_back", "genital_groin", "buttocks", "leg", "foot", "other"]


# --------------------------------------------------------------------------- #
# Data (labels: dermafair.data.labels.harmonized_label, the SCIN keyword rule)
# --------------------------------------------------------------------------- #
@dataclass
class Cohorts:
    internal: pd.DataFrame
    external: pd.DataFrame


def load_cohorts() -> Cohorts:
    man = pd.read_csv(ROOT / "manifest_clean.csv")
    canon = pd.read_csv(ROOT / "internal_canonical_manifest.csv")[["image_name", "age", "sex", "sites"]]
    raw = pd.read_csv(ROOT / "Skin_Metadata-1.csv")[["Image_name", "Disease_label"]]
    raw = raw.rename(columns={"Image_name": "image_name", "Disease_label": "disease"})
    df = man.merge(canon, on="image_name", how="left").merge(raw, on="image_name", how="left")
    files = {}
    for d in ("DATASET_0", "DATASET_1"):
        for f in (ROOT / d).iterdir():
            if f.is_file():
                files[f.name] = f
    df["path"] = df["image_name"].map(lambda n: str(files[n]))
    df["harmonized"] = df["disease"].map(harmonized_label)
    df["sites"] = df["sites"].fillna("")
    df = df.sort_values("image_name").reset_index(drop=True)

    ext = pd.read_csv(ROOT / "external_scin_manifest.csv", dtype={"case_id": str})
    idx = {}
    for f in (ROOT / "scin_images").iterdir():
        if f.is_file():
            idx.setdefault(f.name.split("__")[0], f)
    ext["path"] = ext["case_id"].map(lambda c: str(idx[c]) if c in idx else None)
    ext = ext[ext["path"].notna()].reset_index(drop=True)
    ext["sites"] = ext["sites"].fillna("")
    return Cohorts(df, ext)


def meta_matrix(df: pd.DataFrame, unknown_slots: bool = True) -> np.ndarray:
    """One-hot age / sex / body sites. With unknown_slots, missing age or sex gets its
    own indicator column (the original design); otherwise it is left all-zero so an
    imputer can fill it."""
    ages = AGES + (["unknown"] if unknown_slots else [])
    sexes = SEXES + (["unknown"] if unknown_slots else [])
    X = np.zeros((len(df), len(ages) + len(sexes) + len(SITES) + 1), dtype=np.float32)
    for i, r in enumerate(df.itertuples()):
        a = r.age if r.age in AGES else "unknown"
        s = r.sex if r.sex in SEXES else "unknown"
        if a in ages:
            X[i, ages.index(a)] = 1
        if s in sexes:
            X[i, len(ages) + sexes.index(s)] = 1
        sites = [x for x in str(r.sites).split("|") if x in SITES]
        for x in sites:
            X[i, len(ages) + len(sexes) + SITES.index(x)] = 1
        X[i, -1] = 0 if sites else 1  # "no site recorded" indicator
    return X


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tone_report(y, p, fst, n=2000, seed=0) -> dict:
    """External AUROC for Fitzpatrick I-IV vs V-VI, and the gap with a bootstrap CI."""
    y, p, fst = map(np.asarray, (y, p, fst))
    light, dark = (fst >= 1) & (fst <= 4), fst >= 5
    rng = np.random.default_rng(seed)
    il, idk = np.where(light)[0], np.where(dark)[0]
    gaps = []
    for _ in range(n):
        a, b = rng.choice(il, len(il)), rng.choice(idk, len(idk))
        if len(np.unique(y[a])) > 1 and len(np.unique(y[b])) > 1:
            gaps.append(roc_auc_score(y[a], p[a]) - roc_auc_score(y[b], p[b]))
    return {
        "light": auc(y[light], p[light]),
        "dark": auc(y[dark], p[dark]),
        "gap": auc(y[light], p[light]) - auc(y[dark], p[dark]),
        "gap_ci": (float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))),
        "dark_ci": boot_ci(y[dark], p[dark], n, seed),
    }


def paired_delta(y, p_new, p_base, n=2000, seed=0):
    """AUROC(new) - AUROC(base) on the same cases, with a paired bootstrap CI."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, p_new, p_base))
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            ds.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return auc(y, a) - auc(y, b), (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))


def write_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
