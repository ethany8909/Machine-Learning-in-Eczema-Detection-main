"""Metadata-only EXTERNAL validation: train internally, test on SCIN.

Runs entirely on the harmonized canonical metadata space, so no images are needed.
Reports three numbers so the drop can be attributed correctly:

  1. internal CV (canonical space)  - subject-level 5-fold; the honest internal baseline
  2. external (SCIN, all)           - trained on ALL internal data, tested on SCIN
  3. external by tone group         - FST I-IV vs V-VI (binary split, per Groh et al.)

AUROC is the primary metric because SCIN's class prevalence (9:1) differs sharply
from ours (1.8:1) and AUROC is prevalence-independent.
"""

from __future__ import annotations

import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from dermafair.paths import DATA_DIR, output_dir

REPORT_DIR = output_dir("external")
AGES = ["under_20", "20_39", "40_59", "60_plus", "unknown"]
SEXES = ["male", "female", "unknown"]
SITES = ["head_neck", "arm", "hand", "torso_front", "torso_back", "genital_groin", "buttocks", "leg", "foot", "other"]
FEATS = [f"age={a}" for a in AGES] + [f"sex={s}" for s in SEXES] + [f"site={s}" for s in SITES]


def vec(row):
    v = [0.0] * len(FEATS)
    a = row["age"] if row["age"] in AGES else "unknown"
    v[AGES.index(a)] = 1.0
    s = row["sex"] if row["sex"] in SEXES else "unknown"
    v[len(AGES) + SEXES.index(s)] = 1.0
    for site in row["sites"].split("|") if row["sites"] else []:
        if site in SITES:
            v[len(AGES) + len(SEXES) + SITES.index(site)] = 1.0
    return v


def load(path):
    return list(csv.DictReader(open(DATA_DIR / path, encoding="utf-8")))


def boot_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    internal = load("internal_canonical_manifest.csv")
    external = load("external_scin_manifest.csv")

    Xi = np.array([vec(r) for r in internal])
    yi = np.array([int(r["label"]) for r in internal])
    gi = np.array([r["subject"] for r in internal])
    Xe = np.array([vec(r) for r in external])
    ye = np.array([int(r["label"]) for r in external])
    fe = np.array([int(r["fitzpatrick"]) for r in external])
    we = np.array([float(r["label_weight"]) for r in external])

    L = [
        "# Metadata-only external validation (canonical space)\n",
        f"Features ({len(FEATS)}): coarse age band, sex, 10 coarse body sites.\n",
        f"- internal: n={len(yi)}, psoriasis prevalence {yi.mean():.1%}",
        f"- SCIN:     n={len(ye)}, psoriasis prevalence {ye.mean():.1%}",
        "\n> Prevalence differs sharply, so AUROC (prevalence-independent) is the primary metric.\n",
    ]

    # ---- 1. internal subject-level CV in the canonical space ----
    aucs, baccs = [], []
    for tr, te in GroupKFold(n_splits=5).split(Xi, yi, groups=gi):
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xi[tr], yi[tr])
        p = m.predict_proba(Xi[te])[:, 1]
        aucs.append(roc_auc_score(yi[te], p))
        baccs.append(balanced_accuracy_score(yi[te], (p >= 0.5).astype(int)))
    L += [
        "## 1. Internal 5-fold CV (subject-level, canonical features)\n",
        f"- AUROC **{np.mean(aucs):.3f}** (SD {np.std(aucs):.3f})",
        f"- balanced accuracy {np.mean(baccs):.3f} (SD {np.std(baccs):.3f})\n",
    ]

    # ---- 2. external ----
    model = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xi, yi)
    pe = model.predict_proba(Xe)[:, 1]
    auc_e = roc_auc_score(ye, pe)
    lo, hi = boot_ci(ye, pe)
    L += [
        "## 2. External validation on SCIN (trained on all internal data)\n",
        f"- AUROC **{auc_e:.3f}** (95% CI {lo:.3f}-{hi:.3f})",
        f"- balanced accuracy {balanced_accuracy_score(ye, (pe >= 0.5).astype(int)):.3f}",
        f"- change vs internal CV: **{auc_e - np.mean(aucs):+.3f}**\n",
    ]

    # ---- 3. tone-stratified (binary split, per Groh et al.) ----
    L += [
        "## 3. External, by skin-tone group (FST I-IV vs V-VI)\n",
        "| Group | n | psoriasis | AUROC | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for name, mask in [("Light (FST I-IV)", (fe >= 1) & (fe <= 4)), ("Dark (FST V-VI)", (fe >= 5))]:
        if mask.sum() and len(np.unique(ye[mask])) > 1:
            a = roc_auc_score(ye[mask], pe[mask])
            lo, hi = boot_ci(ye[mask], pe[mask])
            L.append(f"| {name} | {int(mask.sum())} | {int(ye[mask].sum())} | {a:.3f} | {lo:.3f}-{hi:.3f} |")
        else:
            L.append(f"| {name} | {int(mask.sum())} | {int(ye[mask].sum())} | n/a | n/a |")

    # ---- 4. sensitivity: high-confidence SCIN labels only ----
    L += ["\n## 4. Sensitivity: high-confidence SCIN labels (consensus weight >= 0.5)\n"]
    m = we >= 0.5
    if m.sum() and len(np.unique(ye[m])) > 1:
        a = roc_auc_score(ye[m], pe[m])
        lo, hi = boot_ci(ye[m], pe[m])
        L.append(f"- n={int(m.sum())} (psoriasis {int(ye[m].sum())}): AUROC **{a:.3f}** (95% CI {lo:.3f}-{hi:.3f})")

    L += [
        "\n## Interpretation\n",
        "The internal CV figure is the like-for-like baseline: it uses the same coarse features, so the",
        "difference between (1) and (2) isolates cohort shift rather than feature coarsening. SCIN uses",
        "dermatologist differential labels rather than biopsy confirmation, is adult-only, and consists of",
        "consumer smartphone photographs, so a decline is expected.",
    ]

    out = REPORT_DIR / "EXTERNAL_VALIDATION_METADATA.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
