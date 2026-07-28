#!/usr/bin/env python
"""Dummy / trivial baselines — the sanity floor (Week-1 deliverable).

  - majority class     : always predict the most frequent train class
  - stratified random  : predict by class prior
  - color-histogram LR  : logistic regression on per-channel RGB histograms
                          (tests whether the task is trivially solvable from color)

Reports overall + per-Fitzpatrick accuracy on the clean test split, so every real
model can be judged against this floor. If a deep model barely beats these, that is
a red flag.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "manifest_clean.csv"
SOURCE_DIRS = [ROOT.parent / "DATASET_0", ROOT.parent / "DATASET_1"]
OUT = ROOT / "results/baselines"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = [3, 4, 5, 6]
BINS = 16


def color_hist(path):
    im = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=np.float32)
    feats = []
    for c in range(3):
        h, _ = np.histogram(im[:, :, c], bins=BINS, range=(0, 255), density=True)
        feats.append(h)
    return np.concatenate(feats)


def per_tone_acc(y_true, y_pred, fitz):
    out = {}
    for b in BANDS:
        m = fitz == b
        out[b] = float(np.mean(y_pred[m] == y_true[m])) if m.any() else float("nan")
    return out


def main():
    idx = {}
    for d in SOURCE_DIRS:
        for f in d.iterdir():
            if f.is_file():
                idx[f.name] = f

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    ytr = np.array([int(r["label"]) for r in tr])
    yte = np.array([int(r["label"]) for r in te])
    fte = np.array([int(r["fitzpatrick"]) for r in te])

    results = {}

    # majority class
    maj = Counter(ytr).most_common(1)[0][0]
    yp = np.full_like(yte, maj)
    results["majority_class"] = (float(np.mean(yp == yte)), per_tone_acc(yte, yp, fte), float("nan"))

    # stratified random (expected accuracy = sum p_c^2), one sampled realisation for per-tone
    rng = np.random.default_rng(42)
    p1 = ytr.mean()
    yr = (rng.random(len(yte)) < p1).astype(int)
    results["stratified_random"] = (float(np.mean(yr == yte)), per_tone_acc(yte, yr, fte),
                                    float(roc_auc_score(yte, np.full(len(yte), p1))) if len(set(yte)) > 1 else float("nan"))

    # color-histogram logistic regression
    Xtr = np.array([color_hist(idx[r["image_name"]]) for r in tr if r["image_name"] in idx])
    Xte = np.array([color_hist(idx[r["image_name"]]) for r in te if r["image_name"] in idx])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    yp = (prob >= 0.5).astype(int)
    results["color_histogram_LR"] = (float(np.mean(yp == yte)), per_tone_acc(yte, yp, fte),
                                     float(roc_auc_score(yte, prob)))

    lines = ["# Dummy baselines — sanity floor (clean test split)\n",
             "Real models must clear these. Color-hist AUROC near chance => task is NOT "
             "trivially color-separable (good). \n",
             "| Baseline | Overall acc | AUROC | FST3 | FST4 | FST5 | FST6 |",
             "|---|---|---|---|---|---|---|"]
    for name, (acc, tone, auc) in results.items():
        aucs = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
        cells = " | ".join(f"{tone[b]:.2f}" if not np.isnan(tone[b]) else "n/a" for b in BANDS)
        lines.append(f"| {name} | {acc:.3f} | {aucs} | {cells} |")
    (OUT / "BASELINES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT}/BASELINES.md")


if __name__ == "__main__":
    main()
