#!/usr/bin/env python
"""Spectrum figures — accuracy and fairness across the 3 deployment regimes.

  spectrum_accuracy.png       - AUROC of {image, metadata, late, gate} across
                                autonomous -> triage -> expert (value-of-information).
  spectrum_fairness.png       - the CENTERPIECE: skin-tone accuracy gap (+ bootstrap CI)
                                across regimes, per model. Does adding clinical context
                                make diagnosis more/less equitable?

Image is regime-independent (flat reference line). Expert regime is the leaky rung.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from dermafair.fairness import bootstrap_gap_ci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/week5_spectrum"
OUT.mkdir(parents=True, exist_ok=True)
REGIMES = ["autonomous", "triage", "expert"]
IMG_REF = ROOT / "results/image_models_clean/predictions/hybrid.npz"   # regime-independent
COLORS = {"Image (best)": "#4C72B0", "Metadata": "#DD8452",
          "Late fusion": "#55A868", "Gate": "#C44E52"}


def _load(p):
    d = np.load(p)
    keep = d["fitzpatrick"] != -1
    return d["y_true"][keep], d["y_pred"][keep], d["y_prob"][keep][:, 1], d["fitzpatrick"][keep]


def _auroc(yt, p):
    return float(roc_auc_score(yt, p)) if len(np.unique(yt)) > 1 else np.nan


def _acc_gap(yt, yp, g):
    b = bootstrap_gap_ci(yt, yp, g, "accuracy", n_bootstrap=1000)
    return b["point"], b["ci_low"], b["ci_high"]


def collect():
    # image reference (constant across regimes)
    iyt, iyp, ipp, ig = _load(IMG_REF)
    img_auroc = _auroc(iyt, ipp)
    img_gap = _acc_gap(iyt, iyp, ig)
    data = {"auroc": {}, "gap": {}}
    for model, key in [("Metadata", "metadata_mlp"), ("Late fusion", "late_fusion"), ("Gate", "gate_network")]:
        data["auroc"][model] = []
        data["gap"][model] = []
        for r in REGIMES:
            yt, yp, pp, g = _load(ROOT / f"results/spectrum/{r}/predictions/{key}.npz")
            data["auroc"][model].append(_auroc(yt, pp))
            data["gap"][model].append(_acc_gap(yt, yp, g))
    return data, img_auroc, img_gap


def fig_accuracy(data, img_auroc):
    plt.figure(figsize=(8, 5.5))
    x = range(len(REGIMES))
    plt.axhline(img_auroc, ls="--", color=COLORS["Image (best)"], lw=2,
                label=f"Image (best, regime-independent) = {img_auroc:.2f}")
    for model in ("Metadata", "Late fusion", "Gate"):
        plt.plot(x, data["auroc"][model], "o-", color=COLORS[model], lw=2, label=model)
    plt.xticks(list(x), [r + ("\n(primary)" if r == "triage" else "") for r in REGIMES])
    plt.ylabel("AUROC"); plt.ylim(0.5, 1.0)
    plt.title("Spectrum — accuracy as clinical context increases\n(expert rung = descriptor leakage)")
    plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "spectrum_accuracy.png", dpi=150); plt.close()
    print("wrote spectrum_accuracy.png")


def fig_fairness(data, img_gap):
    plt.figure(figsize=(8, 5.5))
    x = np.arange(len(REGIMES))
    plt.axhline(img_gap[0], ls="--", color=COLORS["Image (best)"], lw=2,
                label=f"Image (best) gap = {img_gap[0]:.2f}")
    for i, model in enumerate(("Metadata", "Late fusion", "Gate")):
        pts = data["gap"][model]
        y = [p[0] for p in pts]
        lo = [p[0] - p[1] for p in pts]; hi = [p[2] - p[0] for p in pts]
        off = (i - 1) * 0.05
        plt.errorbar(x + off, y, yerr=[lo, hi], fmt="o-", color=COLORS[model], lw=2,
                     capsize=4, label=model)
    plt.xticks(x, [r + ("\n(primary)" if r == "triage" else "") for r in REGIMES])
    plt.ylabel("skin-tone accuracy gap  (lower = fairer)")
    plt.title("Spectrum — fairness vs clinical context (CENTERPIECE)\n"
              "does adding metadata narrow the skin-tone gap? (95% bootstrap CI)")
    plt.legend(fontsize=9); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "spectrum_fairness.png", dpi=150); plt.close()
    print("wrote spectrum_fairness.png")


if __name__ == "__main__":
    data, img_auroc, img_gap = collect()
    fig_accuracy(data, img_auroc)
    fig_fairness(data, img_gap)
    print("\nAUROC by regime:", {m: [round(v, 3) for v in data["auroc"][m]] for m in data["auroc"]})
    print("Acc-gap by regime:", {m: [round(p[0], 3) for p in data["gap"][m]] for m in data["gap"]})
    print(f"Image ref: auroc={img_auroc:.3f} gap={img_gap[0]:.3f}\nDeliverables -> {OUT}")
