#!/usr/bin/env python
"""Produce the relative-fairness table (paired bootstrap) for all models.

Answers "how often is each model fairer than the reference?" instead of chasing
significance. Writes results/week4_fairness/TABLE_relative_fairness.md.

NOTE: run on whatever predictions you point it at. On the current (descriptor-
contaminated) predictions it is illustrative; rerun on the descriptor-free
triage results once those exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dermafair.fairness.relative import relative_report

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "results/image_models_clean/predictions"
FUS = ROOT / "results/fusion_clean/predictions"

MODELS = {
    "CNN": IMG / "cnn.npz",
    "ResNet-50": IMG / "resnet50.npz",
    "Custom ResNet-50": IMG / "custom_resnet50.npz",
    "ViT-B/16": IMG / "vit_b16.npz",
    "Hybrid": IMG / "hybrid.npz",
    "Metadata-MLP": FUS / "metadata_mlp.npz",
    "Late fusion": FUS / "late_fusion.npz",
    "Gate network": FUS / "gate_network.npz",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="Late fusion")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/week4_fairness/TABLE_relative_fairness.md")
    args = ap.parse_args()

    # load aligned predictions, filter missing skin tone once (same mask for all)
    base = np.load(MODELS["ResNet-50"])
    keep = base["fitzpatrick"] != -1
    models = {}
    for name, f in MODELS.items():
        d = np.load(f)
        models[name] = (d["y_true"][keep], d["y_pred"][keep], d["fitzpatrick"][keep])

    rep = relative_report(models, reference=args.reference, n_boot=args.n_boot)

    ref_gap = rep["_reference_gap_estimate"]["accuracy"]
    lines = [
        f"# Relative fairness (paired bootstrap, n={args.n_boot}) — reference: **{args.reference}**\n",
        "Estimation + relative claims instead of significance (small-N honest framing). "
        "P(fairer) = fraction of paired resamples where the model's gap < the reference's gap.\n",
        f"Reference ({args.reference}) accuracy gap: "
        f"{ref_gap['median']:.3f} [{ref_gap['ci_low']:.3f}, {ref_gap['ci_high']:.3f}]\n",
        "| Model | Acc gap [95% CI] | P(fairer: acc) | P(fairer: TPR) | P(fairer: FPR) | Consistency |",
        "|---|---|---|---|---|---|",
    ]
    # order by consistency (fairest-vs-reference first)
    ranked = sorted((k for k in rep if not k.startswith("_")),
                    key=lambda k: rep[k]["consistency"], reverse=True)
    for name in ranked:
        r = rep[name]
        ge = r["gap_estimate"]["accuracy"]
        pf = r["prob_fairer"]
        lines.append(
            f"| {name} | {ge['median']:.3f} [{ge['ci_low']:.3f}, {ge['ci_high']:.3f}] | "
            f"{pf['accuracy']:.2f} | {pf['tpr']:.2f} | {pf['fpr']:.2f} | {r['consistency']:.2f} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
