#!/usr/bin/env python
"""Cross-model AUROC comparison with 5-fold CV error bars (descriptor-free).

Combines image-model CV (results/cv_clean) with triage fusion CV
(results/cv_triage) into one honest bar chart. Writes to results/paper_figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

img = json.loads((ROOT / "results/cv_clean/cv_summary.json").read_text())
fus = json.loads((ROOT / "results/cv_triage/cv_summary.json").read_text())

# (label, mean, std, group)
rows = []
namemap = {"cnn": "CNN", "resnet50": "ResNet-50", "custom_resnet50": "Custom ResNet-50",
           "vit_b16": "ViT-B/16", "hybrid": "Hybrid"}
for k, disp in namemap.items():
    if k in img:
        rows.append((disp, img[k]["auroc"]["mean"], img[k]["auroc"]["std"], "image"))
fmap = {"metadata_mlp": "Metadata (triage)", "late_fusion": "Late fusion (triage)",
        "gate_network": "Gate (triage)"}
for k, disp in fmap.items():
    if k in fus:
        g = "metadata" if "metadata" in k else "fusion"
        rows.append((disp, fus[k]["auroc"]["mean"], fus[k]["auroc"]["std"], g))

rows.sort(key=lambda r: r[1])
colors = {"image": "#4C72B0", "metadata": "#DD8452", "fusion": "#55A868"}
labels = [r[0] for r in rows]
means = [r[1] for r in rows]
stds = [r[2] for r in rows]
cols = [colors[r[3]] for r in rows]

plt.figure(figsize=(9, 6))
y = np.arange(len(rows))
plt.barh(y, means, xerr=stds, color=cols, capsize=4, height=0.65)
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(m + s + 0.005, i, f"{m:.3f}", va="center", fontsize=9)
plt.axvline(0.5, ls="--", color="gray", lw=1)
plt.yticks(y, labels)
plt.xlabel("AUROC (5-fold CV mean ± std)")
plt.xlim(0.45, 1.0)
plt.title("Cross-model AUROC — 5-fold CV (descriptor-free)\n"
          "blue = image · orange = metadata · green = fusion (triage)")
plt.tight_layout()
plt.savefig(OUT / "fig2b_auroc_comparison_cv.png", dpi=150)
plt.close()
print("AUROC (mean±std):", {r[0]: (round(r[1], 3), round(r[2], 3)) for r in rows})
print(f"wrote {OUT}/fig2b_auroc_comparison_cv.png")
