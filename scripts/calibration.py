#!/usr/bin/env python
"""Figure S1 — reliability diagrams (calibration) for the primary triage models.

Bins predictions by predicted positive-class probability and plots observed vs
predicted, with ECE annotated. Well-calibrated = points on the diagonal.
Writes directly into results/paper_figures/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dermafair.fairness.advanced_metrics import expected_calibration_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Image (ResNet-50)": ROOT / "results/image_models_clean/predictions/resnet50.npz",
    "Metadata-only (triage)": ROOT / "results/spectrum/triage/predictions/metadata_mlp.npz",
    "Late fusion (triage)": ROOT / "results/spectrum/triage/predictions/late_fusion.npz",
    "Gate network (triage)": ROOT / "results/spectrum/triage/predictions/gate_network.npz",
}
N_BINS = 10


def reliability(ax, y_true, p_pos, name):
    bins = np.linspace(0, 1, N_BINS + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_pos > lo) & (p_pos <= hi) if hi < 1 else (p_pos > lo) & (p_pos <= hi + 1e-9)
        if m.sum() == 0:
            continue
        xs.append(p_pos[m].mean()); ys.append(y_true[m].mean()); ns.append(int(m.sum()))
    ece = expected_calibration_error(y_true, p_pos, n_bins=N_BINS)["ece"]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="#4C72B0", lw=2)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), fontsize=7, xytext=(2, 3), textcoords="offset points", color="#555")
    ax.set_title(f"{name}\nECE = {ece:.3f}", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted P(psoriasis)"); ax.set_ylabel("observed fraction psoriasis")
    ax.grid(alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, f) in zip(axes.ravel(), MODELS.items()):
        d = np.load(f)
        reliability(ax, d["y_true"].astype(float), d["y_prob"][:, 1], name)
    plt.suptitle("Figure S1 — Calibration (reliability diagrams), triage regime\n"
                 "bin counts annotated; points below diagonal = over-confident", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "figS1_calibration.png", dpi=150)
    plt.close()
    print(f"wrote {OUT}/figS1_calibration.png")


if __name__ == "__main__":
    main()
