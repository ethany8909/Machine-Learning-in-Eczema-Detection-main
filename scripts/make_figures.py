#!/usr/bin/env python
"""Generate publication figures from saved predictions (.npz).

Outputs to results/figures/:
  roc_image_models.png     - ROC overlay for the 5 image backbones (like the classic comparison plot)
  roc_modality.png         - ROC: best image vs metadata-only vs fusion (the modality story)
  confusion_matrices.png   - 2x4 grid of confusion matrices, all models
  auroc_comparison.png     - AUROC bar chart grouped by model family
  fairness_by_tone.png     - accuracy per Fitzpatrick band per model (heatmap)
  gate_weights.png         - gate image/metadata weight per epoch
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]

def _dir(env, default):
    v = os.environ.get(env)
    return Path(v).resolve() if v else default

# Dirs are overridable via env vars so the overnight runner can target the
# clean-data results (results/*_clean/) without editing this file.
IMG = _dir("DERMAFAIR_IMG_DIR", ROOT / "results/image_models/predictions")
FUS = _dir("DERMAFAIR_FUS_DIR", ROOT / "results/fusion_fixed/predictions")   # fixed fusion
OUT = _dir("DERMAFAIR_FIG_OUT", ROOT / "results/figures")
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_MODELS = {
    "CNN (from scratch)": IMG / "cnn.npz",
    "ResNet-50": IMG / "resnet50.npz",
    "Custom ResNet-50": IMG / "custom_resnet50.npz",
    "ViT-B/16": IMG / "vit_b16.npz",
    "Hybrid CNN-Transformer": IMG / "hybrid.npz",
}
ALL_MODELS = {
    **IMAGE_MODELS,
    "Metadata-only (MLP)": FUS / "metadata_mlp.npz",
    "Late fusion": FUS / "late_fusion.npz",
    "Gate network": FUS / "gate_network.npz",
}


def _load(f):
    d = np.load(f)
    return d["y_true"], d["y_pred"], d["y_prob"][:, 1], d["fitzpatrick"]


def fig_roc(models, title, out, ref=None):
    plt.figure(figsize=(7, 6))
    for name, f in models.items():
        yt, _, pr, _ = _load(f)
        fpr, tpr, _ = roc_curve(yt, pr)
        auc = roc_auc_score(yt, pr)
        lw = 3 if (ref and name == ref) else 1.8
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="navy", lw=1.5, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out.name)


def fig_confusion(out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    labels = ["Eczema", "Psoriasis"]
    for ax, (name, f) in zip(axes.ravel(), ALL_MODELS.items()):
        yt, yp, _, _ = _load(f)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        pso_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() else float("nan")
        ax.set_title(f"{name}\npsoriasis recall = {pso_recall:.2f}", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion matrices (test set, N=171)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out.name)


def fig_auroc_bar(out):
    names, aucs, colors = [], [], []
    palette = {"image": "#4C72B0", "meta": "#DD8452", "fusion": "#55A868"}
    for name, f in ALL_MODELS.items():
        yt, _, pr, _ = _load(f)
        names.append(name); aucs.append(roc_auc_score(yt, pr))
        if "Metadata" in name:
            colors.append(palette["meta"])
        elif "fusion" in name or "Gate" in name:
            colors.append(palette["fusion"])
        else:
            colors.append(palette["image"])
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(names)), aucs, color=colors)
    for b, a in zip(bars, aucs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
    plt.axhline(0.5, ls="--", color="gray", lw=1)
    plt.xticks(range(len(names)), names, rotation=30, ha="right", fontsize=9)
    plt.ylabel("AUROC"); plt.ylim(0.4, 1.0)
    plt.title("AUROC by model  (blue=image, orange=metadata, green=fusion)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out.name)


def fig_fairness(out):
    bands = [3, 4, 5, 6]
    mat, names = [], []
    for name, f in ALL_MODELS.items():
        yt, yp, _, fitz = _load(f)
        row = []
        for b in bands:
            m = fitz == b
            row.append((yp[m] == yt[m]).mean() if m.sum() else np.nan)
        mat.append(row); names.append(name)
    mat = np.array(mat)
    plt.figure(figsize=(7, 7))
    im = plt.imshow(mat, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            plt.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9)
    counts = []
    for b in bands:
        _, _, _, fitz = _load(list(ALL_MODELS.values())[0])
        counts.append(int((fitz == b).sum()))
    plt.xticks(range(4), [f"FST {b}\n(n={c})" for b, c in zip(bands, counts)])
    plt.yticks(range(len(names)), names, fontsize=9)
    plt.colorbar(im, label="accuracy")
    plt.title("Accuracy per Fitzpatrick band\n(FST 6 has too few samples to trust)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out.name)


def fig_gate_weights(out):
    csv_path = FUS.parent / "gate_weights_per_epoch.csv"
    if not csv_path.exists():
        return
    ep, wi, wm = [], [], []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            ep.append(int(r["epoch"])); wi.append(float(r["mean_w_image"])); wm.append(float(r["mean_w_meta"]))
    plt.figure(figsize=(7, 5))
    plt.plot(ep, wi, "o-", label="image weight", color="#4C72B0")
    plt.plot(ep, wm, "s-", label="metadata weight", color="#DD8452")
    plt.axhline(0.5, ls="--", color="gray", label="equal (0.5)")
    plt.xlabel("epoch"); plt.ylabel("mean gate weight"); plt.ylim(-0.05, 1.05)
    plt.title("Gate network weighting per epoch\n(after fix: balanced ~0.56/0.44, no collapse)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out.name)


if __name__ == "__main__":
    fig_roc(IMAGE_MODELS, "ROC — image architectures (eczema vs psoriasis)",
            OUT / "roc_image_models.png", ref="Hybrid CNN-Transformer")
    fig_roc({"Hybrid (best image)": IMG / "hybrid.npz",
             "Metadata-only (MLP)": FUS / "metadata_mlp.npz",
             "Late fusion": FUS / "late_fusion.npz",
             "Gate network": FUS / "gate_network.npz"},
            "ROC — modality comparison", OUT / "roc_modality.png",
            ref="Metadata-only (MLP)")
    fig_confusion(OUT / "confusion_matrices.png")
    fig_auroc_bar(OUT / "auroc_comparison.png")
    fig_fairness(OUT / "fairness_by_tone.png")
    fig_gate_weights(OUT / "gate_weights.png")
    print("\nAll figures ->", OUT)
