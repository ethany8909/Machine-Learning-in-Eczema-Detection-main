#!/usr/bin/env python
"""Week 4 — Fairness Evaluation Engine.

Runs the full Fitzpatrick-stratified fairness protocol across every model's saved
test predictions and emits the Week-4 deliverables:

  TABLE3_fairness_master.(csv|md)  - per-tone metrics + gaps + fairness score
                                     + Kruskal-Wallis (H,p) + bootstrapped CI on the
                                     accuracy gap, for all models.
  figure2_fairness_heatmap.png     - model x Fitzpatrick x {accuracy,TPR,FPR,F1}.
  figure3_accuracy_fairness_pareto.png - accuracy vs fairness score with Pareto frontier.

Uses the CLEAN-data predictions by default. FST 6 is tiny (report CIs, do not over-claim).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from dermafair.fairness import FairnessEvaluator
from dermafair.fairness.advanced_metrics import group_fairness_metrics

import os

ROOT = Path(__file__).resolve().parents[1]

def _dir(env, default):
    v = os.environ.get(env)
    return Path(v).resolve() if v else default

IMG = _dir("DERMAFAIR_IMG_DIR", ROOT / "results/image_models_clean/predictions")
FUS = _dir("DERMAFAIR_FUS_DIR", ROOT / "results/fusion_clean/predictions")
OUT = _dir("DERMAFAIR_FIG_OUT", ROOT / "results/week4_fairness")
OUT.mkdir(parents=True, exist_ok=True)
BANDS = [3, 4, 5, 6]
N_BOOT = 1000

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


def _load(f):
    d = np.load(f)
    fitz = d["fitzpatrick"]
    keep = fitz != -1                       # drop missing skin tone for fairness
    return d["y_true"][keep], d["y_pred"][keep], d["y_prob"][keep][:, 1], fitz[keep]


def _auroc(yt, p):
    return float(roc_auc_score(yt, p)) if len(np.unique(yt)) > 1 else float("nan")


def compute():
    ev = FairnessEvaluator(sensitive_attr="fitzpatrick")
    rows, per_tone_acc, per_tone = {}, {}, {}
    for name, f in MODELS.items():
        yt, yp, pp, g = _load(f)
        rep = ev.evaluate(yt, yp, g, n_bootstrap=N_BOOT)          # gaps, KW, bootstrap CI
        gm = group_fairness_metrics(yt, yp, pp, g)               # equalized odds, worst-group
        boot = rep.bootstrap.get("accuracy", {})
        rows[name] = {
            "overall_accuracy": rep.overall_accuracy,
            "auroc": _auroc(yt, pp),
            "max_accuracy_gap": rep.max_gap,
            "acc_gap_ci_low": boot.get("ci_low", float("nan")),
            "acc_gap_ci_high": boot.get("ci_high", float("nan")),
            "tpr_gap": rep.tpr_gap,
            "fpr_gap": rep.fpr_gap,
            "equalized_odds_diff": gm["summary"]["equalized_odds_diff"],
            "fairness_score": rep.fairness_score,
            "worst_group_acc": gm["summary"]["worst_group_accuracy"],
            "kruskal_h": rep.kruskal_h,
            "kruskal_p": rep.kruskal_p,
        }
        # per-tone metric dicts for the heatmap
        per_tone[name] = {b: rep.group_metrics.get(b, {}) for b in BANDS}
        per_tone_acc[name] = {b: rep.group_metrics.get(b, {}).get("accuracy", np.nan) for b in BANDS}
    return rows, per_tone


def write_table(rows):
    cols = ["overall_accuracy", "auroc", "max_accuracy_gap", "acc_gap_ci_low", "acc_gap_ci_high",
            "tpr_gap", "fpr_gap", "equalized_odds_diff", "fairness_score", "worst_group_acc",
            "kruskal_h", "kruskal_p"]
    with open(OUT / "TABLE3_fairness_master.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + cols)
        for name, r in rows.items():
            w.writerow([name] + [r[c] for c in cols])

    md = ["# Table 3 — Master fairness table (clean data, test set)\n",
          "Positive class = psoriasis. Groups = Fitzpatrick band (missing tone dropped). "
          "Accuracy-gap CI = 1000x bootstrap. Sig. gap if Kruskal-Wallis p < 0.05.\n",
          "| Model | Acc | AUROC | Acc gap [95% CI] | TPR gap | FPR gap | Eq.Odds | Fairness | Worst-grp | KW p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        md.append(
            f"| {name} | {r['overall_accuracy']:.3f} | {r['auroc']:.3f} | "
            f"{r['max_accuracy_gap']:.3f} [{r['acc_gap_ci_low']:.3f}, {r['acc_gap_ci_high']:.3f}] | "
            f"{r['tpr_gap']:.3f} | {r['fpr_gap']:.3f} | {r['equalized_odds_diff']:.3f} | "
            f"{r['fairness_score']:.3f} | {r['worst_group_acc']:.3f} | {r['kruskal_p']:.3f} |")
    (OUT / "TABLE3_fairness_master.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote TABLE3_fairness_master.(csv|md)")


def figure2_heatmap(per_tone):
    metrics = [("accuracy", "RdYlGn"), ("tpr", "RdYlGn"), ("fpr", "RdYlGn_r"), ("f1", "RdYlGn")]
    names = list(per_tone.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (metric, cmap) in zip(axes.ravel(), metrics):
        mat = np.array([[per_tone[n][b].get(metric, np.nan) for b in BANDS] for n in names], float)
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([f"FST {b}" for b in BANDS])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.set_title(f"{metric.upper()}  ({'lower=better' if metric=='fpr' else 'higher=better'})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Figure 2 — Per-Fitzpatrick metrics by model (clean data)\nFST 6 n≈4–5: interpret with caution",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "figure2_fairness_heatmap.png", dpi=150)
    plt.close()
    print("wrote figure2_fairness_heatmap.png")


def figure3_pareto(rows):
    names = list(rows.keys())
    acc = np.array([rows[n]["overall_accuracy"] for n in names])
    fair = np.array([rows[n]["fairness_score"] for n in names])
    # Pareto frontier: maximize both accuracy and fairness
    order = np.argsort(-acc)
    frontier, best_fair = [], -np.inf
    for idx in order:
        if fair[idx] >= best_fair:
            frontier.append(idx); best_fair = fair[idx]
    frontier = sorted(frontier, key=lambda i: acc[i])

    plt.figure(figsize=(8.5, 6.5))
    plt.scatter(acc, fair, s=70, color="#4C72B0", zorder=3)
    for n, a, fr in zip(names, acc, fair):
        plt.annotate(n, (a, fr), fontsize=9, xytext=(5, 4), textcoords="offset points")
    plt.plot(acc[frontier], fair[frontier], "--", color="#55A868", lw=2, label="Pareto frontier", zorder=2)
    plt.xlabel("Overall accuracy"); plt.ylabel("Fairness score  (1 − max acc gap / overall acc)")
    plt.title("Figure 3 — Accuracy vs fairness trade-off (clean data)")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figure3_accuracy_fairness_pareto.png", dpi=150)
    plt.close()
    print("wrote figure3_accuracy_fairness_pareto.png")


if __name__ == "__main__":
    rows, per_tone = compute()
    write_table(rows)
    figure2_heatmap(per_tone)
    figure3_pareto(rows)
    # console summary
    print("\nmodel                overall_acc  auroc  max_acc_gap  fairness  KW_p")
    for n, r in rows.items():
        print(f"{n:18} {r['overall_accuracy']:11.3f} {r['auroc']:6.3f} {r['max_accuracy_gap']:11.3f} "
              f"{r['fairness_score']:8.3f} {r['kruskal_p']:6.3f}")
    print(f"\nDeliverables -> {OUT}")
