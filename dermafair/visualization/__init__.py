"""Publication figures + auto-compiled fairness report.

Figures map directly onto the manuscript:
  fairness_heatmap          -> Figure 2
  accuracy_fairness_pareto  -> Figure 3
  gate_weights_by_tone      -> Figure 4 (signature)
  fusion_comparison         -> Figure 5
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fairness_heatmap(master_df: pd.DataFrame, out_path: Path, metric_cols=None):
    """Heatmap of fairness metrics (rows=models, cols=metrics)."""
    metric_cols = metric_cols or [
        "max_accuracy_gap", "tpr_gap", "fpr_gap", "fairness_score"
    ]
    data = master_df.set_index("model")[metric_cols]
    fig, ax = plt.subplots(figsize=(1.6 * len(metric_cols) + 2, 0.6 * len(data) + 2))
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn_r")
    # fairness_score is "higher is better" — invert its color sense via annotation only
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "—",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="gap (lower=fairer)")
    ax.set_title("Fairness metrics by architecture")
    fig.tight_layout()
    _save(fig, out_path)


def accuracy_fairness_pareto(master_df: pd.DataFrame, out_path: Path):
    """Scatter of overall accuracy vs. fairness score, Pareto frontier marked."""
    df = master_df.dropna(subset=["overall_accuracy", "fairness_score"]).copy()
    x = df["overall_accuracy"].to_numpy()
    y = df["fairness_score"].to_numpy()

    # Pareto frontier: non-dominated on (max acc, max fairness)
    order = np.argsort(-x)
    frontier, best_y = [], -np.inf
    for i in order:
        if y[i] >= best_y:
            frontier.append(i)
            best_y = y[i]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=70)
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["overall_accuracy"], r["fairness_score"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    fx, fy = x[frontier], y[frontier]
    fo = np.argsort(fx)
    ax.plot(fx[fo], fy[fo], "--", alpha=0.6, label="Pareto frontier")
    ax.set_xlabel("Overall accuracy")
    ax.set_ylabel("Fairness score (1 − max gap / acc)")
    ax.set_title("Accuracy–fairness trade-off")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def gate_weights_by_tone(weights_df: pd.DataFrame, out_path: Path,
                         band_col="fitzpatrick", img_w_col="w_image"):
    """Signature figure: mean image-vs-metadata gate weight per Fitzpatrick band."""
    grouped = weights_df.groupby(band_col)[img_w_col].agg(["mean", "std"]).reset_index()
    bands = grouped[band_col].astype(str).to_numpy()
    means = grouped["mean"].to_numpy()
    stds = grouped["std"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(bands, means, yerr=stds, capsize=4, label="image weight")
    ax.bar(bands, 1 - means, bottom=means, alpha=0.5, label="metadata weight")
    ax.axhline(0.5, ls="--", color="k", alpha=0.4)
    ax.set_xlabel("Fitzpatrick band")
    ax.set_ylabel("Mean gate weight")
    ax.set_ylim(0, 1)
    ax.set_title("Learned modality weighting across skin tones")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def fusion_comparison(master_df: pd.DataFrame, out_path: Path,
                      models=("late_fusion", "gate_network")):
    """Grouped bars comparing fairness metrics for late-fusion vs. gate."""
    metrics = ["max_accuracy_gap", "tpr_gap", "fpr_gap"]
    sub = master_df[master_df["model"].isin(models)].set_index("model")
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, m in enumerate(models):
        if m in sub.index:
            ax.bar(x + (k - 0.5) * width, sub.loc[m, metrics].to_numpy(),
                   width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Gap (lower = fairer)")
    ax.set_title("Late-fusion vs. gate network")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def compile_report(master_df: pd.DataFrame, out_path: Path, run_name: str = ""):
    """Auto-write a narrative fairness_report.md from the master table."""
    df = master_df.copy()
    fairest = df.loc[df["fairness_score"].idxmax()] if df["fairness_score"].notna().any() else None
    most_acc = df.loc[df["overall_accuracy"].idxmax()] if df["overall_accuracy"].notna().any() else None

    lines = [f"# Fairness Report — {run_name}", ""]
    lines.append("## Summary")
    if most_acc is not None:
        lines.append(f"- Highest overall accuracy: **{most_acc['model']}** "
                     f"({most_acc['overall_accuracy']:.3f}).")
    if fairest is not None:
        lines.append(f"- Fairest model (highest fairness score): **{fairest['model']}** "
                     f"({fairest['fairness_score']:.3f}, max gap {fairest['max_accuracy_gap']:.3f}).")
    lines += ["", "## Per-model results", "",
              df.to_markdown(index=False, floatfmt=".3f")]
    lines += ["", "## Notes",
              "- Models with `significant_gap = True` show a statistically significant "
              "accuracy disparity across Fitzpatrick bands (Kruskal-Wallis, α=0.05).",
              "- Interpret all gaps alongside their bootstrapped CIs; small per-band N "
              "limits statistical power."]
    Path(out_path).write_text("\n".join(lines))


def _save(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
