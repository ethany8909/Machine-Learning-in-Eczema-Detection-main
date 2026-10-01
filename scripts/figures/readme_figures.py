"""Summary figures for the README (light and dark variants), drawn from the result tables.

    python scripts/figures/readme_figures.py

Reads  results/improvements/results/{final,final_vs_paper,adoption_tests,
       synthetic_images_tests,fitz17k,fitz17k_tests}.csv
Writes docs/figures/{cross_cohort,technique_effects}_{light,dark}.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dermafair.paths import REPO_ROOT, RESULTS_DIR  # noqa: E402

TABLES = RESULTS_DIR / "improvements" / "results"
OUT = REPO_ROOT / "docs" / "figures"

# Validated two-slot categorical palette (blue = new model, orange = original model) and chart chrome.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        new="#2a78d6",
        old="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        new="#3987e5",
        old="#d95926",
    ),
}
plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 11, "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"]}
)
matplotlib.set_loglevel("error")


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def cross_cohort_rows():
    final = pd.read_csv(TABLES / "final.csv").set_index("config")
    paper = final.loc[[c for c in final.index if c.startswith("paper")][0]]
    new = final.loc[[c for c in final.index if c.startswith("final")][0]]
    vs = pd.read_csv(TABLES / "final_vs_paper.csv").iloc[0]
    f = pd.read_csv(TABLES / "fitz17k.csv")
    f = f[(f["labels"] == "primary") & (f["tone_source"] == "fitzpatrick_scale")].set_index("model")
    fp, fn = f.loc[[m for m in f.index if m.startswith("M0")][0]], f.loc[[m for m in f.index if m.startswith("M1")][0]]
    ft = pd.read_csv(TABLES / "fitz17k_tests.csv")
    ft = ft[(ft["labels"] == "primary") & (ft["tone_source"] == "fitzpatrick_scale") & (ft["role"] == "PRIMARY")].iloc[
        0
    ]
    return [  # label, original, new, delta, lo, hi
        ("Internal CV (872 images)", paper.int_auroc, new.int_auroc, vs.int_delta, vs.int_lo, vs.int_hi),
        ("SCIN, all skin tones", paper.ext_auroc, new.ext_auroc, vs.ext_delta, vs.ext_lo, vs.ext_hi),
        ("SCIN, Fitzpatrick V–VI", paper.ext_V_VI, new.ext_V_VI, vs.V_VI_delta, vs.V_VI_lo, vs.V_VI_hi),
        ("Fitzpatrick17k, all skin tones", fp.auroc, fn.auroc, ft.overall_delta, ft.overall_lo, ft.overall_hi),
        ("Fitzpatrick17k, Fitzpatrick V–VI", fp.V_VI, fn.V_VI, ft.V_VI_delta, ft.V_VI_lo, ft.V_VI_hi),
    ]


def cross_cohort(mode):
    t, rows = THEMES[mode], cross_cohort_rows()
    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200, facecolor=t["surface"])
    _style(ax, t)
    for i, (_label, old, new, d, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([old, new], [y, y], color=t["axis"], linewidth=2, solid_capstyle="round", zorder=1)
        tie = abs(old - new) < 0.002  # draw the original model as a ring around the final model
        ax.scatter([old], [y], s=190 if tie else 70, color=t["old"], edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.scatter([new], [y], s=70, color=t["new"], edgecolor=t["surface"], linewidth=2, zorder=4)
        sig = lo > 0 or hi < 0
        d = round(d, 3) + 0.0  # no "-0.000"
        ax.text(
            1.005,
            y,
            f"Δ {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]{'  *' if sig else ''}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            color=t["ink"] if sig else t["ink2"],
            fontsize=10,
            fontweight="bold" if sig else "normal",
        )
    ax.axvline(0.5, color=t["muted"], linewidth=1, zorder=0)
    ax.text(0.503, -0.5, "chance", color=t["muted"], fontsize=9, ha="left", va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=t["ink"])
    ax.set_xlim(0.45, 0.92)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("AUROC", color=t["ink2"])
    ax.scatter([], [], s=70, color=t["old"], label="Original model (fine-tuned ResNet-50)")
    ax.scatter([], [], s=70, color=t["new"], label="Final model (DINOv2 + matched labels)")
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=10, handletextpad=0.3)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.text(0.995, 0.015, "* 95% CI excludes 0", color=t["muted"], fontsize=9, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / f"cross_cohort_{mode}.png", facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)


def technique_rows():
    a = pd.read_csv(TABLES / "adoption_tests.csv").set_index("test")
    s = pd.read_csv(TABLES / "synthetic_images_tests.csv").set_index("test")
    vs = pd.read_csv(TABLES / "final_vs_paper.csv").iloc[0]

    def get(df, key):
        r = df.loc[[k for k in df.index if k.startswith(key)][0]]
        return r.int_delta, r.int_lo, r.int_hi

    return [
        ("Foundation model + matched labels (final vs original)", vs.int_delta, vs.int_lo, vs.int_hi),
        ("Foundation-model features alone (DINOv2 ViT-B)", *get(a, "point 1: DINOv2 ViT-B")),
        ("SCIN-matched labels alone", *get(a, "point 6: SCIN-matched")),
        ("Larger foundation model (ViT-L vs ViT-B)", *get(a, "point 1: DINOv2 ViT-L")),
        ("Mid fusion vs late fusion", *get(a, "points 8-10: mid fusion vs late")),
        ("Mid fusion vs image only", *get(a, "points 8-10: mid fusion vs image")),
        ("Class-weighted loss", *get(a, "point 2: class-weighted")),
        ("Strong augmentation + test-time augmentation", *get(a, "point 3: strong augmentation + TTA")),
        ("Synthetic features: SMOTE", *get(a, "points 4/7: SMOTE")),
        ("Synthetic features: conditional VAE", *get(a, "points 4/7: conditional VAE")),
        ("Synthetic features: conditional diffusion", *get(a, "points 4/7: conditional diffusion")),
        ("Generated images: skin darkening", *get(s, "+ darkened only")),
        ("Generated images: text-to-image", *get(s, "+ text-to-image")),
        ("Feature reduction (PCA, 32 components)", *get(a, "point 6: PCA-32")),
    ]


def technique_effects(mode):
    t, rows = THEMES[mode], technique_rows()
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200, facecolor=t["surface"])
    _style(ax, t)
    for i, (_label, d, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=t["new"], linewidth=2, solid_capstyle="round", alpha=1 if sig else 0.45)
        ax.scatter(
            [d],
            [y],
            s=95 if sig else 64,
            zorder=3,
            linewidth=2,
            color=t["new"] if sig else t["surface"],
            edgecolor=t["new"] if not sig else t["surface"],
        )
    ax.axvline(0, color=t["ink2"], linewidth=1, zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=t["ink"])
    ax.set_xlabel("Change in internal AUROC (95% CI, patient-level bootstrap)", color=t["ink2"])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    fig.text(0.995, 0.01, "Filled: interval excludes 0", color=t["muted"], fontsize=9, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / f"technique_effects_{mode}.png", facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        cross_cohort(mode)
        technique_effects(mode)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
