"""Render every results CSV as a markdown table (results/improvements/results/TABLES.md), so the
report's numbers come straight from the experiment outputs rather than being retyped."""

from __future__ import annotations

import json

import pandas as pd
from common import RESULTS

COLS = [
    ("config", "Configuration"),
    ("int_auroc", "Internal AUROC"),
    ("int_fold_sd", "Fold SD"),
    ("int_bal_acc", "Bal. acc."),
    ("int_sens", "Psoriasis sens."),
    ("ext_auroc", "External AUROC"),
    ("ext_ci", "External 95% CI"),
    ("ext_I_IV", "Ext. FST I–IV"),
    ("ext_V_VI", "Ext. FST V–VI"),
    ("gap", "Gap (I–IV − V–VI)"),
]


def fmt(v):
    if isinstance(v, float):
        return "—" if v != v else f"{v:.3f}"  # nan -> dash
    return str(v).replace(" | ", " — ")  # a pipe would break the markdown table


def table(csv, title, note=""):
    d = pd.read_csv(RESULTS / csv)
    d["ext_ci"] = d.apply(lambda r: f"{r.ext_lo:.3f}–{r.ext_hi:.3f}", axis=1)
    cols = [(c, h) for c, h in COLS if c in d.columns]
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out.append("| " + " | ".join(h for _, h in cols) + " |")
    out.append("|" + "---|" * len(cols))
    for _, r in d.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c, _ in cols) + " |")
    return "\n".join(out) + "\n"


def adoption():
    d = pd.read_csv(RESULTS / "adoption_tests.csv")
    out = [
        "### Adoption tests (paired; internal CI from a patient-level bootstrap)",
        "",
        "| Technique | Internal ΔAUROC [95% CI] | Adopt? | External ΔAUROC [95% CI] | Ext. FST V–VI ΔAUROC [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for _, r in d.iterrows():
        out.append(
            f"| {r['test']} | {r.int_delta:+.3f} [{r.int_lo:+.3f}, {r.int_hi:+.3f}] | {r['adopt (internal CI > 0)']} | "
            f"{r.ext_delta:+.3f} [{r.ext_lo:+.3f}, {r.ext_hi:+.3f}] | {r.dark_delta:+.3f} [{r.dark_lo:+.3f}, {r.dark_hi:+.3f}] |"
        )
    return "\n".join(out) + "\n"


def quality():
    d = pd.read_csv(RESULTS / "synthetic_quality_harmonized.csv")
    d = d[d["mode"] == "cell"]
    out = [
        "### Synthetic-sample quality (fold 0, per-cell generation)",
        "",
        "| Generator | n synthetic | Real-vs-synthetic classifier AUROC (0.5 = indistinguishable) | Near-copy rate | Train-on-synthetic, test-on-real AUROC |",
        "|---|---|---|---|---|",
    ]
    for _, r in d.iterrows():
        out.append(
            f"| {r.method} | {int(r.n_synthetic)} | {r.C2ST_auroc:.3f} | {r.near_copy_rate:.0%} | {r.TSTR_auroc:.3f} |"
        )
    return "\n".join(out) + "\n"


def syn_images():
    """Follow-up: image-level synthetic data (synth_eval.py)."""
    t = pd.read_csv(RESULTS / "synthetic_images_tests.csv")
    q = pd.read_csv(RESULTS / "synthetic_images_quality.csv")
    ci = lambda r, k: f"{r[k + '_delta']:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]"  # noqa: E731
    out = [
        "### Follow-up: image-level synthetic data vs the final model (paired; internal CI from a patient-level bootstrap)",
        "",
        "| Training data | Internal ΔAUROC | Adopt? | Internal V–VI Δ | SCIN Δ | SCIN V–VI Δ |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        out.append(
            f"| {r['test']} | {ci(r, 'int')} | {r['adopt (internal CI > 0)']} | {ci(r, 'int_V_VI')} | {ci(r, 'ext')} | {ci(r, 'ext_V_VI')} |"
        )
    out += [
        "",
        "### Follow-up: image-level synthetic data quality",
        "",
        "| Set | n | Median skin ITA | Cosine to source | Real-vs-synthetic AUROC (0.5 ideal) | Train-on-synthetic → real AUROC | … on V–VI |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in q.iterrows():
        out.append(
            f"| {r.arm} | {int(r.n)} | {r.median_skin_ITA:.1f} | {fmt(r.get('cos_to_source', float('nan')))} | "
            f"{r.C2ST_auroc:.3f} | {r.TSTR_auroc:.3f} | {r.TSTR_auroc_V_VI:.3f} |"
        )
    return "\n".join(out) + "\n"


def fitz17k():
    """Second external test set, run once under the pre-specified plan (fitz17k_eval.py)."""
    d = pd.read_csv(RESULTS / "fitz17k.csv")
    t = pd.read_csv(RESULTS / "fitz17k_tests.csv")
    out = [
        "### Second external test set: Fitzpatrick17k (run once; bootstrap over near-duplicate clusters)",
        "",
        "| Labels | Tone labels | Model | n | V–VI n (psoriasis) | AUROC [95% CI] | FST I–IV | FST V–VI [95% CI] | Gap |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in d.iterrows():
        out.append(
            f"| {r['labels']} | {r.tone_source} | {r.model} | {int(r.n)} | {int(r.n_V_VI)} ({int(r.n_V_VI_psoriasis)}) | "
            f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}] | {r.I_IV:.3f} | {r.V_VI:.3f} [{r.V_VI_lo:.3f}, {r.V_VI_hi:.3f}] | {r.gap:+.3f} |"
        )
    out += [
        "",
        "| Labels | Tone labels | Comparison | Role | Overall Δ [95% CI] | I–IV Δ | V–VI Δ [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        role = r.role if isinstance(r.role, str) else ""
        out.append(
            f"| {r['labels']} | {r.tone_source} | {r.comparison} | {role} | {r.overall_delta:+.3f} [{r.overall_lo:+.3f}, {r.overall_hi:+.3f}] | "
            f"{r.I_IV_delta:+.3f} | {r.V_VI_delta:+.3f} [{r.V_VI_lo:+.3f}, {r.V_VI_hi:+.3f}] |"
        )
    return "\n".join(out) + "\n"


def main():
    parts = [
        "# Improvement experiments: all results tables",
        "",
        f"Logistic-regression C (chosen once by inner patient-grouped CV): {json.loads((RESULTS / 'params.json').read_text())['C']}",
        "",
    ]
    spec = [
        ("baseline.csv", "Baseline and backbone (original labels, all 1,125 images)"),
        ("labels.csv", "Point 6: label definition (scored on the harmonized subset, n = 872)"),
        ("backbones_harmonized.csv", "Point 1: foundation-model size"),
        ("imbalance_harmonized.csv", "Point 2: class and skin-tone imbalance"),
        ("augment_harmonized.csv", "Point 3: augmentation (features of augmented views) and TTA"),
        ("synthetic_harmonized.csv", "Points 4 and 7: synthetic training data (PCA-128 latent)"),
        (
            "missing_harmonized.csv",
            "Point 5: missing metadata",
            "Internal AUROC here is measured with the internal test folds' metadata blanked at SCIN's missing rates (age 51%, sex 44%, site 15%).",
        ),
        ("redundancy_harmonized.csv", "Point 6: redundant feature dimensions (PCA)"),
        ("fusion_dinov2_harmonized.csv", "Points 8–10: mid vs late fusion, DINOv2 features"),
        ("fusion_rn50ft_harmonized.csv", "Points 8–10: mid vs late fusion, the paper's fine-tuned ResNet-50 features"),
        ("final.csv", "Final model vs the paper (scored on the harmonized subset)"),
        ("endtoend.csv", "End-to-end fine-tuned ResNet-50 (harmonized labels; scored on the harmonized subset)"),
    ]
    for item in spec:
        f = RESULTS / item[0]
        if f.exists():
            parts.append(table(*item))
    if (RESULTS / "synthetic_quality_harmonized.csv").exists():
        parts.append(quality())
    if (RESULTS / "adoption_tests.csv").exists():
        parts.append(adoption())
    if (RESULTS / "synthetic_images.csv").exists():
        parts.append(
            table(
                "synthetic_images.csv",
                "Follow-up: image-level synthetic data (SD-Turbo; scored on the harmonized subset)",
                "int_V_VI (internal Fitzpatrick V–VI AUROC) is in synthetic_images.csv.",
            )
        )
    if (RESULTS / "synthetic_images_tests.csv").exists():
        parts.append(syn_images())
    if (RESULTS / "fitz17k.csv").exists():
        parts.append(fitz17k())
    (RESULTS / "TABLES.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", RESULTS / "TABLES.md")


if __name__ == "__main__":
    main()
