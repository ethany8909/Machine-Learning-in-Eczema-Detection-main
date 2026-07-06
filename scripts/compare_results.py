#!/usr/bin/env python
"""Build a comparison table: single-split (clean) metrics vs k-fold CV mean +/- std.

Reads (defaults target the clean-data overnight outputs):
  - results/image_models_clean/metrics.json   (single-split image models)
  - results/fusion_clean/metrics.json          (metadata + fusion)
  - results/cv_clean/cv_summary.json           (CV mean/std per arch)

Writes results/cv_clean/COMPARISON.md. Missing inputs are skipped gracefully.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH_ORDER = ["cnn", "resnet50", "custom_resnet50", "vit_b16", "hybrid"]


def _load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-metrics", default="results/image_models_clean/metrics.json")
    ap.add_argument("--fusion-metrics", default="results/fusion_clean/metrics.json")
    ap.add_argument("--cv-summary", default="results/cv_clean/cv_summary.json")
    ap.add_argument("--out", default="results/cv_clean/COMPARISON.md")
    args = ap.parse_args()

    img = _load(ROOT / args.image_metrics) or {}
    fus = _load(ROOT / args.fusion_metrics) or {}
    cv = _load(ROOT / args.cv_summary) or {}

    lines = ["# Clean-data results — single split vs 5-fold cross-validation\n"]

    # 1) single-split, all models
    lines.append("## Single-split (clean) — all models\n")
    lines.append("| model | accuracy | AUROC | F1 | ECE |")
    lines.append("|---|---|---|---|---|")
    for name, d in {**img, **fus}.items():
        if "overall" not in d:
            continue
        o = d["overall"]
        lines.append(f"| {name} | {o['accuracy']:.3f} | {o['auroc']:.3f} | {o['f1']:.3f} | {o['ece']:.3f} |")

    # 2) single-split vs CV for image backbones
    if cv:
        lines.append("\n## Image backbones — single split vs 5-fold CV\n")
        lines.append("| model | AUROC (single) | AUROC (CV mean ± std) | acc (single) | acc (CV mean ± std) |")
        lines.append("|---|---|---|---|---|")
        for arch in ARCH_ORDER:
            if arch not in cv:
                continue
            s_auroc = img.get(arch, {}).get("overall", {}).get("auroc")
            s_acc = img.get(arch, {}).get("overall", {}).get("accuracy")
            cva = cv[arch]["auroc"]; cvacc = cv[arch]["accuracy"]
            su = f"{s_auroc:.3f}" if s_auroc is not None else "—"
            sa = f"{s_acc:.3f}" if s_acc is not None else "—"
            lines.append(f"| {arch} | {su} | {cva['mean']:.3f} ± {cva['std']:.3f} | "
                         f"{sa} | {cvacc['mean']:.3f} ± {cvacc['std']:.3f} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
