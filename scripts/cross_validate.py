#!/usr/bin/env python
"""K-fold cross-validation for the image models, using the fold-assigned manifest
(subject-level, joint class x Fitzpatrick stratified) from clean_dataset.py.

Reports mean +/- std across folds for each architecture, giving stable headline
numbers with error bars instead of a single-split point estimate.

Usage (from dermafair/):
    python scripts/cross_validate.py --manifest ../manifest_clean.csv \
        --architectures cnn resnet50 custom_resnet50 --k 5 --epochs 15 --device cpu

Note (CPU): each fold retrains every model, so wall-time ~= k x single-split time.
On CPU prefer the lighter backbones; run vit_b16/hybrid CV on a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dermafair.data.folder_split import build_cv_dataloaders_from_manifest
from dermafair.fairness import FairnessEvaluator
from dermafair.fairness.advanced_metrics import evaluate_all
from dermafair.models import build_image_model
from dermafair.models.trainer import TrainConfig, class_weights, predict, train_model
from dermafair.utils import get_logger, resolve_device, set_seed

log = get_logger()
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "ece"]


def _device(name):
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else resolve_device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../manifest_clean.csv")
    ap.add_argument("--source-dirs", nargs="+", default=["../DATASET_0", "../DATASET_1"])
    ap.add_argument("--architectures", nargs="+", default=["cnn", "resnet50", "custom_resnet50"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--out-dir", default="results/cv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = _device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"device: {device} | {args.k}-fold CV | archs: {args.architectures}")

    # per_fold[arch] = list of metric dicts (one per fold)
    per_fold = {a: [] for a in args.architectures}
    evaluator = FairnessEvaluator(sensitive_attr="fitzpatrick")
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(args.k):
        tr, va, te = build_cv_dataloaders_from_manifest(
            args.manifest, args.source_dirs, test_fold=fold, k=args.k,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        cw = class_weights(np.concatenate([b["label"].numpy() for b in tr]))
        log.info(f"--- fold {fold}: train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)} ---")

        for arch in args.architectures:
            model = build_image_model(arch, num_classes=2, pretrained=True)
            cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                              patience=args.patience, device=device, modality="image", loss=args.loss)
            train_model(model, tr, va, cfg, loss_weights=cw)
            preds = predict(model, te, cfg)
            # (a) persist the per-fold best model + its test predictions
            torch.save(model.state_dict(), ckpt_dir / f"{arch}_fold{fold}.pt")
            np.savez(pred_dir / f"{arch}_fold{fold}.npz", **preds)

            m = evaluate_all(preds["y_true"], preds["y_pred"], preds["y_prob"], groups=None)
            row = {k: m["overall"][k] for k in METRICS}
            fitz = preds.get("fitzpatrick")
            if fitz is not None and (fitz != -1).any():
                from dermafair.fairness.advanced_metrics import group_fairness_metrics
                keep = fitz != -1
                gm = group_fairness_metrics(preds["y_true"][keep], preds["y_pred"][keep],
                                            preds["y_prob"][keep], fitz[keep])
                row["accuracy_gap"] = gm["summary"]["accuracy_gap"]
            per_fold[arch].append(row)
            log.info(f"  {arch} fold {fold}: auroc={row['auroc']:.3f} acc={row['accuracy']:.3f} f1={row['f1']:.3f}")
            # persist incrementally
            (out_dir / "cv_per_fold.json").write_text(json.dumps(per_fold, indent=2, default=float), encoding="utf-8")

    # ---- aggregate mean +/- std ----
    summary = {}
    for arch, rows in per_fold.items():
        if not rows:
            continue
        summary[arch] = {}
        for k in list(rows[0].keys()):
            vals = np.array([r[k] for r in rows], dtype=float)
            summary[arch][k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ---- markdown table ----
    lines = [f"# {args.k}-fold cross-validation (mean +/- std across folds)\n",
             "| model | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    for arch, s in summary.items():
        cells = " | ".join(f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}" for k in METRICS)
        lines.append(f"| {arch} | {cells} |")
    (out_dir / "CV_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"Done. Summary -> {out_dir/'cv_summary.json'} and {out_dir/'CV_RESULTS.md'}")


if __name__ == "__main__":
    main()
