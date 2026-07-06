#!/usr/bin/env python
"""Train every image backbone in ``image_models.py`` on the folder-based
eczema/psoriasis split, then evaluate each with the full metric suite
(overall + calibration + skin-tone fairness).

Usage (from the dermafair/ package root):
    python scripts/train_image_models.py \
        --split-root ../dataset_split \
        --metadata-csv ../Skin_Metadata-1.csv \
        --epochs 50 --batch-size 32 --device auto

Quick smoke test (few images, 1 epoch, CPU) to validate the pipeline:
    python scripts/train_image_models.py --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dermafair.data.folder_split import build_dataloaders_from_folders
from dermafair.fairness import FairnessEvaluator
from dermafair.fairness.advanced_metrics import evaluate_all
from dermafair.models import build_image_model
from dermafair.models.trainer import TrainConfig, class_weights, predict, train_model
from dermafair.utils import get_logger, resolve_device, set_seed

log = get_logger()

ALL_ARCHS = ["cnn", "resnet50", "custom_resnet50", "vit_b16", "hybrid"]


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return resolve_device(name)


def _collect_train_labels(loader) -> np.ndarray:
    return np.concatenate([b["label"].numpy() for b in loader])


def _compile_markdown(results: dict, out_path: Path):
    lines = ["# Image-model results — eczema/dermatitis vs psoriasis/lichenoid\n"]
    lines.append("Positive class = psoriasis_lichenoid. Fairness groups = Fitzpatrick band.\n")
    # Overall table
    cols = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1",
            "sensitivity_recall", "specificity", "mcc", "ece", "brier"]
    lines.append("## Overall metrics\n")
    lines.append("| model | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for name, r in results.items():
        o = r["overall"]
        lines.append(f"| {name} | " + " | ".join(f"{o[c]:.3f}" for c in cols) + " |")
    # Fairness table
    fcols = ["equal_opportunity_diff", "equalized_odds_diff", "demographic_parity_diff",
             "accuracy_gap", "auroc_gap", "worst_group_accuracy", "worst_group_auroc"]
    lines.append("\n## Skin-tone fairness (Fitzpatrick)\n")
    lines.append("| model | " + " | ".join(fcols) + " | kruskal_p |")
    lines.append("|" + "---|" * (len(fcols) + 2))
    for name, r in results.items():
        s = r["fairness"]["summary"]
        kp = r.get("kruskal_p", float("nan"))
        cells = " | ".join(f"{s[c]:.3f}" if not np.isnan(s[c]) else "n/a" for c in fcols)
        lines.append(f"| {name} | {cells} | {kp:.3f} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="../dataset_split")
    ap.add_argument("--metadata-csv", default="../Skin_Metadata-1.csv")
    ap.add_argument("--architectures", nargs="+", default=ALL_ARCHS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out-dir", default="results/image_models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce",
                    help="ce=cross-entropy (default), focal=focal loss for imbalance")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch, tiny CNN+resnet only — validate the pipeline end to end")
    args = ap.parse_args()

    if args.smoke:
        args.architectures = ["cnn"]
        args.epochs = 1
        args.n_bootstrap = 50

    set_seed(args.seed)
    device = _resolve_device(args.device)
    log.info(f"device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders_from_folders(
        split_root=args.split_root, metadata_csv=args.metadata_csv,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log.info(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
             f"test={len(test_loader.dataset)}")

    cw = class_weights(_collect_train_labels(train_loader))
    log.info(f"class weights (eczema, psoriasis): {cw.tolist()}")

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    ckpt_dir = out_dir / "checkpoints"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    evaluator = FairnessEvaluator(sensitive_attr="fitzpatrick")
    results: dict = {}

    for arch in args.architectures:
        log.info(f"=== training image model: {arch} ===")
        model = build_image_model(arch, num_classes=2, pretrained=True)
        tcfg = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, device=device, modality="image", loss=args.loss,
        )
        hist = train_model(model, train_loader, val_loader, tcfg, loss_weights=cw)
        torch.save(model.state_dict(), ckpt_dir / f"{arch}.pt")

        preds = predict(model, test_loader, tcfg)
        np.savez(pred_dir / f"{arch}.npz", **preds)

        y_true, y_pred, y_prob = preds["y_true"], preds["y_pred"], preds["y_prob"]
        fitz = preds.get("fitzpatrick")
        # exclude missing skin-tone (-1) from fairness grouping
        if fitz is not None:
            keep = fitz != -1
            groups = fitz[keep]
            gy_true, gy_pred, gy_prob = y_true[keep], y_pred[keep], y_prob[keep]
        else:
            groups = gy_true = gy_pred = gy_prob = None

        metrics = evaluate_all(y_true, y_pred, y_prob, groups=groups)
        if groups is not None:
            report = evaluator.evaluate(gy_true, gy_pred, groups, n_bootstrap=args.n_bootstrap)
            metrics["kruskal_h"] = report.kruskal_h
            metrics["kruskal_p"] = report.kruskal_p
            metrics["acc_gap_bootstrap"] = report.bootstrap.get("accuracy", {})
        metrics["best_val_bacc"] = hist["best_val_bacc"]
        results[arch] = metrics

        o = metrics["overall"]
        log.info(f"{arch}: acc={o['accuracy']:.3f} bacc={o['balanced_accuracy']:.3f} "
                 f"auroc={o['auroc']:.3f} f1={o['f1']:.3f} ece={o['ece']:.3f}")

        # write incrementally so a slow/failed later model can't lose earlier work
        (out_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8")
        _compile_markdown(results, out_dir / "RESULTS.md")
    log.info(f"Done. Metrics -> {out_dir/'metrics.json'} and {out_dir/'RESULTS.md'}")


if __name__ == "__main__":
    main()
