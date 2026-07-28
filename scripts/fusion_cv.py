#!/usr/bin/env python
"""K-fold CV of the metadata + fusion models for one regime (default: triage).

For each fold it REUSES that fold's pre-trained image backbone
(results/cv_clean/checkpoints/<arch>_fold<i>.pt, trained WITHOUT fold i's test
images -> no leakage), then trains only the cheap metadata + fusion layer on top.
Reports mean +/- std across folds for metadata-only, late fusion, and the gate.

Usage:
    python scripts/fusion_cv.py --regime triage --backbone-arch resnet50 --k 5 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dermafair.data.folder_split import build_cv_multimodal_from_manifest
from dermafair.fairness.advanced_metrics import evaluate_all, group_fairness_metrics
from dermafair.models import build_fusion, build_image_model, build_metadata_model
from dermafair.models.trainer import TrainConfig, class_weights, predict, train_model
from dermafair.utils import get_logger, resolve_device, set_seed

log = get_logger()


def _device(n):
    return ("cuda" if torch.cuda.is_available() else "cpu") if n == "auto" else resolve_device(n)


def _metrics(preds):
    yt, yp, yp2 = preds["y_true"], preds["y_pred"], preds["y_prob"]
    m = evaluate_all(yt, yp, yp2, groups=None)["overall"]
    fitz = preds.get("fitzpatrick")
    gap = float("nan")
    if fitz is not None and (fitz != -1).any():
        keep = fitz != -1
        gap = group_fairness_metrics(yt[keep], yp[keep], yp2[keep], fitz[keep])["summary"]["accuracy_gap"]
    return {"auroc": m["auroc"], "accuracy": m["accuracy"], "acc_gap": gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="triage")
    ap.add_argument("--manifest", default="../manifest_clean.csv")
    ap.add_argument("--source-dirs", nargs="+", default=["../DATASET_0", "../DATASET_1"])
    ap.add_argument("--metadata-csv", default="../Skin_Metadata-1.csv")
    ap.add_argument("--backbone-arch", default="resnet50")
    ap.add_argument("--backbone-ckpt-dir", default="results/cv_clean/checkpoints")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gate-entropy", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="results/cv_triage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=99, help="cap folds run (for smoke tests)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = _device(args.device)
    out_dir = Path(args.out_dir); (out_dir / "tmp").mkdir(parents=True, exist_ok=True)
    log.info(f"device {device} | regime={args.regime} | backbone={args.backbone_arch} | {args.k}-fold")

    per_fold = {m: [] for m in ("metadata_mlp", "late_fusion", "gate_network")}

    for fold in range(min(args.k, args.max_folds)):
        bck = Path(args.backbone_ckpt_dir) / f"{args.backbone_arch}_fold{fold}.pt"
        if not bck.exists():
            raise FileNotFoundError(f"missing per-fold backbone {bck} (run image CV first)")

        # metadata-only loaders (no images, fast) + fusion loaders (with images)
        common = dict(manifest_csv=args.manifest, source_dirs=args.source_dirs,
                      metadata_csv=args.metadata_csv, test_fold=fold, k=args.k,
                      regime=args.regime, batch_size=args.batch_size)
        mtr, mva, mte, meta_dim = build_cv_multimodal_from_manifest(load_images=False, **common)
        cw = class_weights(np.concatenate([b["label"].numpy() for b in mtr]))
        log.info(f"--- fold {fold}: train={len(mtr.dataset)} test={len(mte.dataset)} meta_dim={meta_dim} ---")

        # 1) metadata-only
        meta = build_metadata_model("mlp", meta_dim, num_classes=2)
        mcfg = TrainConfig(epochs=args.epochs, lr=args.lr, patience=args.patience,
                           device=device, modality="metadata")
        train_model(meta, mtr, mva, mcfg, loss_weights=cw)
        per_fold["metadata_mlp"].append(_metrics(predict(meta, mte, mcfg)))
        meta_ckpt = out_dir / "tmp" / f"meta_fold{fold}.pt"
        torch.save(meta.state_dict(), meta_ckpt)

        # 2) fusion (late + gate) on the frozen per-fold backbone
        ftr, fva, fte, _ = build_cv_multimodal_from_manifest(load_images=True, **common)
        for strategy in ("late_fusion", "gate_network"):
            img = build_image_model(args.backbone_arch, num_classes=2, pretrained=True)
            img.load_state_dict(torch.load(bck, map_location=device))
            mm = build_metadata_model("mlp", meta_dim, num_classes=2)
            mm.load_state_dict(torch.load(meta_ckpt, map_location=device))     # warm-start
            kwargs = {"hidden_dim": 128, "freeze_image_backbone": True,
                      "normalize_features": True} if strategy == "gate_network" else {}
            fus = build_fusion(strategy, img, mm, num_classes=2, **kwargs)
            for p in fus.image_model.parameters():
                p.requires_grad = False
            aux = None
            if strategy == "gate_network":
                fus.warm_start_heads()
                aux = lambda m, _l=args.gate_entropy: _l * m.entropy_penalty()
            fcfg = TrainConfig(epochs=args.epochs, lr=args.lr, patience=args.patience,
                               device=device, modality="multimodal")
            train_model(fus, ftr, fva, fcfg, loss_weights=cw, aux_loss_fn=aux)
            per_fold[strategy].append(_metrics(predict(fus, fte, fcfg)))

        log.info(f"  fold {fold}: " + " | ".join(
            f"{m}={per_fold[m][-1]['auroc']:.3f}" for m in per_fold))
        (out_dir / "cv_per_fold.json").write_text(json.dumps(per_fold, indent=2, default=float), encoding="utf-8")

    # aggregate
    summ, lines = {}, [f"# Triage {args.k}-fold CV (backbone {args.backbone_arch}) — mean ± std\n",
                       "| model | AUROC | accuracy | acc gap |", "|---|---|---|---|"]
    for m, rows in per_fold.items():
        summ[m] = {}
        for k in ("auroc", "accuracy", "acc_gap"):
            v = np.array([r[k] for r in rows], float)
            summ[m][k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
        s = summ[m]
        lines.append(f"| {m} | {s['auroc']['mean']:.3f} ± {s['auroc']['std']:.3f} | "
                     f"{s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f} | "
                     f"{s['acc_gap']['mean']:.3f} ± {s['acc_gap']['std']:.3f} |")
    (out_dir / "cv_summary.json").write_text(json.dumps(summ, indent=2, default=float), encoding="utf-8")
    (out_dir / "CV_TRIAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Done.\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
