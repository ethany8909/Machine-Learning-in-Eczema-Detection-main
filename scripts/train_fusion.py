#!/usr/bin/env python
"""Week 3 — metadata model + multimodal fusion, on the folder-based split.

Two independently runnable stages:

  --stage metadata   Train metadata-only models (logistic + MLP). Fast; does not
                     need images or the Week-2 checkpoints.
  --stage fusion     Train fixed late-fusion + the gate network on top of the best
                     Week-2 image backbone. Needs results/image_models/checkpoints/.
  --stage all        Both (default).

Gate weights are logged every epoch (mean over val) to gate_weights_per_epoch.csv,
and the final gate-weight distribution is checked for 0.5/0.5 collapse.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dermafair.data.folder_split import build_multimodal_dataloaders
from dermafair.fairness import FairnessEvaluator
from dermafair.fairness.advanced_metrics import evaluate_all
from dermafair.models import build_fusion, build_image_model, build_metadata_model
from dermafair.models.trainer import TrainConfig, class_weights, predict, train_model
from dermafair.utils import get_logger, resolve_device, set_seed

log = get_logger()


def _device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return resolve_device(name)


def _train_labels(loader):
    return np.concatenate([b["label"].numpy() for b in loader])


def _fairness_split(preds):
    """Return (groups, y_true, y_pred, y_prob) with missing skin tone dropped."""
    fitz = preds.get("fitzpatrick")
    if fitz is None:
        return None, None, None, None
    keep = fitz != -1
    return fitz[keep], preds["y_true"][keep], preds["y_pred"][keep], preds["y_prob"][keep]


def _evaluate(name, preds, evaluator, n_bootstrap):
    groups, gy, gp, gpr = _fairness_split(preds)
    metrics = evaluate_all(preds["y_true"], preds["y_pred"], preds["y_prob"], groups=groups)
    if groups is not None:
        rep = evaluator.evaluate(gy, gp, groups, n_bootstrap=n_bootstrap)
        metrics["kruskal_h"], metrics["kruskal_p"] = rep.kruskal_h, rep.kruskal_p
        metrics["acc_gap_bootstrap"] = rep.bootstrap.get("accuracy", {})
    o = metrics["overall"]
    log.info(f"{name}: acc={o['accuracy']:.3f} bacc={o['balanced_accuracy']:.3f} "
             f"auroc={o['auroc']:.3f} f1={o['f1']:.3f} ece={o['ece']:.3f}")
    return metrics


def _pick_best_arch(image_metrics_json, override):
    if override != "auto":
        return override
    data = json.loads(Path(image_metrics_json).read_text(encoding="utf-8"))
    best = max(data.items(), key=lambda kv: kv[1].get("best_val_bacc", float("-inf")))
    return best[0]


@torch.no_grad()
def _mean_gate_weights(model, loader, device):
    model.eval()
    ws = []
    for batch in loader:
        model(batch["image"].to(device), batch["meta"].to(device))
        gw = getattr(model, "last_gate_weights", None)
        if gw is not None:
            ws.append(gw.cpu().numpy())
    if not ws:
        return None
    return np.concatenate(ws).mean(axis=0)  # [w_image, w_meta]


def run_metadata(args, device, out_dir, evaluator):
    tr, va, te, meta_dim = build_multimodal_dataloaders(
        args.split_root, args.metadata_csv, batch_size=args.batch_size,
        num_workers=args.num_workers, load_images=False,
    )
    log.info(f"metadata feature dim: {meta_dim}")
    cw = class_weights(_train_labels(tr))
    results = {}
    for kind in (["mlp"] if args.smoke else ["logistic", "mlp"]):
        log.info(f"=== metadata-only model: {kind} ===")
        model = build_metadata_model(kind, meta_dim, num_classes=2)
        cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                          patience=args.patience, device=device, modality="metadata", loss=args.loss)
        hist = train_model(model, tr, va, cfg, loss_weights=cw)
        torch.save(model.state_dict(), out_dir / "checkpoints" / f"metadata_{kind}.pt")
        preds = predict(model, te, cfg)
        np.savez(out_dir / "predictions" / f"metadata_{kind}.npz", **preds)
        m = _evaluate(f"metadata_{kind}", preds, evaluator, args.n_bootstrap)
        m["best_val_bacc"] = hist["best_val_bacc"]
        results[f"metadata_{kind}"] = m
    return results


def run_fusion(args, device, out_dir, evaluator):
    best_arch = _pick_best_arch(args.image_metrics, args.best_image_arch)
    log.info(f"fusion image backbone: {best_arch}")
    ckpt = Path(args.image_ckpt_dir) / f"{best_arch}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Image checkpoint not found: {ckpt}. Run Week-2 training first.")

    tr, va, te, meta_dim = build_multimodal_dataloaders(
        args.split_root, args.metadata_csv, batch_size=args.batch_size,
        num_workers=args.num_workers, load_images=True,
    )
    cw = class_weights(_train_labels(tr))
    results = {}
    strategies = ["gate_network"] if args.smoke else ["late_fusion", "gate_network"]
    for strategy in strategies:
        log.info(f"=== fusion: {strategy} ===")
        img = build_image_model(best_arch, num_classes=2, pretrained=True)
        img.load_state_dict(torch.load(ckpt, map_location=device))
        meta = build_metadata_model("mlp", meta_dim, num_classes=2)

        # --- FIX: warm-start the metadata branch from the trained metadata-only
        # model instead of random init (random init is what caused the gate to
        # collapse onto the image branch). ---
        meta_ckpt = Path(args.metadata_ckpt)
        if meta_ckpt.exists():
            meta.load_state_dict(torch.load(meta_ckpt, map_location=device))
            log.info(f"warm-started metadata branch from {meta_ckpt.name}")
        else:
            log.warning(f"metadata checkpoint {meta_ckpt} missing — training metadata branch from scratch")

        kwargs = {}
        if strategy == "gate_network":
            kwargs = {"hidden_dim": 128, "freeze_image_backbone": True,
                      "normalize_features": True}
        fusion = build_fusion(strategy, img, meta, num_classes=2, **kwargs)
        # freeze image branch for CPU tractability (image model already trained;
        # on CPU we cannot afford to fine-tune the whole backbone)
        for p in fusion.image_model.parameters():
            p.requires_grad = False
        # warm-start the per-branch heads from each branch's trained classifier
        if strategy == "gate_network":
            fusion.warm_start_heads()

        cb = None
        gate_log = []
        if strategy == "gate_network":
            def cb(epoch, model, _va=va, _dev=device, _log=gate_log):
                mw = _mean_gate_weights(model, _va, _dev)
                if mw is not None:
                    _log.append((epoch + 1, float(mw[0]), float(mw[1])))
                    log.info(f"  gate mean weights epoch {epoch+1}: "
                             f"image={mw[0]:.3f} meta={mw[1]:.3f}")

        # gate-entropy penalty: rewards using both modalities, preventing collapse
        aux = None
        if strategy == "gate_network" and args.gate_entropy > 0:
            aux = lambda m, _lam=args.gate_entropy: _lam * m.entropy_penalty()
            log.info(f"gate entropy penalty lambda = {args.gate_entropy}")

        cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                          patience=args.patience, device=device, modality="multimodal", loss=args.loss)
        hist = train_model(fusion, tr, va, cfg, loss_weights=cw, epoch_callback=cb, aux_loss_fn=aux)
        torch.save(fusion.state_dict(), out_dir / "checkpoints" / f"{strategy}.pt")

        preds = predict(fusion, te, cfg)
        np.savez(out_dir / "predictions" / f"{strategy}.npz", **preds)
        m = _evaluate(strategy, preds, evaluator, args.n_bootstrap)
        m["best_val_bacc"] = hist["best_val_bacc"]

        if strategy == "gate_network":
            with open(out_dir / "gate_weights_per_epoch.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["epoch", "mean_w_image", "mean_w_meta"])
                w.writerows(gate_log)
            gw = preds.get("gate_weights")
            if gw is not None:
                mean_img = float(gw[:, 0].mean())
                collapse = abs(mean_img - 0.5) < 0.02 and float(gw[:, 0].std()) < 0.02
                m["gate_mean_image_weight"] = mean_img
                m["gate_image_weight_std"] = float(gw[:, 0].std())
                m["gate_collapsed_to_half"] = bool(collapse)
                log.info(f"  gate final: mean image weight={mean_img:.3f} "
                         f"std={gw[:,0].std():.3f} collapsed={collapse}")
        results[strategy] = m
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="../dataset_split")
    ap.add_argument("--metadata-csv", default="../Skin_Metadata-1.csv")
    ap.add_argument("--stage", choices=["metadata", "fusion", "all"], default="all")
    ap.add_argument("--best-image-arch", default="auto")
    ap.add_argument("--image-ckpt-dir", default="results/image_models/checkpoints")
    ap.add_argument("--image-metrics", default="results/image_models/metrics.json")
    ap.add_argument("--metadata-ckpt", default="results/fusion/checkpoints/metadata_mlp.pt",
                    help="trained metadata-only MLP to warm-start the fusion metadata branch")
    ap.add_argument("--gate-entropy", type=float, default=0.1,
                    help="weight of the gate entropy penalty (0 disables; prevents modality collapse)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="results/fusion")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce",
                    help="ce=cross-entropy (default), focal=focal loss for imbalance")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.n_bootstrap = 1, 50

    set_seed(args.seed)
    device = _device(args.device)
    log.info(f"device: {device}")

    out_dir = Path(args.out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    evaluator = FairnessEvaluator(sensitive_attr="fitzpatrick")

    # preserve any earlier-stage results (e.g. metadata-only) already on disk
    metrics_path = out_dir / "metrics.json"
    results = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    if args.stage in ("metadata", "all"):
        results.update(run_metadata(args, device, out_dir, evaluator))
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    if args.stage in ("fusion", "all"):
        results.update(run_fusion(args, device, out_dir, evaluator))
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    log.info(f"Done. Metrics -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
