#!/usr/bin/env python
"""Step 2 of the pipeline: train all seven models under identical conditions
and write their test-set predictions for the fairness stage.

This script wires together the dataset, models, and trainer. The Dataset class
is dataset-specific; a reference implementation for DermaCon-IN lives in
``dermafair.data.dermacon`` (you adapt it to the exact CSV/column layout).

Usage:
    python scripts/train_all.py --config configs/dermacon.yaml
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dermafair.models import build_fusion, build_image_model, build_metadata_model
from dermafair.models.trainer import TrainConfig, class_weights, predict, train_model
from dermafair.utils import get_logger, load_config, resolve_device, set_seed

log = get_logger()


def _save_predictions(out_dir: Path, name: str, preds: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}.npz", **preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    log.info(f"device: {device}")

    # ---- Dataset wiring -----------------------------------------------------
    # Adapt this block to your dataset. Expected: a torch Dataset yielding dicts
    # with keys: image [3,H,W], meta [F], label (int), fitzpatrick (int).
    try:
        from dermafair.data.dermacon import build_dataloaders
    except ImportError:
        log.error(
            "No dataset adapter found. Implement dermafair/data/dermacon.py:"
            " build_dataloaders(cfg) -> (train_loader, val_loader, test_loader, meta_dim)."
            " A template is provided in the docstring of that file."
        )
        raise

    train_loader, val_loader, test_loader, meta_dim = build_dataloaders(cfg)

    run_dir = Path(cfg["output"]["dir"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "predictions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # class weights from train labels
    train_labels = np.concatenate([b["label"].numpy() for b in train_loader])
    cw = class_weights(train_labels)

    image_models = {}
    val_bacc = {}

    # ---- 1. Image architectures --------------------------------------------
    for arch in cfg["train"]["architectures"]:
        log.info(f"=== training image model: {arch} ===")
        model = build_image_model(arch, num_classes=2, pretrained=True)
        tcfg = TrainConfig(
            epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            patience=cfg["train"]["early_stopping_patience"],
            device=device, modality="image",
        )
        hist = train_model(model, train_loader, val_loader, tcfg, loss_weights=cw)
        torch.save(model.state_dict(), ckpt_dir / f"{arch}.pt")
        preds = predict(model, test_loader, tcfg)
        _save_predictions(pred_dir, arch, preds)
        image_models[arch] = model
        val_bacc[arch] = hist["best_val_bacc"]

    # pick best image backbone for fusion
    best_arch = (
        max(val_bacc, key=val_bacc.get)
        if cfg["fusion"]["best_image_backbone"] == "auto"
        else cfg["fusion"]["best_image_backbone"]
    )
    log.info(f"best image backbone for fusion: {best_arch}")

    # ---- 2. Metadata-only model --------------------------------------------
    log.info("=== training metadata-only model ===")
    meta_model = build_metadata_model(cfg["fusion"]["metadata_model"], meta_dim, num_classes=2)
    mcfg = TrainConfig(
        epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        patience=cfg["train"]["early_stopping_patience"],
        device=device, modality="metadata",
    )
    train_model(meta_model, train_loader, val_loader, mcfg, loss_weights=cw)
    torch.save(meta_model.state_dict(), ckpt_dir / "metadata.pt")
    _save_predictions(pred_dir, "metadata", predict(meta_model, test_loader, mcfg))

    # ---- 3. Fusion: late-fusion + gate network -----------------------------
    for strategy in cfg["fusion"]["strategies"]:
        log.info(f"=== training fusion: {strategy} ===")
        # fresh copies so branches train within the fusion graph
        img_branch = build_image_model(best_arch, num_classes=2, pretrained=True)
        img_branch.load_state_dict(image_models[best_arch].state_dict())
        meta_branch = build_metadata_model(cfg["fusion"]["metadata_model"], meta_dim, 2)

        kwargs = {}
        if strategy == "gate_network":
            kwargs = {
                "hidden_dim": cfg["fusion"]["gate"]["hidden_dim"],
                "freeze_image_backbone": cfg["fusion"]["gate"]["freeze_image_backbone"],
            }
        fusion = build_fusion(strategy, img_branch, meta_branch, num_classes=2, **kwargs)

        fcfg = TrainConfig(
            epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            patience=cfg["train"]["early_stopping_patience"],
            device=device, modality="multimodal",
        )
        train_model(fusion, train_loader, val_loader, fcfg, loss_weights=cw)
        torch.save(fusion.state_dict(), ckpt_dir / f"{strategy}.pt")
        _save_predictions(pred_dir, strategy, predict(fusion, test_loader, fcfg))

    log.info(f"All models trained. Predictions in {pred_dir}")
    log.info("Next: python scripts/run_fairness.py --config <config>")


if __name__ == "__main__":
    main()
