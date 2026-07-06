#!/usr/bin/env python
"""Montage of the most confidently misclassified test images for a chosen model,
each annotated with true/predicted label, model confidence, skin tone, and a
data-driven 'likely reason'. Reconstructs filenames from the deterministic
(shuffle=False) test order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dermafair.data.folder_split import FolderSplitDataset, _load_fitzpatrick_map, _build_transforms

ROOT = Path(__file__).resolve().parents[1]
CLASS = {0: "Eczema", 1: "Psoriasis"}


def reason(true, pred, conf, fst):
    bits = []
    if conf >= 0.85:
        bits.append("high-confidence error (model was sure but wrong)")
    elif conf <= 0.65:
        bits.append("low-confidence / borderline case")
    if fst in (5, 6):
        bits.append(f"darker skin tone (FST {fst}) — lower image contrast")
    elif fst == -1:
        bits.append("missing skin-tone metadata")
    if true == 1:
        bits.append("minority class (psoriasis) — fewer training examples")
    if not bits:
        bits.append("ambiguous morphology between the two conditions")
    return "; ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid")
    ap.add_argument("--split-root", default=str(ROOT.parent / "dataset_split"))
    ap.add_argument("--metadata-csv", default=str(ROOT.parent / "Skin_Metadata-1.csv"))
    ap.add_argument("--pred", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results/figures/misclassified.png"))
    args = ap.parse_args()

    pred_path = Path(args.pred) if args.pred else ROOT / f"results/image_models/predictions/{args.model}.npz"
    d = np.load(pred_path)
    yt, yp, prob = d["y_true"], d["y_pred"], d["y_prob"]

    # rebuild the test dataset in the same order predictions were written
    fitz_map = _load_fitzpatrick_map(Path(args.metadata_csv))
    ds = FolderSplitDataset(Path(args.split_root) / "test", fitz_map,
                            _build_transforms(224, augment=False))
    samples = ds.samples  # (path, label, fitz, name) in prediction order
    assert len(samples) == len(yt), f"order mismatch {len(samples)} vs {len(yt)}"

    wrong = np.where(yp != yt)[0]
    conf = prob[np.arange(len(yt)), yp]          # confidence in the (wrong) predicted class
    wrong = wrong[np.argsort(-conf[wrong])][:args.n]   # most confident mistakes first

    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, idx in zip(np.atleast_1d(axes).ravel(), wrong):
        path, label, fst, name = samples[idx]
        ax.imshow(Image.open(path).convert("RGB"))
        ax.axis("off")
        c = float(conf[idx])
        ax.set_title(
            f"True: {CLASS[label]}  |  Pred: {CLASS[yp[idx]]}  ({c:.0%})\n"
            f"FST {fst if fst!=-1 else '?'}\n{reason(label, yp[idx], c, fst)}",
            fontsize=8,
        )
    plt.suptitle(f"Most confident misclassifications — {args.model}", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
