"""Mitigation experiment: does upweighting darker-skin cases during training
reduce the EXTERNAL skin-tone performance gap?

Baseline (no tone reweighting), measured on SCIN:
    light FST I-IV  AUROC 0.711
    dark  FST V-VI  AUROC 0.503     gap +0.208 [95% CI +0.024, +0.393]

Here each training sample is weighted by (class weight) x (inverse Fitzpatrick-band
frequency, capped) so scarce darker-skin cases contribute more to the loss. The
retrained model is then evaluated on SCIN with the identical protocol.

Either outcome is informative: a narrowed gap supports algorithmic mitigation, an
unchanged gap argues the limitation is data-driven and needs tone-balanced collection.

    python experiments/external/mitigation_tone_reweight.py [--epochs 15] [--cap 5.0]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score

from dermafair.paths import DATA_DIR, output_dir

REPORT_DIR = output_dir("external")

from torch.utils.data import DataLoader  # noqa: E402

from dermafair.data.folder_split import FolderSplitDataset, _build_transforms, _load_fitzpatrick_map  # noqa: E402
from dermafair.models import build_image_model  # noqa: E402

SPLIT = DATA_DIR / "dataset_split_clean"
META = DATA_DIR / "Skin_Metadata-1.csv"
IMAGES = DATA_DIR / "scin_images"
MANIFEST = DATA_DIR / "external_scin_manifest.csv"
OUT = REPORT_DIR / "MITIGATION_TONE_REWEIGHT.md"


def loaders(bs, workers=0):
    fz = _load_fitzpatrick_map(META)
    tr = FolderSplitDataset(SPLIT / "train", fz, _build_transforms(224, augment=True))
    va = FolderSplitDataset(SPLIT / "val", fz, _build_transforms(224, augment=False))
    return (
        DataLoader(tr, batch_size=bs, shuffle=True, num_workers=workers),
        DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers),
        tr,
    )


def weights_from(ds, cap):
    labs = [s[1] for s in ds.samples]
    fsts = [s[2] for s in ds.samples]
    lc, fc = Counter(labs), Counter(f for f in fsts if f > 0)
    n, k = len(labs), max(len(fc), 1)
    cls_w = {c: len(labs) / (2 * lc[c]) for c in lc}
    tone_w = {f: min(n / (k * fc[f]), cap) for f in fc}
    print(f"  class weights: { {k2: round(v, 2) for k2, v in cls_w.items()} }")
    print(f"  tone weights (capped {cap}): { {k2: round(v, 2) for k2, v in sorted(tone_w.items())} }")
    return cls_w, tone_w


@torch.no_grad()
def val_auroc(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for b in loader:
        p = torch.softmax(model(b["image"].to(dev)), 1)[:, 1]
        ys.append(b["label"].numpy())
        ps.append(p.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def boot_gap(y, p, f, n=4000, seed=42):
    rng = np.random.default_rng(seed)
    iL, iD = np.where((f >= 1) & (f <= 4))[0], np.where(f >= 5)[0]
    d = []
    for _ in range(n):
        a, b = rng.choice(iL, len(iL), True), rng.choice(iD, len(iD), True)
        if len(np.unique(y[a])) > 1 and len(np.unique(y[b])) > 1:
            d.append(roc_auc_score(y[a], p[a]) - roc_auc_score(y[b], p[b]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cap", type=float, default=5.0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, tds = loaders(args.batch_size)
    print(f"train={len(tds)} val={len(va.dataset)} | device {dev}")
    cls_w, tone_w = weights_from(tds, args.cap)

    model = build_image_model("resnet50", num_classes=2, pretrained=True).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction="none")

    best, best_state, bad = -np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in tr:
            x, y, f = b["image"].to(dev), b["label"].to(dev), b["fitzpatrick"].numpy()
            w = torch.tensor(
                [cls_w[int(yy)] * tone_w.get(int(ff), 1.0) for yy, ff in zip(b["label"].numpy(), f)],
                dtype=torch.float32,
                device=dev,
            )
            opt.zero_grad()
            loss = (ce(model(x), y) * w).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * y.size(0)
        a = val_auroc(model, va, dev)
        print(f"  epoch {ep + 1:02d} train_loss {tot / len(tds):.4f} val_auroc {a:.4f}", flush=True)
        if a > best:
            best, bad = a, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping")
                break
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), REPORT_DIR / "mitigation_resnet50_tone_reweighted.pt")

    # ---- external evaluation on SCIN (identical protocol) ----
    idx = {}
    for fp in IMAGES.iterdir():
        if fp.is_file():
            idx.setdefault(fp.name.split("__")[0], fp)
    rows, files = [], []
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        fp = idx.get(r["case_id"])
        if fp is not None:
            rows.append(r)
            files.append(fp)
    y = np.array([int(r["label"]) for r in rows])
    fst = np.array([int(r["fitzpatrick"]) for r in rows])
    tfm = _build_transforms(224, augment=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(files), args.batch_size):
            bt = torch.stack([tfm(Image.open(f).convert("RGB")) for f in files[i : i + args.batch_size]]).to(dev)
            preds.append(torch.softmax(model(bt), 1)[:, 1].cpu().numpy())
    p = np.concatenate(preds)

    L4 = (fst >= 1) & (fst <= 4)
    D = fst >= 5
    al, ad = roc_auc_score(y[L4], p[L4]), roc_auc_score(y[D], p[D])
    lo, hi = boot_gap(y, p, fst)

    L = [
        "# Mitigation experiment: tone-aware loss reweighting\n",
        "ResNet-50 retrained with per-sample weight = class weight x inverse Fitzpatrick-band frequency ",
        f"(capped at {args.cap}), then evaluated on SCIN under the identical external protocol.\n",
        "| Model | overall | light (I-IV) | dark (V-VI) | gap | 95% CI of gap |",
        "|---|---|---|---|---|---|",
        "| baseline (no reweighting) | 0.632 | 0.711 | 0.503 | +0.208 | +0.024 to +0.393 |",
        f"| **tone-reweighted** | **{roc_auc_score(y, p):.3f}** | **{al:.3f}** | **{ad:.3f}** | "
        f"**{al - ad:+.3f}** | {lo:+.3f} to {hi:+.3f} |",
        f"\n- internal validation AUROC of the reweighted model: {best:.3f}",
        f"- gap {'NARROWED' if (al - ad) < 0.208 else 'did NOT narrow'} versus baseline ({al - ad:+.3f} vs +0.208)",
        f"- gap CI {'excludes' if lo > 0 else 'includes'} zero\n",
        "## Interpretation\n",
        "If the gap persists, the limitation is data-driven rather than algorithmic: reweighting cannot",
        "create signal absent from the training distribution, and tone-balanced data collection is the",
        "necessary remedy. If it narrows, cost-sensitive training offers partial mitigation.",
    ]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    np.savez(REPORT_DIR / "mitigation_scin_predictions.npz", y_true=y, p=p, fitzpatrick=fst)
    print("\n".join(L))


if __name__ == "__main__":
    main()
