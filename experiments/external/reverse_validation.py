"""Reverse validation: train on the PUBLIC cohort, test on the INSTITUTIONAL one.

All prior analyses ran in one direction (institutional -> SCIN). Reversing it asks
whether the domain gap is symmetric, which distinguishes three explanations for the
observed drop:

  both directions drop        -> mutual distribution gap; the datasets disagree, and the
                                 failure is not specific to our model or training set
  SCIN->ours transfers better -> asymmetry; SCIN is the more heterogeneous cohort and
                                 narrow training data is the limiting factor
  SCIN->ours transfers well   -> our model or training set is the weak link

Our cohort is also the better-powered test set: 406 psoriasis cases versus SCIN's 113,
and 308 Fitzpatrick V versus 92 (though it contains no Fitzpatrick I-II).

Forward reference (institutional -> SCIN): image 0.632, metadata 0.451.

    python experiments/external/reverse_validation.py [--epochs 15]
"""

from __future__ import annotations

import argparse
import csv
import random

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from dermafair.paths import DATA_DIR, output_dir

REPORT_DIR = output_dir("external")
from external_validate_fusion import canon_vec  # noqa: E402

from dermafair.data.folder_split import _build_transforms  # noqa: E402
from dermafair.models import build_image_model  # noqa: E402
from dermafair.models.trainer import TrainConfig, train_model  # noqa: E402


class DS(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, meta, lab, fst = self.items[i]
        return {
            "image": self.tfm(Image.open(p).convert("RGB")),
            "meta": torch.tensor(meta, dtype=torch.float32),
            "label": torch.tensor(lab, dtype=torch.long),
            "fitzpatrick": int(fst),
        }


def scin_items():
    idx = {}
    for f in (DATA_DIR / "scin_images").iterdir():
        if f.is_file():
            idx.setdefault(f.name.split("__")[0], f)
    return [
        (idx[r["case_id"]], canon_vec(r), int(r["label"]), int(r["fitzpatrick"]))
        for r in csv.DictReader(open(DATA_DIR / "external_scin_manifest.csv", encoding="utf-8"))
        if r["case_id"] in idx
    ]


def institutional_items():
    """ALL institutional images - none were used to train the SCIN model."""
    loc = {}
    for d in ("DATASET_0", "DATASET_1"):
        for f in (DATA_DIR / d).iterdir():
            if f.is_file():
                loc[f.name] = f
    out = []
    for r in csv.DictReader(open(DATA_DIR / "internal_canonical_manifest.csv", encoding="utf-8")):
        f = loc.get(r["image_name"])
        if f is not None:
            out.append((f, canon_vec(r), int(r["label"]), int(r["fitzpatrick"])))
    return out


def boot(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


@torch.no_grad()
def predict(model, loader, dev):
    model.to(dev).eval()
    P, Y, F = [], [], []
    for b in loader:
        P.append(torch.softmax(model(b["image"].to(dev)), 1)[:, 1].cpu().numpy())
        Y.append(b["label"].numpy())
        F.append(np.asarray(b["fitzpatrick"]))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    scin, inst = scin_items(), institutional_items()
    ys = np.array([i[2] for i in scin])
    print(f"SCIN (train) {len(scin)}  psoriasis {ys.sum()} ({ys.mean():.1%})")
    print(f"institutional (test) {len(inst)}  psoriasis {sum(i[2] for i in inst)}")

    # hold out a validation slice of SCIN for early stopping
    rng = random.Random(42)
    order = list(range(len(scin)))
    rng.shuffle(order)
    cut = int(0.85 * len(order))
    tr_i = [scin[i] for i in order[:cut]]
    va_i = [scin[i] for i in order[cut:]]
    ttf, etf = _build_transforms(224, augment=True), _build_transforms(224, augment=False)
    tr = DataLoader(DS(tr_i, ttf), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(DS(va_i, etf), batch_size=args.batch_size)
    te = DataLoader(DS(inst, etf), batch_size=args.batch_size)

    labs = np.array([i[2] for i in tr_i])
    cw = torch.tensor(
        [len(labs) / (2 * max((labs == 0).sum(), 1)), len(labs) / (2 * max((labs == 1).sum(), 1))], dtype=torch.float32
    )
    print(f"class weights (eczema, psoriasis): {cw.tolist()}")

    print("\n=== training image model on SCIN ===", flush=True)
    m = build_image_model("resnet50", num_classes=2, pretrained=True)
    train_model(
        m,
        tr,
        va,
        TrainConfig(epochs=args.epochs, lr=1e-4, patience=args.patience, device=dev, modality="image"),
        loss_weights=cw,
    )
    p_img, y, f = predict(m, te, dev)
    a_img = roc_auc_score(y, p_img)
    lo, hi = boot(y, p_img)

    # metadata model, canonical space, same protocol as the forward direction
    Xtr = np.array([i[1] for i in tr_i])
    ytr = np.array([i[2] for i in tr_i])
    Xte = np.array([i[1] for i in inst])
    yte = np.array([i[2] for i in inst])
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    p_meta = lr.predict_proba(Xte)[:, 1]
    a_meta = roc_auc_score(yte, p_meta)
    mlo, mhi = boot(yte, p_meta)

    L4, D = (f >= 3) & (f <= 4), f >= 5  # our cohort spans FST III-VI only
    L = [
        "# Reverse validation: trained on SCIN, tested on the institutional cohort\n",
        "The forward direction (institutional -> SCIN) gave image 0.632 and metadata 0.451. Reversing it",
        "tests whether the domain gap is symmetric. Note the institutional cohort is the better-powered",
        f"test set ({int(yte.sum())} psoriasis vs SCIN's 113) but contains no Fitzpatrick I-II.\n",
        "| Direction | Image AUROC | 95% CI | Metadata AUROC | 95% CI |",
        "|---|---|---|---|---|",
        "| institutional -> SCIN (forward) | 0.632 | 0.575-0.692 | 0.451 | 0.389-0.515 |",
        f"| **SCIN -> institutional (reverse)** | **{a_img:.3f}** | {lo:.3f}-{hi:.3f} | "
        f"**{a_meta:.3f}** | {mlo:.3f}-{mhi:.3f} |",
        "\n- within-domain reference (institutional -> institutional, 5-fold CV): image 0.779, metadata 0.699",
        "\n## Skin tone, reverse direction (our cohort spans FST III-VI)\n",
        "| Group | n | psoriasis | Image AUROC |",
        "|---|---|---|---|",
    ]
    for nm, msk in [("FST III-IV", L4), ("FST V-VI", D)]:
        if msk.sum() and len(np.unique(y[msk])) > 1:
            L.append(f"| {nm} | {int(msk.sum())} | {int(y[msk].sum())} | {roc_auc_score(y[msk], p_img[msk]):.3f} |")
    L += [
        "\n## Interpretation\n",
        "Comparable degradation in both directions indicates a mutual distribution gap: the two cohorts",
        "disagree about the mapping from image to diagnosis, and the failure is not peculiar to our model",
        "or training set. Markedly better transfer in the reverse direction would instead indicate that",
        "SCIN is the more heterogeneous cohort and that training on narrow institutional data is the",
        "limiting factor. Both readings bear on whether the remedy is broader training data or better",
        "modelling.\n",
        "**Caveats.** SCIN carries a steep class imbalance (113 psoriasis of 1,128), so the reverse model",
        "trains on far fewer positive examples than the forward model did; some degradation is therefore",
        "attributable to training-set composition rather than domain shift alone. SCIN labels are also",
        "dermatologist differentials rather than biopsy-confirmed.",
    ]
    (REPORT_DIR / "REVERSE_VALIDATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    np.savez(REPORT_DIR / "reverse_validation_predictions.npz", y=y, p_img=p_img, p_meta=p_meta, fst=f)
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
