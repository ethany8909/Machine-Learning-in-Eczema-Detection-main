"""Leave-one-source-out (internal-external) validation.

Asks whether a shift-aware evaluation available WITHIN a single cohort anticipates
the failures we observed against a genuinely external cohort. Our two acquisition
sources differ in class balance, skin-tone distribution and capture channel, and
share no subjects, so training on one and testing on the other simulates deployment
shift without new data.

Three settings are run:
  D0 -> D1        train on source 0, test on source 1
  D1 -> D0        the reverse
  size-matched    train on a RANDOM subject-level subset of the pooled data of the
                  same size, test on a random held-out set - so any drop unique to
                  the LOSO settings is attributable to domain shift rather than to
                  the smaller training set.

If LOSO reproduces the external pattern (metadata collapsing further than images, a
tone gap emerging), it offers a cheap internal proxy for external validation.

    python experiments/external/leave_one_source_out.py [--epochs 12]
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from dermafair.paths import DATA_DIR, output_dir

REPORT_DIR = output_dir("external")

from external_fusion_sweep import auc  # noqa: E402
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
        p, meta, lab, fst, _subj = self.items[i]
        return {
            "image": self.tfm(Image.open(p).convert("RGB")),
            "meta": torch.tensor(meta, dtype=torch.float32),
            "label": torch.tensor(lab, dtype=torch.long),
            "fitzpatrick": int(fst),
        }


def load_items():
    loc = {}
    for d in ("DATASET_0", "DATASET_1"):
        for f in (DATA_DIR / d).iterdir():
            if f.is_file():
                loc[f.name] = (d, f)
    meta = {
        r["image_name"]: canon_vec(r)
        for r in csv.DictReader(open(DATA_DIR / "internal_canonical_manifest.csv", encoding="utf-8"))
    }
    by_src = defaultdict(list)
    for r in csv.DictReader(open(DATA_DIR / "manifest_clean.csv", encoding="utf-8-sig")):
        n = r["image_name"]
        if n in loc and n in meta:
            src, path = loc[n]
            by_src[src].append((path, meta[n], int(r["label"]), int(r["fitzpatrick"]), r["subject"]))
    return by_src


def subject_split(items, frac, seed=42):
    subs = sorted({i[4] for i in items})
    random.Random(seed).shuffle(subs)
    cut = set(subs[: int(len(subs) * frac)])
    return [i for i in items if i[4] in cut], [i for i in items if i[4] not in cut]


def run(train_items, test_items, tag, args, dev):
    ttf, etf = _build_transforms(224, augment=True), _build_transforms(224, augment=False)
    tr_i, va_i = subject_split(train_items, 0.85)
    tr = DataLoader(DS(tr_i, ttf), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(DS(va_i, etf), batch_size=args.batch_size)
    te = DataLoader(DS(test_items, etf), batch_size=args.batch_size)
    print(f"\n=== {tag}: train {len(tr_i)} val {len(va_i)} test {len(test_items)} ===", flush=True)

    labs = np.array([i[2] for i in tr_i])
    cw = torch.tensor(
        [len(labs) / (2 * max((labs == 0).sum(), 1)), len(labs) / (2 * max((labs == 1).sum(), 1))], dtype=torch.float32
    )

    # image model
    m = build_image_model("resnet50", num_classes=2, pretrained=True)
    train_model(
        m,
        tr,
        va,
        TrainConfig(epochs=args.epochs, lr=1e-4, patience=args.patience, device=dev, modality="image"),
        loss_weights=cw,
    )
    m.to(dev).eval()
    ps, ys, fs = [], [], []
    with torch.no_grad():
        for b in te:
            ps.append(torch.softmax(m(b["image"].to(dev)), 1)[:, 1].cpu().numpy())
            ys.append(b["label"].numpy())
            fs.append(np.asarray(b["fitzpatrick"]))
    p_img, y, f = np.concatenate(ps), np.concatenate(ys), np.concatenate(fs)

    # metadata model (canonical, logistic — matches the external protocol)
    Xtr = np.array([i[1] for i in tr_i])
    ytr = np.array([i[2] for i in tr_i])
    Xte = np.array([i[1] for i in test_items])
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    p_meta = lr.predict_proba(Xte)[:, 1]

    L, D = (f >= 1) & (f <= 4), f >= 5
    return {
        "tag": tag,
        "n_train": len(tr_i),
        "n_test": len(test_items),
        "img": auc(y, p_img),
        "meta": auc(y, p_meta),
        "img_light": auc(y[L], p_img[L]) if L.sum() else np.nan,
        "img_dark": auc(y[D], p_img[D]) if D.sum() else np.nan,
        "n_dark": int(D.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    by_src = load_items()
    d0, d1 = by_src["DATASET_0"], by_src["DATASET_1"]
    print(f"DATASET_0 n={len(d0)}  DATASET_1 n={len(d1)}  device={dev}")

    res = [run(d0, d1, "LOSO: D0 -> D1", args, dev), run(d1, d0, "LOSO: D1 -> D0", args, dev)]

    # size-matched control: pooled random subject split of comparable size
    pooled = d0 + d1
    ctrl_tr, rest = subject_split(pooled, len(d0) / len(pooled), seed=7)
    res.append(run(ctrl_tr, rest, "size-matched control (pooled random)", args, dev))

    out = [
        "# Leave-one-source-out (internal-external) validation\n",
        "Does a shift-aware evaluation available WITHIN our cohort anticipate the external failures?\n",
        "| Setting | n train | n test | image AUROC | metadata AUROC | img light (I-IV) | img dark (V-VI) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in res:
        out.append(
            f"| {r['tag']} | {r['n_train']} | {r['n_test']} | **{r['img']:.3f}** | **{r['meta']:.3f}** | "
            f"{r['img_light']:.3f} | {r['img_dark']:.3f} (n={r['n_dark']}) |"
        )
    out += [
        "| *reference: internal random split* | 802 | 146 | *0.792* | *0.699* | - | - |",
        "| *reference: external SCIN* | 802 | 1128 | *0.632* | *0.451* | *0.711* | *0.503* |",
        "\n## Reading this\n",
        "The size-matched control holds training-set size constant while removing domain shift, so the",
        "difference between it and the two LOSO settings isolates the effect of shift. If the LOSO",
        "settings reproduce the external pattern - metadata degrading further than images, and a",
        "light/dark gap opening - then leave-one-source-out offers a cheap internal proxy for external",
        "validation, usable by groups without access to a second cohort.",
    ]
    (REPORT_DIR / "LEAVE_ONE_SOURCE_OUT.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
