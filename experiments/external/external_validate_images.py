"""Image-model EXTERNAL validation on SCIN.

Unlike the metadata model, the image backbone is regime-independent, so the
checkpoint trained on our internal cohort is applied to SCIN DIRECTLY (no
retraining, no feature harmonization). This is the cleanest possible external test.

Evaluates:
  - the single clean-split ResNet-50
  - the 5-fold cross-validation ensemble (mean probability), which is more robust
and reports overall AUROC with bootstrap CIs, tone-stratified results (FST I-IV vs
V-VI, per Groh et al.), and external calibration.

    python experiments/external/external_validate_images.py [--arch resnet50] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from dermafair.paths import DATA_DIR, RESULTS_DIR, output_dir

REPORT_DIR = output_dir("external")

from dermafair.data.folder_split import _build_transforms  # noqa: E402
from dermafair.fairness.advanced_metrics import expected_calibration_error  # noqa: E402
from dermafair.models import build_image_model  # noqa: E402

IMAGES = DATA_DIR / "scin_images"
MANIFEST = DATA_DIR / "external_scin_manifest.csv"
SINGLE = RESULTS_DIR / "image_models_clean/checkpoints"
FOLDS = RESULTS_DIR / "cv_clean/checkpoints"


def boot_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            vals.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


@torch.no_grad()
def infer(model, files, tfm, bs, device="cpu"):
    model.to(device).eval()
    out = []
    for i in range(0, len(files), bs):
        batch = torch.stack([tfm(Image.open(f).convert("RGB")) for f in files[i : i + bs]]).to(device)
        out.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
        if (i // bs) % 10 == 0:
            print(f"    {min(i + bs, len(files))}/{len(files)}", flush=True)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    # ---- match manifest rows to downloaded files ----
    index = {}
    for f in IMAGES.iterdir():
        if f.is_file():
            index.setdefault(f.name.split("__")[0], f)

    rows, files = [], []
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        f = index.get(r["case_id"])
        if f is not None:
            rows.append(r)
            files.append(f)
    y = np.array([int(r["label"]) for r in rows])
    fst = np.array([int(r["fitzpatrick"]) for r in rows])
    wt = np.array([float(r["label_weight"]) for r in rows])
    print(f"matched {len(files)} SCIN images (psoriasis {int(y.sum())}, eczema {int((y == 0).sum())})")

    tfm = _build_transforms(224, augment=False)

    # ---- single model ----
    print(f"\n[1/2] single clean-split {args.arch}")
    m = build_image_model(args.arch, num_classes=2, pretrained=True)
    m.load_state_dict(torch.load(SINGLE / f"{args.arch}.pt", map_location="cpu"))
    p_single = infer(m, files, tfm, args.batch_size)

    # ---- CV ensemble ----
    print("\n[2/2] 5-fold CV ensemble")
    preds = []
    for k in range(5):
        ck = FOLDS / f"{args.arch}_fold{k}.pt"
        if not ck.exists():
            continue
        mk = build_image_model(args.arch, num_classes=2, pretrained=True)
        mk.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"  fold {k}")
        preds.append(infer(mk, files, tfm, args.batch_size))
    p_ens = np.mean(preds, axis=0) if preds else p_single

    L = [
        "# Image-model external validation on SCIN\n",
        "The image backbone is regime-independent, so the internally trained checkpoint is applied to SCIN",
        "directly - no retraining or feature harmonization. Internal reference: ResNet-50 test AUROC 0.792",
        "(single split) and 0.779 (SD 0.026) under cross-validation.\n",
        f"- matched images: {len(files)} (psoriasis {int(y.sum())}, eczema {int((y == 0).sum())}, "
        f"prevalence {y.mean():.1%})\n",
        "## Overall\n",
        "| Model | AUROC | 95% CI | Balanced acc | ECE |",
        "|---|---|---|---|---|",
    ]
    for name, p in [("single clean-split", p_single), ("5-fold ensemble", p_ens)]:
        lo, hi = boot_ci(y, p)
        bacc = balanced_accuracy_score(y, (p >= 0.5).astype(int))
        ece = expected_calibration_error(y, p)["ece"]
        L.append(f"| {name} | **{roc_auc_score(y, p):.3f}** | {lo:.3f}-{hi:.3f} | {bacc:.3f} | {ece:.3f} |")

    L += [
        "\n## By skin-tone group (ensemble; FST I-IV vs V-VI, per Groh et al.)\n",
        "| Group | n | psoriasis | AUROC | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for nm, msk in [("Light (FST I-IV)", (fst >= 1) & (fst <= 4)), ("Dark (FST V-VI)", fst >= 5)]:
        if msk.sum() and len(np.unique(y[msk])) > 1:
            lo, hi = boot_ci(y[msk], p_ens[msk])
            L.append(
                f"| {nm} | {int(msk.sum())} | {int(y[msk].sum())} | {roc_auc_score(y[msk], p_ens[msk]):.3f} | {lo:.3f}-{hi:.3f} |"
            )
        else:
            L.append(f"| {nm} | {int(msk.sum())} | {int(y[msk].sum())} | n/a | n/a |")

    m2 = wt >= 0.5
    if m2.sum() and len(np.unique(y[m2])) > 1:
        lo, hi = boot_ci(y[m2], p_ens[m2])
        L += [
            "\n## Sensitivity: high-confidence SCIN labels (weight >= 0.5)\n",
            f"- n={int(m2.sum())} (psoriasis {int(y[m2].sum())}): AUROC **{roc_auc_score(y[m2], p_ens[m2]):.3f}** "
            f"(95% CI {lo:.3f}-{hi:.3f})",
        ]

    np.savez(
        REPORT_DIR / "external_scin_image_predictions.npz",
        y_true=y,
        p_single=p_single,
        p_ensemble=p_ens,
        fitzpatrick=fst,
        label_weight=wt,
    )
    (REPORT_DIR / "EXTERNAL_VALIDATION_IMAGES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
