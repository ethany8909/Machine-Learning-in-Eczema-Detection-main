"""External fusion analysis with a late-fusion WEIGHT SWEEP and paired comparisons.

Improves on the first fusion run in three ways:
  1. saves every prediction, enabling PAIRED bootstrap comparisons (the models are
     evaluated on identical cases, so paired tests are far more powerful);
  2. sweeps the fixed late-fusion weight - including the 0.75 image / 0.25 metadata
     configuration - rather than assuming 0.5/0.5;
  3. reports the sweep as a CURVE.

Method note: late fusion is a fixed linear combination of two independently trained
branches' logits, so the sweep is computed analytically from one set of image logits
and one set of metadata logits - no retraining per weight.

IMPORTANT CAVEAT recorded in the output: choosing the best weight by looking at the
external result would constitute test-set selection. The curve is reported
descriptively, to characterise how robustness varies with weighting, not to pick a
deployment value.

    python experiments/external/external_fusion_sweep.py
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dermafair.paths import DATA_DIR, output_dir

REPORT_DIR = output_dir("external")

from external_validate_fusion import CKPT, MM, NF, canon_vec, internal_items  # noqa: E402

from dermafair.data.folder_split import _build_transforms, _load_fitzpatrick_map  # noqa: E402
from dermafair.models import build_fusion, build_image_model, build_metadata_model  # noqa: E402
from dermafair.models.trainer import TrainConfig, train_model  # noqa: E402

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]  # w_image; 1.0 == image-only


@torch.no_grad()
def logits_for(model, loader, dev, modality):
    model.to(dev).eval()
    L, Y, F = [], [], []
    for b in loader:
        if modality == "image":
            o = model(b["image"].to(dev))
        elif modality == "metadata":
            o = model(b["meta"].to(dev))
        else:
            o = model(b["image"].to(dev), b["meta"].to(dev))
        L.append(o.cpu().numpy())
        Y.append(b["label"].numpy())
        F.append(np.asarray(b["fitzpatrick"]))
    return np.concatenate(L), np.concatenate(Y), np.concatenate(F)


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def paired_diff(y, sa, sb, n=4000, seed=42):
    """Paired bootstrap of AUROC(a) - AUROC(b) on identical cases."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            d.append(auc(y[i], sa[i]) - auc(y[i], sb[i]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta_by_name = {
        r["image_name"]: canon_vec(r)
        for r in csv.DictReader(open(DATA_DIR / "internal_canonical_manifest.csv", encoding="utf-8"))
    }
    fz = _load_fitzpatrick_map(DATA_DIR / "Skin_Metadata-1.csv")
    ttf, etf = _build_transforms(224, augment=True), _build_transforms(224, augment=False)
    tr = DataLoader(MM(internal_items("train", meta_by_name, fz), ttf), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(MM(internal_items("val", meta_by_name, fz), etf), batch_size=args.batch_size)
    te = DataLoader(MM(internal_items("test", meta_by_name, fz), etf), batch_size=args.batch_size)

    idx = {}
    for f in (DATA_DIR / "scin_images").iterdir():
        if f.is_file():
            idx.setdefault(f.name.split("__")[0], f)
    ext = [
        (idx[r["case_id"]], canon_vec(r), int(r["label"]), int(r["fitzpatrick"]))
        for r in csv.DictReader(open(DATA_DIR / "external_scin_manifest.csv", encoding="utf-8"))
        if r["case_id"] in idx
    ]
    ex = DataLoader(MM(ext, etf), batch_size=args.batch_size)

    labs = np.array([i[2] for i in internal_items("train", meta_by_name, fz)])
    cw = torch.tensor([len(labs) / (2 * (labs == 0).sum()), len(labs) / (2 * (labs == 1).sum())], dtype=torch.float32)

    # ---- branches ----
    print("training metadata branch (canonical)...", flush=True)
    meta = build_metadata_model("mlp", NF, num_classes=2)
    train_model(
        meta,
        tr,
        va,
        TrainConfig(epochs=args.epochs, lr=1e-3, patience=6, device=dev, modality="metadata"),
        loss_weights=cw,
    )
    img = build_image_model("resnet50", num_classes=2, pretrained=True)
    img.load_state_dict(torch.load(CKPT, map_location=dev))

    print("computing logits...", flush=True)
    Li_int, y_int, f_int = logits_for(img, te, dev, "image")
    Lm_int, _, _ = logits_for(meta, te, dev, "metadata")
    Li_ext, y_ext, f_ext = logits_for(img, ex, dev, "image")
    Lm_ext, _, _ = logits_for(meta, ex, dev, "metadata")

    def score(Li, Lm, w):
        z = w * Li + (1 - w) * Lm
        return z[:, 1] - z[:, 0]

    # ---- gate (requires training) ----
    print("training gate network...", flush=True)
    g_img = build_image_model("resnet50", num_classes=2, pretrained=True)
    g_img.load_state_dict(torch.load(CKPT, map_location=dev))
    g_meta = build_metadata_model("mlp", NF, num_classes=2)
    g_meta.load_state_dict(meta.state_dict())
    gate = build_fusion(
        "gate_network",
        g_img,
        g_meta,
        num_classes=2,
        hidden_dim=128,
        freeze_image_backbone=True,
        normalize_features=True,
    )
    for p in gate.image_model.parameters():
        p.requires_grad = False
    gate.warm_start_heads()
    train_model(
        gate,
        tr,
        va,
        TrainConfig(epochs=args.epochs, lr=1e-3, patience=6, device=dev, modality="multimodal"),
        loss_weights=cw,
        aux_loss_fn=lambda m: 0.1 * m.entropy_penalty(),
    )
    Lg_int, _, _ = logits_for(gate, te, dev, "multimodal")
    Lg_ext, _, _ = logits_for(gate, ex, dev, "multimodal")
    s_gate_int, s_gate_ext = Lg_int[:, 1] - Lg_int[:, 0], Lg_ext[:, 1] - Lg_ext[:, 0]

    # ---- report ----
    L4, D = (f_ext >= 1) & (f_ext <= 4), f_ext >= 5
    out = [
        "# Late-fusion weight sweep and paired external comparison\n",
        "Late fusion combines two independently trained branches' logits with fixed weights, so the",
        "sweep is computed analytically. w=1.0 is image-only; w=0.0 is metadata-only.\n",
        "> **Caveat:** selecting a weight by inspecting the external column would be test-set selection.",
        "> The curve is descriptive - it characterises how robustness varies with weighting.\n",
        "| w_image | w_metadata | internal test | external | ext. light (I-IV) | ext. dark (V-VI) |",
        "|---|---|---|---|---|---|",
    ]
    ext_scores = {}
    for w in WEIGHTS:
        si, se = score(Li_int, Lm_int, w), score(Li_ext, Lm_ext, w)
        ext_scores[w] = se
        tag = (
            " (image-only)"
            if w == 1.0
            else (" (metadata-only)" if w == 0.0 else (" **(0.75/0.25)**" if w == 0.75 else ""))
        )
        out.append(
            f"| {w:.2f}{tag} | {1 - w:.2f} | {auc(y_int, si):.3f} | **{auc(y_ext, se):.3f}** | "
            f"{auc(y_ext[L4], se[L4]):.3f} | {auc(y_ext[D], se[D]):.3f} |"
        )
    out.append(
        f"| gate (learned) | adaptive | {auc(y_int, s_gate_int):.3f} | **{auc(y_ext, s_gate_ext):.3f}** | "
        f"{auc(y_ext[L4], s_gate_ext[L4]):.3f} | {auc(y_ext[D], s_gate_ext[D]):.3f} |"
    )

    out += [
        "\n## Paired comparisons on the external cohort (vs image-only)\n",
        "| Comparison | mean difference | 95% CI |",
        "|---|---|---|",
    ]
    base = ext_scores[1.0]
    for w in [0.5, 0.75]:
        m, lo, hi = paired_diff(y_ext, ext_scores[w], base)
        out.append(f"| late fusion w={w:.2f} - image-only | {m:+.4f} | {lo:+.4f} to {hi:+.4f} |")
    m, lo, hi = paired_diff(y_ext, s_gate_ext, base)
    out.append(f"| gate - image-only | {m:+.4f} | {lo:+.4f} to {hi:+.4f} |")
    m, lo, hi = paired_diff(y_ext, s_gate_ext, ext_scores[0.5])
    out.append(f"| gate - late fusion (0.5) | {m:+.4f} | {lo:+.4f} to {hi:+.4f} |")

    np.savez(
        REPORT_DIR / "external_fusion_sweep_predictions.npz",
        y_ext=y_ext,
        f_ext=f_ext,
        y_int=y_int,
        gate_ext=s_gate_ext,
        gate_int=s_gate_int,
        **{f"w{int(w * 100)}_ext": ext_scores[w] for w in WEIGHTS},
    )
    (REPORT_DIR / "EXTERNAL_FUSION_SWEEP.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
