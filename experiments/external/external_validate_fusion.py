"""External validation of the FUSION models — the decisive test of multimodal viability.

The fusion models cannot be transferred directly: they were trained on fine-grained
metadata SCIN cannot supply. Both branches are therefore retrained in the shared
canonical metadata space (18 features), with the image backbone frozen at its
internally trained weights, and then evaluated on SCIN.

Central hypothesis: metadata collapses externally (0.451, chance) while images hold
(0.632). Fixed late fusion is locked at 0.5/0.5 and should therefore be dragged
toward the failing modality, whereas the gate network can learn per-sample weights
and may down-weight metadata, preserving image-level performance. If so, adaptive
fusion is ROBUST TO MODALITY FAILURE under distribution shift - a concrete argument
for multimodal architectures that fixed fusion cannot make.

    python experiments/external/external_validate_fusion.py [--epochs 20]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from dermafair.paths import DATA_DIR, RESULTS_DIR, output_dir

REPORT_DIR = output_dir("external")

from dermafair.data.folder_split import _build_transforms, _load_fitzpatrick_map  # noqa: E402
from dermafair.models import build_fusion, build_image_model, build_metadata_model  # noqa: E402
from dermafair.models.trainer import TrainConfig, train_model  # noqa: E402

SPLIT = DATA_DIR / "dataset_split_clean"
CKPT = RESULTS_DIR / "image_models_clean/checkpoints/resnet50.pt"
AGES = ["under_20", "20_39", "40_59", "60_plus", "unknown"]
SEXES = ["male", "female", "unknown"]
SITES = ["head_neck", "arm", "hand", "torso_front", "torso_back", "genital_groin", "buttocks", "leg", "foot", "other"]
NF = len(AGES) + len(SEXES) + len(SITES)


def canon_vec(row):
    v = [0.0] * NF
    v[AGES.index(row["age"] if row["age"] in AGES else "unknown")] = 1.0
    v[len(AGES) + SEXES.index(row["sex"] if row["sex"] in SEXES else "unknown")] = 1.0
    for s in row["sites"].split("|") if row["sites"] else []:
        if s in SITES:
            v[len(AGES) + len(SEXES) + SITES.index(s)] = 1.0
    return v


class MM(Dataset):
    """image + canonical metadata + label + fitzpatrick."""

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


def internal_items(split, meta_by_name, fz, tfm_dummy=None):
    items = []
    for cls, lab in (("eczema_dermatitis", 0), ("psoriasis_lichenoid", 1)):
        d = SPLIT / split / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name in meta_by_name:
                items.append((f, meta_by_name[f.name], lab, fz.get(f.name, -1)))
    return items


@torch.no_grad()
def predict(model, loader, dev, want_gate=False):
    model.to(dev).eval()
    ps, ys, fs, gw = [], [], [], []
    for b in loader:
        out = model(b["image"].to(dev), b["meta"].to(dev))
        ps.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        ys.append(b["label"].numpy())
        fs.append(np.asarray(b["fitzpatrick"]))
        if want_gate and getattr(model, "last_gate_weights", None) is not None:
            gw.append(model.last_gate_weights.cpu().numpy())
    return (np.concatenate(ps), np.concatenate(ys), np.concatenate(fs), np.concatenate(gw) if gw else None)


def boot(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gate-entropy", type=float, default=0.1)
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

    # external SCIN
    idx = {}
    for f in (DATA_DIR / "scin_images").iterdir():
        if f.is_file():
            idx.setdefault(f.name.split("__")[0], f)
    ext = []
    for r in csv.DictReader(open(DATA_DIR / "external_scin_manifest.csv", encoding="utf-8")):
        f = idx.get(r["case_id"])
        if f is not None:
            ext.append((f, canon_vec(r), int(r["label"]), int(r["fitzpatrick"])))
    ex = DataLoader(MM(ext, etf), batch_size=args.batch_size)
    print(f"internal train/val/test = {len(tr.dataset)}/{len(va.dataset)}/{len(te.dataset)} | external {len(ext)}")

    labs = np.array([i[2] for i in internal_items("train", meta_by_name, fz)])
    cw = torch.tensor([len(labs) / (2 * (labs == 0).sum()), len(labs) / (2 * (labs == 1).sum())], dtype=torch.float32)

    # warm-start metadata branch (canonical space)
    meta_seed = build_metadata_model("mlp", NF, num_classes=2)
    train_model(
        meta_seed,
        tr,
        va,
        TrainConfig(epochs=args.epochs, lr=1e-3, patience=6, device=dev, modality="metadata"),
        loss_weights=cw,
    )

    rows = []
    for strategy in ("late_fusion", "gate_network"):
        print(f"\n=== {strategy} (canonical space) ===", flush=True)
        img = build_image_model("resnet50", num_classes=2, pretrained=True)
        img.load_state_dict(torch.load(CKPT, map_location=dev))
        mm = build_metadata_model("mlp", NF, num_classes=2)
        mm.load_state_dict(meta_seed.state_dict())
        kw = (
            {"hidden_dim": 128, "freeze_image_backbone": True, "normalize_features": True}
            if strategy == "gate_network"
            else {}
        )
        fus = build_fusion(strategy, img, mm, num_classes=2, **kw)
        for p in fus.image_model.parameters():
            p.requires_grad = False
        aux = None
        if strategy == "gate_network":
            fus.warm_start_heads()
            aux = lambda m, _l=args.gate_entropy: _l * m.entropy_penalty()
        cfg = TrainConfig(epochs=args.epochs, lr=1e-3, patience=6, device=dev, modality="multimodal")
        train_model(fus, tr, va, cfg, loss_weights=cw, aux_loss_fn=aux)

        pi, yi, _, gi = predict(fus, te, dev, want_gate=True)
        pe, ye, fe, ge = predict(fus, ex, dev, want_gate=True)
        lo, hi = boot(ye, pe)
        L4, D = (fe >= 1) & (fe <= 4), fe >= 5
        rows.append(
            {
                "name": strategy,
                "int": roc_auc_score(yi, pi),
                "ext": roc_auc_score(ye, pe),
                "ci": (lo, hi),
                "light": roc_auc_score(ye[L4], pe[L4]) if len(np.unique(ye[L4])) > 1 else np.nan,
                "dark": roc_auc_score(ye[D], pe[D]) if len(np.unique(ye[D])) > 1 else np.nan,
                "gmi": float(gi[:, 1].mean()) if gi is not None else None,
                "gme": float(ge[:, 1].mean()) if ge is not None else None,
            }
        )
        print(f"  internal test {rows[-1]['int']:.3f} | external {rows[-1]['ext']:.3f}", flush=True)

    L = [
        "# External validation of fusion models (canonical metadata space)\n",
        "Both branches retrained in the shared canonical space; image backbone frozen at its internally",
        "trained weights. Reference points on the same external cohort: **metadata-only 0.451**",
        "(chance) and **image-only 0.632**.\n",
        "| Model | internal test | external | 95% CI | ext. light (I-IV) | ext. dark (V-VI) |",
        "|---|---|---|---|---|---|",
        "| metadata-only (canonical) | 0.699 (CV) | 0.451 | 0.389-0.515 | 0.421 | 0.392 |",
        "| image-only (ensemble) | 0.779 (CV) | 0.632 | 0.575-0.692 | 0.711 | 0.503 |",
    ]
    for r in rows:
        L.append(
            f"| {r['name']} | {r['int']:.3f} | **{r['ext']:.3f}** | {r['ci'][0]:.3f}-{r['ci'][1]:.3f} | "
            f"{r['light']:.3f} | {r['dark']:.3f} |"
        )
    g = next((r for r in rows if r["name"] == "gate_network"), None)
    if g and g["gmi"] is not None:
        L += [
            f"\n- Gate mean METADATA weight: internal {g['gmi']:.3f} -> external {g['gme']:.3f} "
            f"({'down-weights' if g['gme'] < g['gmi'] else 'does not down-weight'} metadata externally)"
        ]
    lf = next((r for r in rows if r["name"] == "late_fusion"), None)
    if lf and g:
        L += [
            "\n## Interpretation\n",
            f"Late fusion reached {lf['ext']:.3f} externally and the gate {g['ext']:.3f}, against 0.632 for",
            "images alone and 0.451 for metadata alone. If fixed fusion falls below the image-only figure",
            "while the gate does not, adaptive weighting confers robustness to modality failure under",
            "distribution shift; if both fall, fusion inherits the weakest modality's failure regardless of",
            "how the modalities are combined.",
        ]
    (REPORT_DIR / "EXTERNAL_VALIDATION_FUSION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
