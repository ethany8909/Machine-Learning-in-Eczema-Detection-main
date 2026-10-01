"""Can the gate be made to generalize rather than fit one population?

Two targeted interventions, run as an ablation:

  A  feature-gate, no dropout        - reproduces the current design (ext. ~0.617)
  B  feature-gate + modality dropout - metadata randomly withheld during training, so
                                       the gate cannot become structurally dependent on it
  C  confidence-gate + dropout       - the gate is conditioned on BRANCH RELIABILITY
                                       (predictive entropy, max probability, margin,
                                       disagreement) instead of raw features. Raw features
                                       shift across cohorts; uncertainty does not, so a
                                       rule learned over confidence should transfer.

The frozen ResNet-50 backbone's features and logits are precomputed once, so the image
branch contributes exactly the image-only model's logits and each variant trains in
seconds. Every variant is evaluated on SCIN under the identical external protocol, with
paired bootstrap comparisons against image-only.

    python experiments/external/gate_robust_experiment.py
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dermafair.paths import DATA_DIR, output_dir

REPORT_DIR = output_dir("external")

from external_validate_fusion import CKPT, MM, NF, canon_vec, internal_items  # noqa: E402

from dermafair.data.folder_split import _build_transforms, _load_fitzpatrick_map  # noqa: E402
from dermafair.models import build_image_model  # noqa: E402

CACHE = REPORT_DIR / "gate_feature_cache.npz"


# --------------------------------------------------------------------------- #
# Feature precomputation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode(model, loader, dev):
    model.to(dev).eval()
    Fe, Lo, Y, Fz, Me = [], [], [], [], []
    for b in loader:
        x = b["image"].to(dev)
        feats = model.features(x)
        Fe.append(feats.cpu().numpy())
        Lo.append(model.classifier(feats).cpu().numpy())
        Y.append(b["label"].numpy())
        Fz.append(np.asarray(b["fitzpatrick"]))
        Me.append(b["meta"].numpy())
    return (np.concatenate(Fe), np.concatenate(Lo), np.concatenate(Y), np.concatenate(Fz), np.concatenate(Me))


def build_cache(dev, bs=32):
    if CACHE.exists():
        d = np.load(CACHE)
        return {
            k: {n: d[f"{k}_{n}"] for n in ("feat", "logit", "y", "fst", "meta")}
            for k in ("train", "val", "test", "ext")
        }
    meta_by_name = {
        r["image_name"]: canon_vec(r)
        for r in csv.DictReader(open(DATA_DIR / "internal_canonical_manifest.csv", encoding="utf-8"))
    }
    fz = _load_fitzpatrick_map(DATA_DIR / "Skin_Metadata-1.csv")
    etf = _build_transforms(224, augment=False)
    sets = {
        s: DataLoader(MM(internal_items(s, meta_by_name, fz), etf), batch_size=bs) for s in ("train", "val", "test")
    }
    idx = {}
    for f in (DATA_DIR / "scin_images").iterdir():
        if f.is_file():
            idx.setdefault(f.name.split("__")[0], f)
    ext_items = [
        (idx[r["case_id"]], canon_vec(r), int(r["label"]), int(r["fitzpatrick"]))
        for r in csv.DictReader(open(DATA_DIR / "external_scin_manifest.csv", encoding="utf-8"))
        if r["case_id"] in idx
    ]
    sets["ext"] = DataLoader(MM(ext_items, etf), batch_size=bs)

    img = build_image_model("resnet50", num_classes=2, pretrained=True)
    img.load_state_dict(torch.load(CKPT, map_location=dev))
    out, flat = {}, {}
    for k, ld in sets.items():
        print(f"  encoding {k} ({len(ld.dataset)})...", flush=True)
        fe, lo, y, fs, me = encode(img, ld, dev)
        out[k] = {"feat": fe, "logit": lo, "y": y, "fst": fs, "meta": me}
        for n, v in out[k].items():
            flat[f"{k}_{n}"] = v
    np.savez(CACHE, **flat)
    return out


# --------------------------------------------------------------------------- #
# Gate variants
# --------------------------------------------------------------------------- #
def reliability(li, lm):
    """Distribution-independent reliability signals from two branches' logits."""

    def stats(logits):
        p = F.softmax(logits, dim=1)
        ent = -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True)
        mx = p.max(1, keepdim=True).values
        marg = (logits[:, 1:2] - logits[:, 0:1]).abs()
        return ent, mx, marg

    ei, mi, gi = stats(li)
    em, mm, gm = stats(lm)
    return torch.cat([ei, em, mi, mm, gi, gm, (gi - gm).abs()], dim=1)


class Fusion(nn.Module):
    def __init__(self, meta_dim, img_feat_dim, mode="feature", hidden=128):
        super().__init__()
        self.mode = mode
        self.meta_enc = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.ReLU()
        )
        self.meta_head = nn.Linear(128, 2)
        if mode == "feature":
            self.inorm, self.mnorm = nn.LayerNorm(img_feat_dim), nn.LayerNorm(128)
            self.gate = nn.Sequential(nn.Linear(img_feat_dim + 128, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        else:  # confidence-conditioned
            self.rnorm = nn.LayerNorm(7)
            self.gate = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 2))
        self.last_w = None

    def forward(self, img_feat, img_logit, meta):
        fm = self.meta_enc(meta)
        lm = self.meta_head(fm)
        if self.mode == "feature":
            g = self.gate(torch.cat([self.inorm(img_feat), self.mnorm(fm)], dim=1))
        else:
            g = self.gate(self.rnorm(reliability(img_logit, lm)))
        w = F.softmax(g, dim=1)
        self.last_w = w.detach()
        return w[:, 0:1] * img_logit + w[:, 1:2] * lm


def train(model, d, epochs, p_drop, dev, lr=1e-3, patience=8, lam_ent=0.1):
    Xf = torch.tensor(d["train"]["feat"])
    Xl = torch.tensor(d["train"]["logit"])
    Xm = torch.tensor(d["train"]["meta"])
    Y = torch.tensor(d["train"]["y"]).long()
    Vf, Vl = torch.tensor(d["val"]["feat"]), torch.tensor(d["val"]["logit"])
    Vm, Vy = torch.tensor(d["val"]["meta"]), d["val"]["y"]
    cw = torch.tensor([len(Y) / (2 * (Y == 0).sum().item()), len(Y) / (2 * (Y == 1).sum().item())])
    ce = nn.CrossEntropyLoss(weight=cw.float())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(dev)
    best, best_state, bad = -np.inf, None, 0
    n, bs = len(Y), 64
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i : i + bs]
            if len(j) < 2:
                continue
            m = Xm[j].clone()
            if p_drop > 0:  # per-sample modality dropout
                mask = torch.rand(len(j)) < p_drop
                m[mask] = 0.0
            out = model(Xf[j].to(dev), Xl[j].to(dev), m.to(dev))
            loss = ce(out, Y[j].to(dev))
            if model.last_w is not None:
                w = model.last_w
                loss = loss - lam_ent * (-(w * w.clamp_min(1e-8).log()).sum(1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(Vf.to(dev), Vl.to(dev), Vm.to(dev)), 1)[:, 1].cpu().numpy()
        a = roc_auc_score(Vy, p) if len(np.unique(Vy)) > 1 else 0.0
        if a > best:
            best, bad = a, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def infer(model, d, key, dev):
    model.to(dev).eval()
    out = model(
        torch.tensor(d[key]["feat"]).to(dev),
        torch.tensor(d[key]["logit"]).to(dev),
        torch.tensor(d[key]["meta"]).to(dev),
    )
    return (
        torch.softmax(out, 1)[:, 1].cpu().numpy(),
        model.last_w[:, 1].cpu().numpy() if model.last_w is not None else None,
    )


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def paired(y, a, b, n=4000, seed=42):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            d.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--p-drop", type=float, default=0.4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("building / loading feature cache...", flush=True)
    d = build_cache(dev)

    ye, fe = d["ext"]["y"], d["ext"]["fst"]
    yi = d["test"]["y"]
    img_ext = d["ext"]["logit"][:, 1] - d["ext"]["logit"][:, 0]
    L4, D = (fe >= 1) & (fe <= 4), fe >= 5

    variants = [
        ("A. feature-gate, no dropout", "feature", 0.0),
        ("B. feature-gate + dropout", "feature", args.p_drop),
        ("C. confidence-gate + dropout", "confidence", args.p_drop),
    ]
    rows = []
    for name, mode, pd in variants:
        torch.manual_seed(42)
        np.random.seed(42)
        m = Fusion(NF, d["train"]["feat"].shape[1], mode=mode)
        vb = train(m, d, args.epochs, pd, dev)
        pe, we = infer(m, d, "ext", dev)
        pi, wi = infer(m, d, "test", dev)
        mu, lo, hi = paired(ye, pe, img_ext)
        rows.append(
            {
                "name": name,
                "val": vb,
                "int": auc(yi, pi),
                "ext": auc(ye, pe),
                "light": auc(ye[L4], pe[L4]),
                "dark": auc(ye[D], pe[D]),
                "wmi": float(wi.mean()),
                "wme": float(we.mean()),
                "d": mu,
                "lo": lo,
                "hi": hi,
            }
        )
        print(
            f"  {name}: int {rows[-1]['int']:.3f} ext {rows[-1]['ext']:.3f} (vs image {mu:+.4f} [{lo:+.4f},{hi:+.4f}])",
            flush=True,
        )

    out = [
        "# Making the gate generalize: modality dropout + confidence conditioning\n",
        "Frozen ResNet-50 features/logits precomputed, so the image branch contributes exactly the",
        "image-only model's logits. All variants evaluated on SCIN under the identical protocol.\n",
        "Reference: **image-only (single) 0.596** external, image 5-fold ensemble 0.632, prior feature-gate 0.617.\n",
        "| Variant | internal test | external | ext. light | ext. dark | meta weight int->ext | "
        "paired vs image-only | 95% CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['name']} | {r['int']:.3f} | **{r['ext']:.3f}** | {r['light']:.3f} | {r['dark']:.3f} | "
            f"{r['wmi']:.3f} -> {r['wme']:.3f} | {r['d']:+.4f} | {r['lo']:+.4f} to {r['hi']:+.4f} |"
        )
    out += [
        "\n## Reading this\n",
        "A larger drop in metadata weight from internal to external indicates the gate is detecting that",
        "metadata has become unreliable. A positive paired difference whose CI excludes zero means the",
        "variant beats the image-only model it wraps. Confidence conditioning is the intervention",
        "expected to transfer, because predictive uncertainty carries comparable meaning across cohorts",
        "whereas raw feature values do not.",
    ]
    (REPORT_DIR / "GATE_ROBUSTNESS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
