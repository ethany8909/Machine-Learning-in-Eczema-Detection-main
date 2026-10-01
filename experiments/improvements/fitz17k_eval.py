"""Second external test set: Fitzpatrick17k (Groh et al., 2021), run ONCE under the plan
in docs/PREREGISTRATION.md, written on 2026-09-24 before any image was scored.

Nothing is tuned or refit on these images. The models are the ones already fixed:
  M0  the paper's fine-tuned ResNet-50, 5-fold ensemble (original preprocessing)
  M1  final: frozen DINOv2 ViT-B/14 + LR + harmonized labels + class weights (5 fold models averaged)
  M2  (secondary) the same recipe on DINOv2 ViT-L/14 - does SCIN's ViT-L advantage replicate?
  M3  (secondary) the synthetic-image arm with the best INTERNAL AUROC, adopted or not

Labels (primary = clinical eczema spectrum; S1 = the mechanical SCIN keyword rule; S2 = strict core),
skin tone from `fitzpatrick_scale` (I-IV vs V-VI; `fitzpatrick_centaur` as a sensitivity check),
and a bootstrap over near-duplicate clusters for every interval.

    python experiments/improvements/fitz17k_eval.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import CACHE, FEAT, RESULTS, ROOT, K, auc, write_rows
from lab import features, load_bank
from PIL import Image
from run_experiments import log, params
from synth_eval import ARMS, fold_predictor, syn_bank

from dermafair.paths import RESULTS_DIR

F17 = ROOT / "fitzpatrick17k"
PSO = {"psoriasis", "pustular psoriasis"}
ECZ_PRIMARY = {
    "eczema",
    "dyshidrotic eczema",
    "allergic contact dermatitis",
    "seborrheic dermatitis",
    "neurodermatitis",
}
ECZ_STRICT = {"eczema", "dyshidrotic eczema", "allergic contact dermatitis"}
DUP = 0.95  # cosine threshold for near-duplicates and for leakage against internal/SCIN images
N_BOOT = 2000


def label_sets(lab: pd.Series) -> dict[str, np.ndarray]:
    """1 = psoriasis, 0 = eczema, nan = not in this label set."""

    def mk(pos, neg):
        return np.where(lab.isin(pos), 1.0, np.where(lab.isin(neg), 0.0, np.nan))

    kw_pos = lab.str.contains("psoriasis")
    kw_neg = ~kw_pos & lab.str.contains("eczema|dermatitis")
    return {
        "primary": mk(PSO, ECZ_PRIMARY),
        "S1 keyword rule": np.where(kw_pos, 1.0, np.where(kw_neg, 0.0, np.nan)),
        "S2 strict core": mk({"psoriasis"}, ECZ_STRICT),
    }


def cohort() -> pd.DataFrame:
    d = pd.read_csv(F17 / "fitzpatrick17k.csv")
    d = d[d["label"].str.contains("psoria|eczema|dermatitis")].drop_duplicates("md5hash")
    d = d[d["qc"].fillna("") != "3 Wrongly labelled"]
    d["path"] = d["md5hash"].map(lambda h: F17 / "images" / f"{h}.jpg")
    d = d[d["path"].map(Path.exists)].reset_index(drop=True)
    d["source"] = np.where(d["url"].str.contains("atlasdermatologico"), "atlas", "archive")
    return d


def cache_256(d: pd.DataFrame) -> np.ndarray:
    f = CACHE / f"fitz17k_256_{len(d)}.npy"
    if f.exists():
        return np.load(f)
    from extract_features import _load

    arr = np.stack([_load(str(p)) for p in d["path"]])
    np.save(f, arr)
    return arr


def dino_feats(d, imgs, size):
    f = FEAT / f"dinov2_{size}14_fitz17k_{len(d)}.npz"
    if not f.exists():
        from extract_features import dinov2, encode

        log(f"DINOv2 ViT-{size.upper()} features for {len(d)} Fitzpatrick17k images")
        np.savez(f, X=encode(dinov2(size), imgs))
    return np.load(f)["X"]


@torch.no_grad()
def paper_resnet(d) -> np.ndarray:
    from dermafair.data.folder_split import _build_transforms
    from dermafair.models import build_image_model

    tfm = _build_transforms(224, augment=False)
    preds = []
    for k in range(K):
        m = build_image_model("resnet50", num_classes=2, pretrained=False)
        m.load_state_dict(torch.load(RESULTS_DIR / f"cv_clean/checkpoints/resnet50_fold{k}.pt", map_location="cpu"))
        m.eval()
        out = []
        for i in range(0, len(d), 32):
            x = torch.stack([tfm(Image.open(p).convert("RGB")) for p in d["path"].iloc[i : i + 32]])
            out.append(torch.softmax(m(x), 1)[:, 1].numpy())
        preds.append(np.concatenate(out))
        log(f"  paper ResNet-50 fold {k}")
    return np.mean(preds, 0)


def dino_model(bank, feat_name, Z, syn=None) -> tuple[np.ndarray, np.ndarray]:
    """Refit the 5 fixed fold models exactly as in the CV and average their predictions on Z.
    Also returns the internal out-of-fold predictions, to check they reproduce the saved ones."""
    C = params()["C"]
    X = features(bank, feat_name)["int"]
    labels = bank.yh
    usable = ~np.isnan(labels)
    oof, preds = np.full(len(labels), np.nan), []
    for k in range(K):
        tr = np.where((bank.fold != k) & usable)[0]
        te = np.where(bank.fold == k)[0]
        f = fold_predictor(X, labels, tr, k, C, syn)
        oof[te] = f(X[te])
        preds.append(f(Z))
    return np.mean(preds, 0), oof


def clusters(Xb: np.ndarray) -> np.ndarray:
    """Connected components of the near-duplicate graph (cosine >= DUP)."""
    U = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    S = U @ U.T
    parent = list(range(len(U)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(S >= DUP, 1))):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(U))])


def cboot(y, cl, fn, n=N_BOOT, seed=0):
    """Percentile CI of fn(index) under a bootstrap over clusters."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(y))).groupby(cl).apply(list).tolist()
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) > 1:
            v = fn(idx)
            if not np.isnan(v):
                vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    bank = load_bank()
    d = cohort()
    log(
        f"{len(d)} Fitzpatrick17k images on disk ({(d['source'] == 'atlas').sum()} atlas, {(d['source'] == 'archive').sum()} archive)"
    )
    imgs = cache_256(d)
    Zb, Zl = dino_feats(d, imgs, "b"), dino_feats(d, imgs, "l")

    # leakage: any image nearly identical to an internal or SCIN image is dropped
    fb = features(bank, "dinov2_b14_clean")
    U = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)  # noqa: E731
    leak = (U(Zb) @ U(fb["int"]).T).max(1) >= DUP
    leak |= (U(Zb) @ U(fb["ext"]).T).max(1) >= DUP
    log(f"leakage check: {int(leak.sum())} images excluded (cosine >= {DUP} to an internal or SCIN image)")
    keep = ~leak
    d, Zb, Zl = d[keep].reset_index(drop=True), Zb[keep], Zl[keep]
    cl = clusters(Zb)
    log(f"near-duplicate clusters: {len(np.unique(cl))} for {len(d)} images")

    # models (all fixed in advance)
    p = {"M0 paper ResNet-50": paper_resnet(d)}
    p["M1 final DINOv2 ViT-B"], oof_b = dino_model(bank, "dinov2_b14_clean", Zb)
    p["M2 DINOv2 ViT-L (secondary)"], _ = dino_model(bank, "dinov2_l14_clean", Zl)
    saved = np.load(RESULTS / "backbones_harmonized_preds.npz")["DINOv2 ViT-B/14 (86M)|oof"]
    log(
        f"check: refit ViT-B fold models reproduce the saved internal predictions (max diff {np.nanmax(np.abs(oof_b - saved)):.2e})"
    )
    syn_rows = pd.read_csv(RESULTS / "synthetic_images.csv")
    syn_rows = syn_rows[syn_rows["config"].str.startswith("+ ")]
    if len(syn_rows):
        best = syn_rows.loc[syn_rows["int_auroc"].idxmax(), "config"]
        arm = {f"+ {v}": k for k, v in ARMS.items()}[best]
        man, S, _ = syn_bank(arm)
        p[f"M3 {best[2:]} (secondary)"], _ = dino_model(bank, "dinov2_b14_clean", Zb, syn=(man, S))

    fst, cen = d["fitzpatrick_scale"].to_numpy(), d["fitzpatrick_centaur"].to_numpy()
    rows, tests = [], []
    for ls_name, yall in label_sets(d["label"]).items():
        m = ~np.isnan(yall)
        y, c = yall[m].astype(int), cl[m]
        for tone_name, tone in (("fitzpatrick_scale", fst[m]), ("fitzpatrick_centaur", cen[m])):
            if tone_name == "fitzpatrick_centaur" and ls_name != "primary":
                continue
            light, dark = (tone >= 1) & (tone <= 4), tone >= 5
            for name, pr in p.items():
                q = pr[m]
                r = {
                    "labels": ls_name,
                    "tone_source": tone_name,
                    "model": name,
                    "n": int(m.sum()),
                    "n_psoriasis": int(y.sum()),
                    "n_V_VI": int(dark.sum()),
                    "n_V_VI_psoriasis": int(y[dark].sum()),
                }
                r["auroc"] = auc(y, q)
                r["auroc_lo"], r["auroc_hi"] = cboot(y, c, lambda i: auc(y[i], q[i]))
                r["bal_acc"] = float(
                    (
                        ((q >= 0.5) & (y == 1)).sum() / max(1, y.sum())
                        + ((q < 0.5) & (y == 0)).sum() / max(1, (y == 0).sum())
                    )
                    / 2
                )
                r["I_IV"], r["V_VI"] = auc(y[light], q[light]), auc(y[dark], q[dark])
                r["V_VI_lo"], r["V_VI_hi"] = cboot(y[dark], c[dark], lambda i: auc(y[dark][i], q[dark][i]))
                r["gap"] = r["I_IV"] - r["V_VI"]
                rows.append(r)
            # paired comparisons
            pairs = [
                (
                    "M1 final DINOv2 ViT-B",
                    "M0 paper ResNet-50",
                    "PRIMARY" if ls_name == "primary" and tone_name == "fitzpatrick_scale" else "",
                )
            ]
            pairs += [(n, "M1 final DINOv2 ViT-B", "secondary") for n in p if n.startswith(("M2", "M3"))]
            for a, b, role in pairs:
                qa, qb = p[a][m], p[b][m]
                t = {"labels": ls_name, "tone_source": tone_name, "comparison": f"{a} vs {b}", "role": role}
                for grp, g in (("overall", np.ones(len(y), bool)), ("I_IV", light), ("V_VI", dark)):
                    yy, aa, bb, cc = y[g], qa[g], qb[g], c[g]
                    t[f"{grp}_delta"] = auc(yy, aa) - auc(yy, bb)
                    t[f"{grp}_lo"], t[f"{grp}_hi"] = cboot(yy, cc, lambda i: auc(yy[i], aa[i]) - auc(yy[i], bb[i]))
                tests.append(t)
    write_rows(RESULTS / "fitz17k.csv", rows)
    write_rows(RESULTS / "fitz17k_tests.csv", tests)
    np.savez(RESULTS / "fitz17k_preds.npz", md5=d["md5hash"].to_numpy(), cluster=cl, **{k: v for k, v in p.items()})
    for r in rows:
        if r["tone_source"] == "fitzpatrick_scale":
            log(
                f"  [{r['labels']}] {r['model']:<48} n={r['n']:<4} AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]  "
                f"I-IV {r['I_IV']:.3f}  V-VI {r['V_VI']:.3f} (n={r['n_V_VI']})  gap {r['gap']:+.3f}"
            )
    for t in tests:
        log(
            f"  [{t['labels']}|{t['tone_source']}] {t['comparison']}: overall {t['overall_delta']:+.3f} [{t['overall_lo']:+.3f},{t['overall_hi']:+.3f}]  "
            f"V-VI {t['V_VI_delta']:+.3f} [{t['V_VI_lo']:+.3f},{t['V_VI_hi']:+.3f}]  {t['role']}"
        )


if __name__ == "__main__":
    main()
