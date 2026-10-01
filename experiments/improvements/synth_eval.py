"""Evaluate the image-space synthetic data from synth_images.py under the protocol in
docs/PREREGISTRATION.md, written on 2026-09-24 before any of it was scored:

  * final recipe: frozen DINOv2 ViT-B/14 + logistic regression (C from params.json),
    harmonized labels, class weights recomputed on real + synthetic training images;
  * a derived image is used only when its SOURCE is in the training folds; test folds,
    SCIN and Fitzpatrick17k contain real images only;
  * adoption: paired internal AUROC gain over the final model with a patient-bootstrap
    95% CI above zero. External results are reported, never used to choose.

    python experiments/improvements/synth_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import FEAT, RESULTS, K, auc, paired_delta, write_rows
from lab import class_weights, cross_validate, features, load_bank, logreg, summarize
from PIL import Image
from run_experiments import cluster_delta, eval_args, log, params
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from synth_images import SYN, ita

ARMS = {
    "tone": "darkened + generator pass",
    "color": "darkened only (no generator)",
    "neutral": "generator pass only (no darkening)",
    "text": "text-to-image from scratch",
}
BASE = "final model (no synthetic images)"


def load_images(folder, files, size=256):
    return np.stack(
        [np.asarray(Image.open(folder / f).convert("RGB").resize((size, size), Image.BICUBIC)) for f in files]
    )


def syn_bank(arm: str):
    """(manifest, DINOv2 ViT-B features, skin ITA) for one synthetic arm, cached."""
    man = pd.read_csv(SYN / arm / "manifest.csv")
    out = FEAT / f"dinov2_b14_syn_{arm}.npz"
    if not out.exists():
        from extract_features import dinov2, encode

        imgs = load_images(SYN / arm, man["file"])
        log(f"encoding {len(imgs)} {arm} images")
        np.savez(out, X=encode(dinov2("b"), imgs), ita=np.array([ita(x) for x in imgs]))
    z = np.load(out)
    return man, z["X"], z["ita"]


def fold_predictor(X, labels, tr, k, C, syn=None):
    """Logistic regression on the real training fold plus (optionally) the synthetic images
    whose source lies outside test fold k (text-to-image images have fold -1: always usable)."""
    Xtr, ytr = X[tr], labels[tr].astype(int)
    if syn is not None:
        man, S = syn
        use = (man["fold"] != k).to_numpy()
        Xtr, ytr = np.r_[Xtr, S[use]], np.r_[ytr, man["label"].to_numpy()[use]]
    return logreg(Xtr, ytr, class_weights(ytr), C=C)


def main():
    bank = load_bank()
    C = params()["C"]
    labels = bank.yh
    mh = ~np.isnan(bank.yh)
    yh = np.nan_to_num(bank.yh).astype(int)
    X, Xe = features(bank, "dinov2_b14_clean")["int"], features(bank, "dinov2_b14_clean")["ext"]
    dark_int = mh & (bank.fst >= 5)
    dark_ext = bank.fst_ext >= 5

    runs = {BASE: None}
    for arm in ARMS:
        if (SYN / arm / "manifest.csv").exists():
            man, S, _ = syn_bank(arm)
            runs[f"+ {ARMS[arm]}"] = (man, S)
    rows, preds = [], {}
    for name, syn in runs.items():

        def fp(tr, te, k, syn=syn):
            f = fold_predictor(X, labels, tr, k, C, syn)
            return f(X[te]), f(Xe)

        oof, ext, per = cross_validate(bank, fp, labels)
        r = summarize(bank, name, oof, ext, per, **eval_args(bank, "harmonized"), n_boot=2000)
        r["int_V_VI"] = auc(yh[dark_int], oof[dark_int])
        r["n_synthetic"] = 0 if syn is None else len(syn[0])
        rows.append(r)
        preds[name] = (oof, ext)
        log(
            f"  {name:<52} int {r['int_auroc']:.3f} (V-VI {r['int_V_VI']:.3f})  ext {r['ext_auroc']:.3f}  "
            f"ext V-VI {r['ext_V_VI']:.3f}  sens {r['int_sens']:.3f}"
        )
    write_rows(RESULTS / "synthetic_images.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    np.savez(
        RESULTS / "synthetic_images_preds.npz",
        **{f"{n}|oof": o for n, (o, e) in preds.items()},
        **{f"{n}|ext": e for n, (o, e) in preds.items()},
    )

    # adoption tests vs the final model (internal decides; external reported)
    o0, e0 = preds[BASE]
    tests = []
    for name, (o1, e1) in preds.items():
        if name == BASE:
            continue
        d, lo, hi = cluster_delta(bank, mh, yh, o1, o0)
        dv, vlo, vhi = cluster_delta(bank, dark_int, yh, o1, o0)
        de, (elo, ehi) = paired_delta(bank.y_ext, e1, e0)
        dd, (dlo, dhi) = paired_delta(bank.y_ext[dark_ext], e1[dark_ext], e0[dark_ext])
        tests.append(
            {
                "test": f"{name} vs final",
                "int_delta": d,
                "int_lo": lo,
                "int_hi": hi,
                "adopt (internal CI > 0)": "yes" if lo > 0 else "no",
                "int_V_VI_delta": dv,
                "int_V_VI_lo": vlo,
                "int_V_VI_hi": vhi,
                "ext_delta": de,
                "ext_lo": elo,
                "ext_hi": ehi,
                "ext_V_VI_delta": dd,
                "ext_V_VI_lo": dlo,
                "ext_V_VI_hi": dhi,
            }
        )
        log(
            f"  {name:<52} int {d:+.3f} [{lo:+.3f},{hi:+.3f}] {'ADOPT' if lo > 0 else '     '} "
            f"int V-VI {dv:+.3f} [{vlo:+.3f},{vhi:+.3f}]  ext {de:+.3f} [{elo:+.3f},{ehi:+.3f}]  ext V-VI {dd:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
        )
    write_rows(RESULTS / "synthetic_images_tests.csv", tests)

    # quality of each arm
    real_ita = np.array([ita(x) for x in np.load(RESULTS.parent / "cache/internal_256.npy")])
    pos = {n: i for i, n in enumerate(bank.df_int["image_name"])}
    unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)  # noqa: E731
    quality = []
    for arm in ARMS:
        if not (SYN / arm / "manifest.csv").exists():
            continue
        man, S, s_ita = syn_bank(arm)
        q = {
            "arm": ARMS[arm],
            "n": len(man),
            "n_psoriasis": int(man["label"].sum()),
            "median_skin_ITA": float(np.median(s_ita)),
            "real_V_VI_median_ITA": float(np.median(real_ita[dark_int])),
            "real_III_IV_median_ITA": float(np.median(real_ita[mh & (bank.fst <= 4)])),
        }
        if arm != "text":
            src = man["source"].map(pos).to_numpy()
            q["source_median_ITA"] = float(np.median(real_ita[src]))
            q["cos_to_source"] = float(np.mean(np.sum(unit(S) * unit(X[src]), 1)))
        # C2ST: can a classifier tell these from real photos? (0.5 = indistinguishable)
        R = X[mh]
        lab = np.r_[np.zeros(len(R)), np.ones(len(S))]
        p = cross_val_predict(LogisticRegression(C=C, max_iter=3000), np.r_[R, S], lab, cv=5, method="predict_proba")[
            :, 1
        ]
        q["C2ST_auroc"] = auc(lab, p)
        # TSTR: train on synthetic only, test on real images of the held-out fold
        tstr = np.full(len(yh), np.nan)
        for k in range(K):
            te = np.where((bank.fold == k) & mh)[0]
            use = (man["fold"] != k).to_numpy()
            f = logreg(S[use], man["label"].to_numpy()[use], class_weights(man["label"].to_numpy()[use]), C=C)
            tstr[te] = f(X[te])
        q["TSTR_auroc"] = auc(yh[mh], tstr[mh])
        q["TSTR_auroc_V_VI"] = auc(yh[dark_int], tstr[dark_int])
        quality.append(q)
        log("  quality", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
    write_rows(RESULTS / "synthetic_images_quality.csv", quality)


if __name__ == "__main__":
    main()
