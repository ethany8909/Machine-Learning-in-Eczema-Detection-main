"""Run the improvement experiments requested in Prof. Homayouni's feedback.

    python experiments/improvements/run_experiments.py --groups baseline labels imbalance augment \
        synthetic missing redundancy fusion final

Selection rule (fixed before running): every choice between variants is made on INTERNAL
cross-validated AUROC only. External (SCIN) results are reported for every row but are
never used to pick a winner; otherwise SCIN would stop being an independent test.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from common import RESULTS, auc, paired_delta, write_rows
from lab import (
    augment_with_synthetic,
    cell_weights,
    class_weights,
    cross_validate,
    features,
    load_bank,
    logreg,
    summarize,
    train_fusion,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from dermafair.paths import RESULTS_DIR

SEEDS = (0, 1, 2)
PARAMS = RESULTS / "params.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(group, rows, bank):
    write_rows(RESULTS / f"{group}.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    np.savez(
        RESULTS / f"{group}_preds.npz",
        **{f"{r['config']}|oof": r["_oof"] for r in rows},
        **{f"{r['config']}|ext": r["_ext"] for r in rows},
    )
    for r in rows:
        log(
            f"  {r['config']:<46} int {r['int_auroc']:.3f}  ext {r['ext_auroc']:.3f} "
            f"[{r['ext_lo']:.3f}-{r['ext_hi']:.3f}]  I-IV {r['ext_I_IV']:.3f}  V-VI {r['ext_V_VI']:.3f}  "
            f"gap {r['gap']:+.3f}  balacc {r['int_bal_acc']:.3f}"
        )


def params():
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else {}


def labels_for(bank, which):
    return bank.y.astype(float) if which == "original" else bank.yh


def eval_args(bank, which):
    """Internal scoring set: harmonized comparisons are scored on the harmonized subset,
    with harmonized labels, for EVERY model, so rows are comparable."""
    if which == "original":
        return {}
    return {"eval_mask": ~np.isnan(bank.yh), "y_eval": np.nan_to_num(bank.yh).astype(int)}


# --------------------------------------------------------------------------- #
def paper_predictions(bank):
    import pandas as pd
    from common import ROOT

    man = pd.read_csv(ROOT / "manifest_clean.csv")
    pos = {n: i for i, n in enumerate(bank.df_int["image_name"])}
    oof = np.full(len(bank.y), np.nan)
    for k in range(5):
        z = np.load(RESULTS_DIR / f"cv_clean/predictions/resnet50_fold{k}.npz")
        idx = [pos[n] for n in man.loc[man["fold"] == k, "image_name"]]
        assert (bank.y[idx] == z["y_true"]).all() and (bank.fst[idx] == z["fitzpatrick"]).all()
        oof[idx] = z["y_prob"][:, 1]
    e = np.load(RESULTS_DIR / "external" / "external_scin_image_predictions.npz")
    assert (e["y_true"] == bank.y_ext).all()
    return oof, e["p_ensemble"]


def g_baseline(bank):
    rows = []
    # (a) the paper's fine-tuned ResNet-50: its own saved predictions, so the numbers match
    #     the manuscript exactly (fold files are in manifest order; mapped back by name)
    oof, ext = paper_predictions(bank)
    per = [auc(bank.y[bank.fold == k], oof[bank.fold == k]) for k in range(5)]
    rows.append(summarize(bank, "paper: fine-tuned ResNet-50", oof, ext, per))
    rows.append(
        summarize(
            bank, "paper: fine-tuned ResNet-50 [harmonized subset]", oof, ext, per, **eval_args(bank, "harmonized")
        )
    )

    # (b) choose the logistic-regression C once, by inner subject-grouped CV (internal only)
    X, Xe = features(bank, "dinov2_b14_clean")["int"], features(bank, "dinov2_b14_clean")["ext"]
    grid = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    scores = {C: [] for C in grid}
    for k in range(5):
        tr = np.where(bank.fold != k)[0]
        for itr, iva in GroupKFold(3).split(tr, groups=bank.subject[tr]):
            a, b = tr[itr], tr[iva]
            for C in grid:
                scores[C].append(auc(bank.y[b], logreg(X[a], bank.y[a], C=C)(X[b])))
    best_C = max(grid, key=lambda C: np.mean(scores[C]))
    log("inner-CV AUROC by C:", {C: round(float(np.mean(v)), 4) for C, v in scores.items()}, "-> C =", best_C)
    PARAMS.write_text(json.dumps({**params(), "C": best_C}, indent=2))

    for name, fe in (
        ("frozen ImageNet ResNet-50 + LR", "rn50_in_clean"),
        ("frozen DINOv2 ViT-B/14 + LR", "dinov2_b14_clean"),
    ):
        X, Xe = features(bank, fe)["int"], features(bank, fe)["ext"]

        def fp(tr, te, k, X=X, Xe=Xe):
            f = logreg(X[tr], bank.y[tr], C=best_C)
            return f(X[te]), f(Xe)

        oof, ext, per = cross_validate(bank, fp, bank.y.astype(float))
        rows.append(summarize(bank, name, oof, ext, per))
    save("baseline", rows, bank)


# --------------------------------------------------------------------------- #
def g_labels(bank):
    """Point 6: remove redundant / mismatched classes."""
    C = params()["C"]
    X, Xe = features(bank, "dinov2_b14_clean")["int"], features(bank, "dinov2_b14_clean")["ext"]
    rows = []
    for which in ("original", "harmonized"):
        labels = labels_for(bank, which)

        def fp(tr, te, k):
            f = logreg(X[tr], labels[tr].astype(int), C=C)
            return f(X[te]), f(Xe)

        oof, ext, per = cross_validate(bank, fp, labels)
        rows.append(summarize(bank, f"train on {which} labels", oof, ext, per, **eval_args(bank, "harmonized")))

    # 3-class: keep the excluded lichenoid/other images as their own class instead of
    # discarding them; score P(psoriasis) / (P(psoriasis) + P(eczema)).
    y3 = np.where(np.isnan(bank.yh), 2, np.nan_to_num(bank.yh)).astype(int)

    def fp3(tr, te, k):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y3[tr])
        r = lambda Z: (lambda P: P[:, 1] / (P[:, 0] + P[:, 1]))(m.predict_proba(sc.transform(Z)))
        return r(X[te]), r(Xe)

    oof, ext, per = cross_validate(bank, fp3, y3.astype(float), score_labels=bank.yh)
    rows.append(summarize(bank, "3-class (eczema / psoriasis / other)", oof, ext, per, **eval_args(bank, "harmonized")))
    save("labels", rows, bank)


# --------------------------------------------------------------------------- #
def g_imbalance(bank, which):
    """Point 2: class and skin-tone imbalance."""
    C = params()["C"]
    X, Xe = features(bank, "dinov2_b14_clean")["int"], features(bank, "dinov2_b14_clean")["ext"]
    labels = labels_for(bank, which)
    rows = []

    def run(name, weight_fn=None, resample=None):
        def fp(tr, te, k):
            ytr = labels[tr].astype(int)
            Xtr, w = X[tr], None
            if weight_fn is not None:
                w = weight_fn(ytr, tr)
            if resample is not None:
                Xtr, ytr = resample(Xtr, ytr, tr, k)
            f = logreg(Xtr, ytr, w, C=C)
            return f(X[te]), f(Xe)

        oof, ext, per = cross_validate(bank, fp, labels)
        rows.append(summarize(bank, name, oof, ext, per, **eval_args(bank, which)))

    def oversample(Xtr, ytr, tr, k):
        rng = np.random.default_rng(k)
        cells = 2 * ytr + bank.dark[tr]
        idx = []
        mx = max((cells == c).sum() for c in range(4))
        for c in range(4):
            ids = np.where(cells == c)[0]
            if len(ids):
                idx += list(ids) + list(rng.choice(ids, int(min(mx, 3 * len(ids)) - len(ids))))
        idx = np.array(idx)
        return Xtr[idx], ytr[idx]

    run("no rebalancing")
    run("class-weighted loss", lambda y, tr: class_weights(y))
    run("class x tone weighted loss (cap 5)", lambda y, tr: cell_weights(2 * y + bank.dark[tr]))
    run("class x tone random oversampling (cap 3x)", resample=oversample)
    save(f"imbalance_{which}", rows, bank)


# --------------------------------------------------------------------------- #
def g_augment(bank, which):
    """Point 3: data augmentation (features of augmented views) and test-time augmentation."""
    C = params()["C"]
    clean = features(bank, "dinov2_b14_clean")
    tta = features(bank, "dinov2_b14_ext_tta")["ext"]
    labels = labels_for(bank, which)
    rows = []
    for pol in ("none", "mild", "strong", "tone", "strong+tone"):
        views = [] if pol == "none" else [features(bank, f"dinov2_b14_{p}")["int"] for p in pol.split("+")]
        views = [v for vv in views for v in vv]
        for use_tta in (False, True) if pol != "none" else (False,):

            def fp(tr, te, k):
                ytr = labels[tr].astype(int)
                Xtr = np.concatenate([clean["int"][tr]] + [v[tr] for v in views])
                Ytr = np.tile(ytr, 1 + len(views))
                f = logreg(Xtr, Ytr, np.tile(class_weights(ytr), 1 + len(views)), C=C)
                p_te = f(clean["int"][te])
                p_ext = f(clean["ext"])
                if use_tta:
                    mild = features(bank, "dinov2_b14_mild")["int"]
                    p_te = np.mean([p_te] + [f(mild[v][te]) for v in range(len(mild))], 0)
                    p_ext = np.mean([p_ext] + [f(tta[v]) for v in range(len(tta))], 0)
                return p_te, p_ext

            oof, ext, per = cross_validate(bank, fp, labels)
            rows.append(
                summarize(bank, f"aug={pol}" + (" + TTA" if use_tta else ""), oof, ext, per, **eval_args(bank, which))
            )
    save(f"augment_{which}", rows, bank)


# --------------------------------------------------------------------------- #
def g_synthetic(bank, which):
    """Points 4 and 7: synthetic training data, from interpolation to generative AI."""
    C = params()["C"]
    clean = features(bank, "dinov2_b14_clean")
    labels = labels_for(bank, which)
    rows, quality = [], []
    for method in ("none", "noise", "smote", "gaussian", "cvae", "ddpm"):
        for mode in ("class", "cell") if method != "none" else ("-",):

            def fp(tr, te, k):
                ytr = labels[tr].astype(int)
                cells = 2 * ytr + bank.dark[tr]
                pca, sc, Z, Y, syn = augment_with_synthetic(clean["int"][tr], ytr, cells, method, mode, seed=k)
                w = class_weights(Y) if method == "none" else np.ones(len(Y))
                m = LogisticRegression(C=C, max_iter=3000).fit(
                    (Z - Z[~syn].mean(0)) / Z[~syn].std(0), Y, sample_weight=w
                )
                mu, sd = Z[~syn].mean(0), Z[~syn].std(0)
                f = lambda A: m.predict_proba((pca.transform(sc.transform(A)) - mu) / sd)[:, 1]
                if k == 0 and method != "none":
                    quality.append(synthetic_quality(method, mode, Z, Y, syn, pca, sc, clean["int"][te], labels[te], C))
                return f(clean["int"][te]), f(clean["ext"])

            oof, ext, per = cross_validate(bank, fp, labels)
            name = "no synthetic data (PCA-128, class-weighted)" if method == "none" else f"{method} [{mode}]"
            rows.append(summarize(bank, name, oof, ext, per, **eval_args(bank, which)))
            log(f"  done {name}")
    write_rows(RESULTS / f"synthetic_quality_{which}.csv", quality)
    save(f"synthetic_{which}", rows, bank)


def synthetic_quality(method, mode, Z, Y, syn, pca, sc, X_test_real, y_test, C):
    """Two standard checks on the synthetic samples (fold 0):
    TSTR - train on synthetic only, test on real held-out images (utility; higher is better)
    C2ST - can a classifier tell real from synthetic? (fidelity; 0.5 = indistinguishable)."""
    from sklearn.model_selection import cross_val_predict

    S, ys = Z[syn], Y[syn]
    R = Z[~syn]
    out = {"method": method, "mode": mode, "n_synthetic": int(syn.sum())}
    ok = ~np.isnan(y_test.astype(float))
    if len(np.unique(ys)) > 1:
        m = LogisticRegression(C=C, max_iter=3000).fit(S, ys)
        out["TSTR_auroc"] = auc(
            y_test[ok].astype(int), m.predict_proba(pca.transform(sc.transform(X_test_real[ok])))[:, 1]
        )
    lab = np.r_[np.zeros(len(R)), np.ones(len(S))]
    p = cross_val_predict(LogisticRegression(C=C, max_iter=3000), np.r_[R, S], lab, cv=5, method="predict_proba")[:, 1]
    out["C2ST_auroc"] = auc(lab, p)
    # memorization: share of synthetic points closer to a real point than real points are to each other
    from sklearn.neighbors import NearestNeighbors

    nn_ = NearestNeighbors(n_neighbors=2).fit(R)
    rr = nn_.kneighbors(R)[0][:, 1]
    sr = nn_.kneighbors(S, n_neighbors=1)[0][:, 0]
    out["near_copy_rate"] = float((sr < np.percentile(rr, 5)).mean())
    return out


# --------------------------------------------------------------------------- #
MISS = {"age": 0.51, "sex": 0.44, "site": 0.15}  # SCIN missingness rates


def simulate_missing(M, rng, rates=MISS):
    """Blank fields at SCIN's missing rates, marking them with the unknown indicators.
    Layout (meta_matrix): 5 age | 3 sex | 10 sites | 1 no-site flag."""
    n = len(M)
    a = rng.random(n) < rates["age"]
    M[a, 0:5] = 0
    M[a, 4] = 1
    s = rng.random(n) < rates["sex"]
    M[s, 5:8] = 0
    M[s, 7] = 1
    t = rng.random(n) < rates["site"]
    M[t, 8:18] = 0
    M[t, 18] = 1
    return M


def impute(Mtr_obs, Mq, how, img_models=None):
    """Fill unknown age / sex / site in Mq using training data (point 5)."""
    Mq = Mq.copy()
    ua, us, ut = Mq[:, 4] == 1, Mq[:, 7] == 1, Mq[:, 18] == 1
    if how == "mode":
        a = Mtr_obs[:, 0:4].sum(0).argmax()
        s = Mtr_obs[:, 5:7].sum(0).argmax()
        t = Mtr_obs[:, 8:18].sum(0).argmax()
        Mq[ua, 0:5] = 0
        Mq[ua, a] = 1
        Mq[us, 5:8] = 0
        Mq[us, 5 + s] = 1
        Mq[ut, 8:19] = 0
        Mq[ut, 8 + t] = 1
    elif how == "knn":
        # nearest training cases on whichever fields ARE observed, then average their values
        for i in np.where(ua | us | ut)[0]:
            cols = []
            if not ua[i]:
                cols += list(range(0, 4))
            if not us[i]:
                cols += list(range(5, 7))
            if not ut[i]:
                cols += list(range(8, 18))
            d = np.abs(Mtr_obs[:, cols] - Mq[i, cols]).sum(1) if cols else np.zeros(len(Mtr_obs))
            nb = Mtr_obs[np.argsort(d, kind="stable")[:15]]
            if ua[i]:
                Mq[i, 0:5] = np.r_[nb[:, 0:4].mean(0), 0]
            if us[i]:
                Mq[i, 5:8] = np.r_[nb[:, 5:7].mean(0), 0]
            if ut[i]:
                Mq[i, 8:19] = np.r_[nb[:, 8:18].mean(0), 0]
    elif how == "image":
        pa, ps, pt = img_models
        Mq[ua, 0:5] = np.c_[pa[ua], np.zeros(ua.sum())]
        Mq[us, 5:8] = np.c_[ps[us], np.zeros(us.sum())]
        Mq[ut, 8:19] = np.c_[pt[ut], np.zeros(ut.sum())]
    return Mq


def image_imputers(Xtr, Mtr, Xq, C):
    """Predict age group, sex and body sites from the image (body site is recoverable from
    the image at AUROC 0.905 in the paper) and use the probabilities to fill gaps."""
    sc = StandardScaler().fit(Xtr)
    A, Q = sc.transform(Xtr), sc.transform(Xq)
    age = LogisticRegression(C=C, max_iter=3000).fit(A, Mtr[:, 0:4].argmax(1))
    pa = np.zeros((len(Xq), 4))
    pa[:, age.classes_] = age.predict_proba(Q)
    sex = LogisticRegression(C=C, max_iter=3000).fit(A, Mtr[:, 5:7].argmax(1))
    ps = np.zeros((len(Xq), 2))
    ps[:, sex.classes_] = sex.predict_proba(Q)
    pt = np.zeros((len(Xq), 10))
    for j in range(10):
        yj = Mtr[:, 8 + j]
        if 0 < yj.sum() < len(yj):
            pt[:, j] = LogisticRegression(C=C, max_iter=3000).fit(A, yj).predict_proba(Q)[:, 1]
    return pa, ps, pt


def g_missing(bank, which):
    """Point 5: missing values. The internal cohort has none; SCIN is missing 51% of ages,
    44% of sexes, 15% of body sites, so the model meets 'unknown' for the first time at test."""
    C = params()["C"]
    clean = features(bank, "dinov2_b14_clean")
    X, Xe = clean["int"], clean["ext"]
    labels = labels_for(bank, which)
    M, Me = bank.meta_int, bank.meta_ext
    rows = []
    for strategy in (
        "unknown indicator (as in paper)",
        "simulated missingness in training",
        "mode imputation",
        "kNN imputation",
        "image-based imputation",
    ):
        for model in ("metadata-only LR", "mid fusion (concat)"):

            def fp(tr, te, k):
                ytr = labels[tr].astype(int)
                rng = np.random.default_rng(k)
                Mtr, Mte_int, Mext = M[tr].copy(), M[te].copy(), Me.copy()
                # internal robustness check: blank the TEST fold at SCIN's rates too
                Mte_int = simulate_missing(Mte_int, np.random.default_rng(100 + k))
                noise = None
                if strategy == "simulated missingness in training":
                    noise = lambda B, r: simulate_missing(B, r)
                elif strategy == "mode imputation":
                    Mte_int, Mext = impute(Mtr, Mte_int, "mode"), impute(Mtr, Mext, "mode")
                elif strategy == "kNN imputation":
                    Mte_int, Mext = impute(Mtr, Mte_int, "knn"), impute(Mtr, Mext, "knn")
                elif strategy == "image-based imputation":
                    Mte_int = impute(Mtr, Mte_int, "image", image_imputers(X[tr], Mtr, X[te], C))
                    Mext = impute(Mtr, Mext, "image", image_imputers(X[tr], Mtr, Xe, C))
                w = class_weights(ytr)
                if model == "metadata-only LR":
                    if noise is not None:  # 5 masked copies of the training set
                        Mtr_aug = np.concatenate([Mtr] + [simulate_missing(Mtr.copy(), rng) for _ in range(5)])
                        f = logreg(Mtr_aug, np.tile(ytr, 6), np.tile(w, 6), C=1.0)
                    else:
                        f = logreg(Mtr, ytr, w, C=1.0)
                    return f(Mte_int), f(Mext)
                preds_te, preds_ext = [], []
                for s in SEEDS:
                    g = train_fusion(X[tr], Mtr, ytr, w, "concat", s, meta_noise=noise)
                    preds_te.append(g(X[te], Mte_int))
                    preds_ext.append(g(Xe, Mext))
                return np.mean(preds_te, 0), np.mean(preds_ext, 0)

            oof, ext, per = cross_validate(bank, fp, labels)
            rows.append(summarize(bank, f"{model} | {strategy}", oof, ext, per, **eval_args(bank, which)))
            log(f"  done {model} | {strategy}")
    save(f"missing_{which}", rows, bank)


# --------------------------------------------------------------------------- #
def g_redundancy(bank, which):
    """Point 6 (features): remove redundant feature dimensions with PCA."""
    C = params()["C"]
    clean = features(bank, "dinov2_b14_clean")
    labels = labels_for(bank, which)
    rows = []
    from sklearn.decomposition import PCA

    for nc in (16, 32, 64, 128, 256, None):

        def fp(tr, te, k):
            ytr = labels[tr].astype(int)
            sc = StandardScaler().fit(clean["int"][tr])
            if nc is None:
                tfm = sc.transform
            else:
                pca = PCA(nc, random_state=0).fit(sc.transform(clean["int"][tr]))
                tfm = lambda A: pca.transform(sc.transform(A))
            f = logreg(tfm(clean["int"][tr]), ytr, class_weights(ytr), C=C)
            return f(tfm(clean["int"][te])), f(tfm(clean["ext"]))

        oof, ext, per = cross_validate(bank, fp, labels)
        rows.append(
            summarize(
                bank, f"PCA {nc} components" if nc else "all 1,536 dimensions", oof, ext, per, **eval_args(bank, which)
            )
        )
    save(f"redundancy_{which}", rows, bank)


# --------------------------------------------------------------------------- #
def g_fusion(bank, which, feature_set="dinov2_b14_clean"):
    """Points 8-10: mid fusion vs late fusion."""
    labels = labels_for(bank, which)
    M, Me = bank.meta_int, bank.meta_ext
    rows = []
    fe = features(bank, feature_set)
    per_fold_feats = fe["int"].ndim == 3  # rn50_ft: one feature set per fold model

    def feats_for(k):
        return (fe["int"][k], fe["ext"][k]) if per_fold_feats else (fe["int"], fe["ext"])

    configs = [
        ("image only", "image", 0.0),
        ("metadata only", "meta", 0.0),
        ("mid fusion: concat", "concat", 0.0),
        ("mid fusion: concat + modality dropout", "concat", 0.3),
        ("mid fusion: FiLM", "film", 0.0),
        ("mid fusion: FiLM + modality dropout", "film", 0.3),
    ]
    store = {}
    for name, mode, pm in configs:

        def fp(tr, te, k):
            X, Xe = feats_for(k)
            ytr = labels[tr].astype(int)
            w = class_weights(ytr)
            pt, pe = [], []
            for s in SEEDS:
                g = train_fusion(X[tr], M[tr], ytr, w, mode, s, p_modal=pm)
                pt.append(g(X[te], M[te]))
                pe.append(g(Xe, Me))
            return np.mean(pt, 0), np.mean(pe, 0)

        oof, ext, per = cross_validate(bank, fp, labels)
        store[mode if pm == 0 else f"{mode}+md"] = (oof, ext)
        rows.append(summarize(bank, name, oof, ext, per, **eval_args(bank, which)))
        log(f"  done {name}")
    # late fusion = average of the separately trained image and metadata models (the paper's design)
    for wimg in (0.5, 0.75):
        oof = wimg * store["image"][0] + (1 - wimg) * store["meta"][0]
        ext = wimg * store["image"][1] + (1 - wimg) * store["meta"][1]
        m = ~np.isnan(labels)
        per = [auc(labels[(bank.fold == k) & m], oof[(bank.fold == k) & m]) for k in range(5)]
        rows.append(summarize(bank, f"late fusion (image weight {wimg})", oof, ext, per, **eval_args(bank, which)))
    tag = "dinov2" if feature_set.startswith("dinov2") else "rn50ft"
    save(f"fusion_{tag}_{which}", rows, bank)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["baseline"])
    ap.add_argument("--labels", nargs="+", default=["harmonized"], choices=["original", "harmonized"])
    args = ap.parse_args()
    bank = load_bank()
    for g in args.groups:
        log(f"=== {g} ===")
        if g == "baseline":
            g_baseline(bank)
        elif g == "labels":
            g_labels(bank)
        elif g == "backbones":
            g_backbones(bank)
        elif g == "final":
            g_final(bank)
        else:
            for which in args.labels:
                log(f"--- {g} / {which} labels ---")
                {
                    "imbalance": g_imbalance,
                    "augment": g_augment,
                    "synthetic": g_synthetic,
                    "missing": g_missing,
                    "redundancy": g_redundancy,
                    "fusion": g_fusion,
                }[g](bank, which)
                if g == "fusion":
                    g_fusion(bank, which, "rn50_ft_clean")


# --------------------------------------------------------------------------- #
# Backbone size (point 1) and the final combined model
# --------------------------------------------------------------------------- #
def g_backbones(bank, which="harmonized"):
    C = params()["C"]
    labels = labels_for(bank, which)
    rows = []
    for name, fe in (
        ("DINOv2 ViT-S/14 (21M params)", "dinov2_s14_clean"),
        ("DINOv2 ViT-B/14 (86M)", "dinov2_b14_clean"),
        ("DINOv2 ViT-L/14 (300M)", "dinov2_l14_clean"),
    ):
        X, Xe = features(bank, fe)["int"], features(bank, fe)["ext"]

        def fp(tr, te, k, X=X, Xe=Xe):
            ytr = labels[tr].astype(int)
            f = logreg(X[tr], ytr, class_weights(ytr), C=C)
            return f(X[te]), f(Xe)

        oof, ext, per = cross_validate(bank, fp, labels)
        rows.append(summarize(bank, name, oof, ext, per, **eval_args(bank, which)))
    save(f"backbones_{which}", rows, bank)


def cluster_delta(bank, mask, y, p_new, p_base, n=2000, seed=0):
    """Paired AUROC difference on internal data, bootstrapping PATIENTS (not photos), since
    several photos of one person are correlated."""
    rng = np.random.default_rng(seed)
    subj = bank.subject[mask]
    y, a, b = y[mask], p_new[mask], p_base[mask]
    groups = {s: np.where(subj == s)[0] for s in np.unique(subj)}
    keys = list(groups)
    ds = []
    for _ in range(n):
        idx = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        if len(np.unique(y[idx])) > 1:
            ds.append(auc(y[idx], a[idx]) - auc(y[idx], b[idx]))
    return auc(y, a) - auc(y, b), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def g_final(bank):
    """Adoption tests for each technique, then the final model vs the paper.

    Rule (fixed before running): a technique enters the final model only if its paired
    improvement in INTERNAL AUROC has a patient-bootstrap 95% CI above zero; otherwise
    the simpler option is kept. External results are reported, never used to choose."""
    mh = ~np.isnan(bank.yh)
    yh = np.nan_to_num(bank.yh).astype(int)
    all_ = np.ones(len(bank.y), bool)

    def preds(group, config):
        if group == "paper":
            return paper_predictions(bank)
        z = np.load(RESULTS / f"{group}_preds.npz")
        return z[f"{config}|oof"], z[f"{config}|ext"]

    H = "harmonized"
    base_syn = "no synthetic data (PCA-128, class-weighted)"
    tests = [
        # (label, new group/config, reference group/config, internal scoring: "all"=original labels, all images)
        (
            "point 1: DINOv2 ViT-B vs paper ResNet-50 (same original labels)",
            ("baseline", "frozen DINOv2 ViT-B/14 + LR"),
            ("paper", ""),
            "all",
        ),
        (
            "point 1: DINOv2 ViT-L vs ViT-B",
            ("backbones_harmonized", "DINOv2 ViT-L/14 (300M)"),
            ("backbones_harmonized", "DINOv2 ViT-B/14 (86M)"),
            H,
        ),
        (
            "point 6: SCIN-matched labels vs original labels",
            ("labels", "train on harmonized labels"),
            ("labels", "train on original labels"),
            H,
        ),
        (
            "point 6: 3-class vs SCIN-matched 2-class",
            ("labels", "3-class (eczema / psoriasis / other)"),
            ("labels", "train on harmonized labels"),
            H,
        ),
        (
            "point 2: class-weighted loss",
            ("imbalance_harmonized", "class-weighted loss"),
            ("imbalance_harmonized", "no rebalancing"),
            H,
        ),
        (
            "point 2: class x tone weighting",
            ("imbalance_harmonized", "class x tone weighted loss (cap 5)"),
            ("imbalance_harmonized", "no rebalancing"),
            H,
        ),
        ("point 3: strong augmentation", ("augment_harmonized", "aug=strong"), ("augment_harmonized", "aug=none"), H),
        (
            "point 3: strong augmentation + TTA",
            ("augment_harmonized", "aug=strong + TTA"),
            ("augment_harmonized", "aug=none"),
            H,
        ),
        (
            "points 4/7: Gaussian-noise synthetic",
            ("synthetic_harmonized", "noise [class]"),
            ("synthetic_harmonized", base_syn),
            H,
        ),
        ("points 4/7: SMOTE", ("synthetic_harmonized", "smote [class]"), ("synthetic_harmonized", base_syn), H),
        (
            "points 4/7: conditional VAE (generative AI)",
            ("synthetic_harmonized", "cvae [class]"),
            ("synthetic_harmonized", base_syn),
            H,
        ),
        (
            "points 4/7: conditional diffusion (generative AI)",
            ("synthetic_harmonized", "ddpm [class]"),
            ("synthetic_harmonized", base_syn),
            H,
        ),
        (
            "point 6: PCA-32 feature reduction",
            ("redundancy_harmonized", "PCA 32 components"),
            ("redundancy_harmonized", "all 1,536 dimensions"),
            H,
        ),
        (
            "point 5: simulated missingness (mid fusion)",
            ("missing_harmonized", "mid fusion (concat) | simulated missingness in training"),
            ("missing_harmonized", "mid fusion (concat) | unknown indicator (as in paper)"),
            H,
        ),
        (
            "points 8-10: mid fusion vs late fusion",
            ("fusion_dinov2_harmonized", "mid fusion: concat + modality dropout"),
            ("fusion_dinov2_harmonized", "late fusion (image weight 0.5)"),
            H,
        ),
        (
            "points 8-10: mid fusion vs image only",
            ("fusion_dinov2_harmonized", "mid fusion: concat + modality dropout"),
            ("fusion_dinov2_harmonized", "image only"),
            H,
        ),
        (
            "points 8-10: mid vs late (paper's ResNet-50 features)",
            ("fusion_rn50ft_harmonized", "mid fusion: concat + modality dropout"),
            ("fusion_rn50ft_harmonized", "late fusion (image weight 0.5)"),
            H,
        ),
    ]
    dark = bank.fst_ext >= 5
    light = (bank.fst_ext >= 1) & (bank.fst_ext <= 4)
    rows, adopted = [], {}
    for label, (g1, c1), (g0, c0), scoring in tests:
        o1, e1 = preds(g1, c1)
        o0, e0 = preds(g0, c0)
        mask, yy = (all_, bank.y) if scoring == "all" else (mh, yh)
        d, lo, hi = cluster_delta(bank, mask, yy, o1, o0)
        de, (elo, ehi) = paired_delta(bank.y_ext, e1, e0)
        dd, (dlo, dhi) = paired_delta(bank.y_ext[dark], e1[dark], e0[dark])
        adopted[label] = lo > 0
        rows.append(
            {
                "test": label,
                "int_delta": d,
                "int_lo": lo,
                "int_hi": hi,
                "adopt (internal CI > 0)": "yes" if lo > 0 else "no",
                "ext_delta": de,
                "ext_lo": elo,
                "ext_hi": ehi,
                "dark_delta": dd,
                "dark_lo": dlo,
                "dark_hi": dhi,
            }
        )
        log(
            f"  {label:<62} int {d:+.3f} [{lo:+.3f},{hi:+.3f}] {'ADOPT' if lo > 0 else '     '}  "
            f"ext {de:+.3f} [{elo:+.3f},{ehi:+.3f}]  V-VI {dd:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
        )
    write_rows(RESULTS / "adoption_tests.csv", rows)

    # final model, assembled strictly by the rule
    big = adopted["point 1: DINOv2 ViT-L vs ViT-B"]
    best = "DINOv2 ViT-L/14 (300M)" if big else "DINOv2 ViT-B/14 (86M)"
    log(f"backbone by the adoption rule: {best}")
    o_new, e_new = preds("backbones_harmonized", best)
    o_pap, e_pap = paper_predictions(bank)
    finals = [
        ("paper: fine-tuned ResNet-50", o_pap, e_pap),
        (f"final: {best} + SCIN-matched labels + class weights", o_new, e_new),
    ]
    if not big:  # reported for transparency only; NOT selected by the rule
        o_l, e_l = preds("backbones_harmonized", "DINOv2 ViT-L/14 (300M)")
        finals.append(("(not selected) DINOv2 ViT-L/14 + SCIN-matched labels + class weights", o_l, e_l))
    save("final", [summarize(bank, n, o, e, [], **eval_args(bank, H), n_boot=2000) for n, o, e in finals], bank)
    comp = []
    for n, o, e in finals[1:]:
        d, lo, hi = cluster_delta(bank, mh, yh, o, o_pap)
        de, (elo, ehi) = paired_delta(bank.y_ext, e, e_pap)
        dl, (llo, lhi) = paired_delta(bank.y_ext[light], e[light], e_pap[light])
        dd, (dlo, dhi) = paired_delta(bank.y_ext[dark], e[dark], e_pap[dark])
        comp.append(
            {
                "comparison": f"{n} vs paper",
                "int_delta": d,
                "int_lo": lo,
                "int_hi": hi,
                "ext_delta": de,
                "ext_lo": elo,
                "ext_hi": ehi,
                "I_IV_delta": dl,
                "I_IV_lo": llo,
                "I_IV_hi": lhi,
                "V_VI_delta": dd,
                "V_VI_lo": dlo,
                "V_VI_hi": dhi,
            }
        )
        log(
            f"  {n} vs paper: internal {d:+.3f} [{lo:+.3f},{hi:+.3f}]  external {de:+.3f} [{elo:+.3f},{ehi:+.3f}]  "
            f"I-IV {dl:+.3f} [{llo:+.3f},{lhi:+.3f}]  V-VI {dd:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
        )
    write_rows(RESULTS / "final_vs_paper.csv", comp)


if __name__ == "__main__":
    main()
