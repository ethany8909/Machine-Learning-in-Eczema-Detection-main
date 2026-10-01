"""Experiment library: classifier heads, imbalance handling, synthetic-data generators,
metadata imputation, and mid-fusion networks, all run under one evaluation protocol.

Protocol (identical for every row of every table):
  * internal: 5-fold CV on the manifest folds (subject-level, class x tone stratified);
    heads train on the 4 other folds with hyperparameters fixed in advance.
  * external: the 5 fold models' SCIN predictions are averaged (as in the paper).
  * anything learned from data (scalers, PCA, generators, imputers) is fit on the
    training folds only, so no information leaks from the fold being scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import FEAT, K, auc, boot_ci, load_cohorts, meta_matrix, tone_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(8)


# --------------------------------------------------------------------------- #
# Data bank
# --------------------------------------------------------------------------- #
@dataclass
class Bank:
    y: np.ndarray  # original labels (eczema & dermatitis vs psoriasis & lichenoid)
    yh: np.ndarray  # harmonized labels (nan = excluded under the SCIN keyword rule)
    fold: np.ndarray
    fst: np.ndarray
    subject: np.ndarray
    y_ext: np.ndarray
    fst_ext: np.ndarray
    meta_int: np.ndarray  # one-hot age/sex/sites with "unknown" slots
    meta_ext: np.ndarray
    df_int: object
    df_ext: object
    feats: dict = field(default_factory=dict)

    @property
    def dark(self):
        return (self.fst >= 5).astype(int)

    def cell(self, y):
        """0..3 = (class, dark-skin) cell, used for tone-aware weighting and generation."""
        return 2 * y.astype(int) + self.dark


def load_bank() -> Bank:
    c = load_cohorts()
    d, e = c.internal, c.external
    return Bank(
        y=d["label"].to_numpy(),
        yh=d["harmonized"].to_numpy(dtype=float),
        fold=d["fold"].to_numpy(),
        fst=d["fitzpatrick"].to_numpy(),
        subject=d["subject"].to_numpy(),
        y_ext=e["label"].to_numpy(),
        fst_ext=e["fitzpatrick"].to_numpy(),
        meta_int=meta_matrix(d),
        meta_ext=meta_matrix(e),
        df_int=d,
        df_ext=e,
    )


def features(bank: Bank, name: str):
    """Returns (internal, external) arrays. rn50_ft has one feature set per fold model."""
    if name not in bank.feats:
        z = np.load(FEAT / f"{name}.npz")
        bank.feats[name] = {k: z[k] for k in z.files}
    return bank.feats[name]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
def summarize(
    bank: Bank,
    name: str,
    oof: np.ndarray,
    ext: np.ndarray,
    per_fold: list[float],
    eval_mask: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    n_boot=1000,
) -> dict:
    m = np.ones(len(oof), bool) if eval_mask is None else eval_mask
    yy = (bank.y if y_eval is None else y_eval)[m]
    tr = tone_report(bank.y_ext, ext, bank.fst_ext, n=n_boot)
    ci = boot_ci(bank.y_ext, ext, n=n_boot)
    pred = (oof[m] >= 0.5).astype(int)
    sens = float((pred[yy == 1] == 1).mean())
    spec = float((pred[yy == 0] == 0).mean())
    return {
        "config": name,
        "int_auroc": auc(yy, oof[m]),
        "int_fold_mean": float(np.nanmean(per_fold)),
        "int_fold_sd": float(np.nanstd(per_fold)),
        "int_bal_acc": (sens + spec) / 2,
        "int_sens": sens,
        "ext_auroc": auc(bank.y_ext, ext),
        "ext_lo": ci[0],
        "ext_hi": ci[1],
        "ext_I_IV": tr["light"],
        "ext_V_VI": tr["dark"],
        "gap": tr["gap"],
        "gap_lo": tr["gap_ci"][0],
        "gap_hi": tr["gap_ci"][1],
        "_oof": oof,
        "_ext": ext,
    }


# --------------------------------------------------------------------------- #
# Generic CV driver
# --------------------------------------------------------------------------- #
def cross_validate(
    bank: Bank,
    fit_predict,
    labels: np.ndarray,
    train_mask: np.ndarray | None = None,
    score_labels: np.ndarray | None = None,
):
    """fit_predict(tr_idx, te_idx, fold) -> (p_test, p_ext). Labels may contain nan
    (excluded rows are never trained on). Returns oof, ext (fold-averaged), per-fold AUROC."""
    n = len(labels)
    oof = np.full(n, np.nan)
    exts, per = [], []
    usable = ~np.isnan(labels.astype(float))
    if train_mask is not None:
        usable &= train_mask
    for k in range(K):
        te = np.where(bank.fold == k)[0]
        tr = np.where((bank.fold != k) & usable)[0]
        p_te, p_ext = fit_predict(tr, te, k)
        oof[te] = p_te
        exts.append(p_ext)
        lab = (labels if score_labels is None else score_labels)[te].astype(float)
        ok = ~np.isnan(lab)
        per.append(auc(lab[ok], p_te[ok]))
    return oof, np.mean(exts, 0), per


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
def logreg(Xtr, ytr, w=None, C=0.1):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(sc.transform(Xtr), ytr, sample_weight=w)
    return lambda X: m.predict_proba(sc.transform(X))[:, 1]


def class_weights(y):
    y = y.astype(int)
    w = np.ones(len(y))
    for c in (0, 1):
        w[y == c] = len(y) / (2 * max(1, (y == c).sum()))
    return w


def cell_weights(cells, cap=5.0):
    """Inverse frequency of each (class x tone) cell, capped (the paper's tone reweighting,
    extended to also balance the classes)."""
    w = np.ones(len(cells))
    counts = {c: (cells == c).sum() for c in np.unique(cells)}
    mx = max(counts.values())
    for c, n in counts.items():
        w[cells == c] = min(cap, mx / n)
    return w / w.mean()


# --------------------------------------------------------------------------- #
# Synthetic data (points 4 and 7). All generators work in a PCA latent space fit on the
# training fold only, and are conditioned on the (class x tone) cell.
# --------------------------------------------------------------------------- #
def smote(X, n_new, rng, k=5):
    if n_new <= 0 or len(X) < 2:
        return np.zeros((0, X.shape[1]), np.float32)
    nn_ = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    idx = nn_.kneighbors(X, return_distance=False)[:, 1:]
    i = rng.integers(0, len(X), n_new)
    j = idx[i, rng.integers(0, idx.shape[1], n_new)]
    lam = rng.random((n_new, 1))
    return (X[i] + lam * (X[j] - X[i])).astype(np.float32)


class CVAE(nn.Module):
    def __init__(self, d, n_cond=4, h=256, z=16):
        super().__init__()
        self.z = z
        self.enc = nn.Sequential(nn.Linear(d + n_cond, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.mu, self.lv = nn.Linear(h, z), nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z + n_cond, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x, c):
        h = self.enc(torch.cat([x, c], 1))
        mu, lv = self.mu(h), self.lv(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.dec(torch.cat([z, c], 1)), mu, lv


def train_cvae(X, cells, seed, epochs=400, beta=0.5):
    torch.manual_seed(seed)
    m = CVAE(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32)
    Ct = F.one_hot(torch.tensor(cells), 4).float()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i : i + 128]
            rec, mu, lv = m(Xt[b], Ct[b])
            kl = -0.5 * (1 + lv - mu**2 - lv.exp()).sum(1).mean()
            loss = F.mse_loss(rec, Xt[b], reduction="none").sum(1).mean() + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()

    @torch.no_grad()
    def sample(cell, n):
        c = F.one_hot(torch.full((n,), cell), 4).float()
        return m.dec(torch.cat([torch.randn(n, m.z), c], 1)).numpy()

    return sample


class Denoiser(nn.Module):
    """epsilon-prediction network for a small conditional DDPM in the PCA latent."""

    def __init__(self, d, n_cond=5, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + n_cond + 32, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, d),
        )
        self.freqs = torch.exp(torch.linspace(0, 6, 16))

    def forward(self, x, t, c):
        a = t[:, None] * self.freqs[None]
        return self.net(torch.cat([x, c, a.sin(), a.cos()], 1))


def train_ddpm(X, cells, seed, steps=4000, T=200, guidance=1.5):
    """Conditional denoising diffusion (Ho et al. 2020) with classifier-free guidance
    (Ho & Salimans 2022): the condition is dropped 15% of the time during training."""
    torch.manual_seed(seed)
    d = X.shape[1]
    betas = torch.linspace(1e-4, 0.02, T)
    abar = torch.cumprod(1 - betas, 0)
    m = Denoiser(d)
    opt = torch.optim.AdamW(m.parameters(), 2e-4)
    Xt = torch.tensor(X, dtype=torch.float32)
    C = F.one_hot(torch.tensor(cells), 5).float()  # slot 4 = "unconditional"
    for _ in range(steps):
        b = torch.randint(0, len(Xt), (128,))
        t = torch.randint(0, T, (128,))
        eps = torch.randn(128, d)
        xt = abar[t].sqrt()[:, None] * Xt[b] + (1 - abar[t]).sqrt()[:, None] * eps
        c = C[b].clone()
        drop = torch.rand(128) < 0.15
        c[drop] = F.one_hot(torch.full((int(drop.sum()),), 4), 5).float()
        loss = F.mse_loss(m(xt, t.float() / T, c), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()

    @torch.no_grad()
    def sample(cell, n):
        x = torch.randn(n, d)
        cc = F.one_hot(torch.full((n,), cell), 5).float()
        cu = F.one_hot(torch.full((n,), 4), 5).float()
        for t in reversed(range(T)):
            tt = torch.full((n,), t / T)
            e = (1 + guidance) * m(x, tt, cc) - guidance * m(x, tt, cu)
            x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / (1 - betas[t]).sqrt()
            if t > 0:
                x += betas[t].sqrt() * torch.randn_like(x)
        return x.numpy()

    return sample


def gaussian_gen(X, cells):
    """Per-cell Gaussian with shrunk covariance (the simplest generative model)."""
    from sklearn.covariance import LedoitWolf

    fits = {}
    for c in np.unique(cells):
        Z = X[cells == c]
        fits[c] = (Z.mean(0), LedoitWolf().fit(Z).covariance_) if len(Z) > 2 else (Z.mean(0), np.eye(X.shape[1]) * 1e-2)

    # One generator shared across calls, created once (reproduces the reported samples).
    def sample(cell, n, rng=np.random.default_rng(0)):  # noqa: B008
        mu, cov = fits[cell]
        return rng.multivariate_normal(mu, cov, n)

    return sample


def synth_targets(cells, mode: str, cap=3.0) -> dict[int, int]:
    """How many synthetic samples to add per (class x tone) cell.
    'class': bring the minority class up to the majority (tone-proportional).
    'cell' : bring every cell up to the largest cell, capped at `cap` x its real size."""
    counts = {c: int((cells == c).sum()) for c in range(4)}
    if mode == "class":
        n0, n1 = counts[0] + counts[1], counts[2] + counts[3]
        minority, deficit = (0, n1 - n0) if n0 < n1 else (1, n0 - n1)
        cs = [2 * minority, 2 * minority + 1]
        tot = sum(counts[c] for c in cs) or 1
        return {c: int(round(deficit * counts[c] / tot)) for c in cs}
    mx = max(counts.values())
    return {c: int(min(mx, cap * counts[c]) - counts[c]) for c in range(4) if counts[c] > 1}


def augment_with_synthetic(Xtr, ytr, cells, method, mode, seed, n_pca=128):
    """Returns (X_aug, y_aug, is_synthetic) in the ORIGINAL feature space's PCA latent.
    The caller trains the classifier in the same latent for every method, including
    'none', so the comparison is fair."""
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_pca, random_state=seed).fit(sc.transform(Xtr))
    Z = pca.transform(sc.transform(Xtr))
    zs = Z.std(0) + 1e-6
    Zn = Z / zs
    if method == "none":
        return pca, sc, Z, ytr, np.zeros(len(Z), bool)
    targets = synth_targets(cells, mode)
    if method == "cvae":
        gen = train_cvae(Zn, cells, seed)
    elif method == "ddpm":
        gen = train_ddpm(Zn, cells, seed)
    elif method == "gaussian":
        gen = gaussian_gen(Zn, cells)
    new, lab = [], []
    for c, n in targets.items():
        if n <= 0:
            continue
        if method == "smote":
            s = smote(Zn[cells == c], n, rng)
        elif method == "noise":
            src = Zn[cells == c][rng.integers(0, (cells == c).sum(), n)]
            s = src + rng.normal(0, 0.3, src.shape)
        else:
            s = gen(c, n)
        new.append(s)
        lab.append(np.full(len(s), c // 2))
    if not new:
        return pca, sc, Z, ytr, np.zeros(len(Z), bool)
    S = np.concatenate(new) * zs
    return (
        pca,
        sc,
        np.concatenate([Z, S]),
        np.concatenate([ytr, np.concatenate(lab)]),
        np.r_[np.zeros(len(Z), bool), np.ones(len(S), bool)],
    )


# --------------------------------------------------------------------------- #
# Mid fusion (points 8-10)
# --------------------------------------------------------------------------- #
class FusionNet(nn.Module):
    """mode:
    image  - image branch only
    meta   - metadata branch only
    concat - MID fusion: the two branches' learned features are concatenated and a joint
             network learns interactions and makes one combined decision
    film   - MID fusion: metadata features scale/shift the image features (FiLM),
             so metadata changes HOW the image is read rather than casting a separate vote
    """

    def __init__(self, d_img, d_meta, mode="concat", h=128, hm=32, p_drop=0.4, p_modal=0.0):
        super().__init__()
        self.mode, self.p_modal = mode, p_modal
        self.img = nn.Sequential(nn.LayerNorm(d_img), nn.Dropout(p_drop), nn.Linear(d_img, h), nn.GELU())
        self.meta = nn.Sequential(nn.Linear(d_meta, hm), nn.GELU(), nn.Linear(hm, hm), nn.GELU())
        if mode == "concat":
            self.head = nn.Sequential(nn.Linear(h + hm, h), nn.GELU(), nn.Dropout(p_drop), nn.Linear(h, 1))
        elif mode == "film":
            self.film = nn.Linear(hm, 2 * h)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(p_drop), nn.Linear(h, 1))
        elif mode == "image":
            self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(p_drop), nn.Linear(h, 1))
        elif mode == "meta":
            self.head = nn.Sequential(nn.Linear(hm, hm), nn.GELU(), nn.Linear(hm, 1))

    def forward(self, xi, xm):
        if self.mode == "meta":
            return self.head(self.meta(xm)).squeeze(1)
        hi = self.img(xi)
        if self.mode == "image":
            return self.head(hi).squeeze(1)
        hm = self.meta(xm)
        if self.training and self.p_modal > 0:  # modality dropout: sometimes hide the metadata
            keep = (torch.rand(len(hm), 1) > self.p_modal).float()
            hm = hm * keep
        if self.mode == "concat":
            return self.head(torch.cat([hi, hm], 1)).squeeze(1)
        g, b = self.film(hm).chunk(2, 1)
        return self.head(hi * (1 + g) + b).squeeze(1)


def train_fusion(Xi, Xm, y, w, mode, seed, epochs=60, lr=1e-3, wd=1e-2, p_modal=0.0, meta_noise=None):
    """meta_noise(Xm_batch, rng) optionally corrupts metadata during training (e.g. to
    simulate missing values, point 5)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(Xi)
    Xi_t = torch.tensor(sc.transform(Xi), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    net = FusionNet(Xi.shape[1], Xm.shape[1], mode, p_modal=p_modal)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=wd)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(y_t))
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            xm = Xm[b.numpy()]
            if meta_noise is not None:
                xm = meta_noise(xm.copy(), rng)
            out = net(Xi_t[b], torch.tensor(xm, dtype=torch.float32))
            loss = (F.binary_cross_entropy_with_logits(out, y_t[b], reduction="none") * w_t[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    @torch.no_grad()
    def predict(Xi2, Xm2):
        return torch.sigmoid(
            net(torch.tensor(sc.transform(Xi2), dtype=torch.float32), torch.tensor(Xm2, dtype=torch.float32))
        ).numpy()

    return predict
