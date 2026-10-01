"""End-to-end confirmation: fine-tune ResNet-50 with the improvements chosen in the
frozen-feature screening, optionally with a TRUE mid-fusion head (metadata is joined to
the image features inside the network and gradients flow back into the CNN).

Same 5 folds as the paper (test = fold k, val = fold k+1 for early stopping), same
external SCIN evaluation (average of the 5 fold models).

    python experiments/improvements/train_endtoend.py --name image_improved --fusion none
    python experiments/improvements/train_endtoend.py --name midfusion_improved --fusion mid
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from common import CACHE, RESULTS, auc, load_cohorts, meta_matrix, tone_report  # noqa: E402
from extract_features import POLICIES  # noqa: E402

torch.set_num_threads(8)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Net(nn.Module):
    def __init__(self, d_meta: int, fusion: str):
        super().__init__()
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Identity()
        self.cnn, self.fusion = m, fusion
        self.img = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 256), nn.GELU())
        if fusion == "mid":
            self.meta = nn.Sequential(nn.Linear(d_meta, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
            self.head = nn.Sequential(nn.Linear(256 + 32, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))
        else:
            self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x, meta, p_modal=0.0):
        h = self.img(self.cnn(x))
        if self.fusion == "mid":
            hm = self.meta(meta)
            if self.training and p_modal > 0:
                hm = hm * (torch.rand(len(hm), 1) > p_modal).float()
            h = torch.cat([h, hm], 1)
        return self.head(h).squeeze(1)


def batches(imgs, idx, policy, bs, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    tf = POLICIES[policy]
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        x = torch.from_numpy(imgs[b]).permute(0, 3, 1, 2)
        yield b, torch.stack([tf(im) for im in x])


@torch.no_grad()
def predict(net, imgs, meta, idx, bs=64):
    net.eval()
    out = []
    for b, x in batches(imgs, idx, "clean", bs, False, None):
        out.append(torch.sigmoid(net(x, torch.from_numpy(meta[b]))).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", choices=["original", "harmonized"], default="harmonized")
    ap.add_argument("--fusion", choices=["none", "mid"], default="none")
    ap.add_argument("--policy", default="strong")
    ap.add_argument("--weighting", choices=["class", "cell"], default="class")
    ap.add_argument("--p-modal", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    c = load_cohorts()
    d = c.internal
    imgs = np.load(CACHE / "internal_256.npy", mmap_mode="r")
    imgs_ext = np.load(CACHE / "external_256.npy", mmap_mode="r")
    y = d["label"].to_numpy().astype(float) if args.labels == "original" else d["harmonized"].to_numpy(dtype=float)
    fold = d["fold"].to_numpy()
    dark = (d["fitzpatrick"].to_numpy() >= 5).astype(int)
    M, Me = meta_matrix(d), meta_matrix(c.external)
    out_dir = RESULTS / "endtoend" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    for k in args.folds:
        pf = out_dir / f"fold{k}.npz"
        if pf.exists():
            log(f"fold {k} done already")
            continue
        torch.manual_seed(k)
        rng = np.random.default_rng(k)
        vk = (k + 1) % 5
        ok = ~np.isnan(y)
        tr = np.where((fold != k) & (fold != vk) & ok)[0]
        va = np.where((fold == vk) & ok)[0]
        te = np.where(fold == k)[0]
        ytr = y[tr].astype(int)
        cells = 2 * ytr + dark[tr]
        if args.weighting == "class":
            w = np.where(ytr == 1, len(ytr) / (2 * ytr.sum()), len(ytr) / (2 * (len(ytr) - ytr.sum())))
        else:
            cnt = {cc: (cells == cc).sum() for cc in np.unique(cells)}
            w = np.array([min(5.0, max(cnt.values()) / cnt[cc]) for cc in cells])
            w = w / w.mean()
        wmap = dict(zip(tr, w))
        net = Net(M.shape[1], args.fusion)
        opt = torch.optim.AdamW(net.parameters(), 1e-4, weight_decay=1e-4)
        best, best_state, bad = -1, None, 0
        log(f"fold {k}: train {len(tr)} val {len(va)} test {len(te)} | {args.name}")
        for ep in range(args.epochs):
            net.train()
            tot = 0.0
            for b, x in batches(imgs, tr, args.policy, 32, True, rng):
                logits = net(x, torch.from_numpy(M[b]), args.p_modal)
                wt = torch.tensor([wmap[i] for i in b], dtype=torch.float32)
                loss = (
                    F.binary_cross_entropy_with_logits(
                        logits, torch.tensor(y[b], dtype=torch.float32), reduction="none"
                    )
                    * wt
                ).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            v = auc(y[va].astype(int), predict(net, imgs, M, va))
            log(f"  epoch {ep + 1:02d} loss {tot / len(tr):.4f} val AUROC {v:.3f}")
            if v > best:
                best, bad = v, 0
                best_state = {kk: vv.clone() for kk, vv in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(best_state)
        p_te = predict(net, imgs, M, te)
        p_ext = predict(net, imgs_ext, Me, np.arange(len(imgs_ext)))
        np.savez(pf, te=te, p_te=p_te, p_ext=p_ext, best_val=best)
        log(
            f"fold {k}: test AUROC {auc(y[te][~np.isnan(y[te])].astype(int), p_te[~np.isnan(y[te])]):.3f}  ext AUROC {auc(c.external['label'], p_ext):.3f}"
        )

    if all((out_dir / f"fold{k}.npz").exists() for k in range(5)):
        oof = np.full(len(d), np.nan)
        exts = []
        for k in range(5):
            z = np.load(out_dir / f"fold{k}.npz")
            oof[z["te"]] = z["p_te"]
            exts.append(z["p_ext"])
        ext = np.mean(exts, 0)
        np.savez(out_dir / "predictions.npz", oof=oof, ext=ext)
        tr = tone_report(c.external["label"].to_numpy(), ext, c.external["fitzpatrick"].to_numpy())
        yh = d["harmonized"].to_numpy(dtype=float)
        m = ~np.isnan(yh)
        summary = {
            "int_auroc_harmonized": auc(yh[m].astype(int), oof[m]),
            "int_auroc_original": auc(d["label"], oof),
            "ext_auroc": auc(c.external["label"], ext),
            "ext_I_IV": tr["light"],
            "ext_V_VI": tr["dark"],
            "gap": tr["gap"],
            "gap_ci": tr["gap_ci"],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log("summary", summary)


if __name__ == "__main__":
    main()
