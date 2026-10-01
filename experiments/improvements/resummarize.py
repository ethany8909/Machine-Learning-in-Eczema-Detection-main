"""Recompute every summary CSV from the saved full-precision predictions, so tables never
double-round (e.g. 0.5025 -> 0.502 -> '0.502' instead of the manuscript's 0.503)."""

from __future__ import annotations

import numpy as np
from common import RESULTS, auc, write_rows
from lab import load_bank, summarize

GROUPS = [
    "baseline",
    "labels",
    "backbones_harmonized",
    "imbalance_harmonized",
    "imbalance_original",
    "augment_harmonized",
    "synthetic_harmonized",
    "missing_harmonized",
    "redundancy_harmonized",
    "fusion_dinov2_harmonized",
    "fusion_rn50ft_harmonized",
    "fusion_dinov2_original",
    "fusion_rn50ft_original",
    "final",
]


def main():
    bank = load_bank()
    mh = ~np.isnan(bank.yh)
    yh = np.nan_to_num(bank.yh).astype(int)
    for g in GROUPS:
        f = RESULTS / f"{g}_preds.npz"
        if not f.exists():
            continue
        z = np.load(f)
        configs = list(dict.fromkeys(k.rsplit("|", 1)[0] for k in z.files))
        rows = []
        for c in configs:
            oof, ext = z[f"{c}|oof"], z[f"{c}|ext"]
            harmonized = not (g.endswith("_original") or (g == "baseline" and "harmonized subset" not in c))
            mask, y = (mh, yh) if harmonized else (np.ones(len(bank.y), bool), bank.y)
            per = [auc(y[(bank.fold == k) & mask], oof[(bank.fold == k) & mask]) for k in range(5)]
            kw = {"eval_mask": mh, "y_eval": yh} if harmonized else {}
            r = summarize(bank, c, oof, ext, per, n_boot=2000, **kw)
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        write_rows(RESULTS / f"{g}.csv", rows)
        print("rewrote", g, len(rows))


if __name__ == "__main__":
    main()
