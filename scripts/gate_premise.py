#!/usr/bin/env python
"""Week-5 premise check: does the gate lean on METADATA more for DARKER skin?

The paper's novelty hypothesis is that the learned gate adaptively shifts toward
metadata for higher Fitzpatrick bands (where image contrast is lower). This tests
it directly on the saved per-sample gate weights:

  - mean metadata-weight per Fitzpatrick band (+ n)
  - Spearman correlation between per-sample metadata-weight and FST band (rho, p)

Verdict: premise HOLDS if metadata-weight rises with band (positive rho).
If flat / negative -> honest pivot (gate is a benchmarked strategy, no tone story).
Also emits Figure 4 (gate metadata-weight per tone) for the primary triage regime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/week5_gate"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = [3, 4, 5, 6]
REGIMES = ["autonomous", "triage", "expert"]


def load(regime):
    d = np.load(ROOT / f"results/spectrum/{regime}/predictions/gate_network.npz")
    w_meta = d["gate_weights"][:, 1]          # metadata weight per sample
    fitz = d["fitzpatrick"]
    keep = fitz != -1
    return w_meta[keep], fitz[keep]


def analyze(regime):
    w_meta, fitz = load(regime)
    per_band = {b: (w_meta[fitz == b].mean() if (fitz == b).any() else np.nan,
                    int((fitz == b).sum())) for b in BANDS}
    rho, p = spearmanr(fitz, w_meta)          # per-sample, all bands
    # robust version excluding tiny FST6
    m35 = fitz <= 5
    rho35, p35 = spearmanr(fitz[m35], w_meta[m35])
    return per_band, (rho, p), (rho35, p35)


def main():
    lines = ["# Week-5 gate premise check — does metadata-weight rise with darker skin?\n",
             "Hypothesis: gate metadata-weight increases with Fitzpatrick band. "
             "Spearman rho > 0 (positive) supports the tone-adaptive claim.\n",
             "| Regime | FST3 | FST4 | FST5 | FST6 | Spearman rho (p) | rho excl. FST6 (p) |",
             "|---|---|---|---|---|---|---|"]
    print("regime      FST3  FST4  FST5  FST6   rho     p      rho(no6)")
    for r in REGIMES:
        pb, (rho, p), (rho35, p35) = analyze(r)
        cells = " | ".join(f"{pb[b][0]:.2f} (n={pb[b][1]})" for b in BANDS)
        lines.append(f"| {r} | {cells} | {rho:+.3f} ({p:.3f}) | {rho35:+.3f} ({p35:.3f}) |")
        print(f"{r:11} " + "  ".join(f"{pb[b][0]:.2f}" for b in BANDS) +
              f"  {rho:+.3f}  {p:.3f}  {rho35:+.3f}")

    # honest three-way verdict from the primary triage regime + cross-regime consistency
    _, (rho_t, p_t), _ = analyze("triage")
    rhos = [analyze(r)[1][0] for r in REGIMES]
    all_positive = all(r > 0 for r in rhos)
    if rho_t > 0.1 and p_t < 0.05:
        verdict = "HOLDS — significant tone-adaptive signal"
    elif rho_t > 0.05 and all_positive:
        verdict = ("SUGGESTIVE but UNDERPOWERED — metadata-weight trends up with darker skin "
                   f"consistently across all 3 regimes (rho {min(rhos):+.2f}..{max(rhos):+.2f}), "
                   "but not significant (p>0.05). Report as hedged secondary finding, not headline.")
    else:
        verdict = "FAILS — no tone-adaptive signal; pivot to honest negative"
    lines.append(f"\n**Triage verdict:** Spearman rho = {rho_t:+.3f} (p={p_t:.3f}); "
                 f"all-regime rhos positive = {all_positive}. → {verdict}")
    (OUT / "PREMISE_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {verdict}")

    # Figure 4 — gate metadata-weight per tone (triage primary) + regime comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, r in zip(axes, REGIMES):
        w_meta, fitz = load(r)
        data = [w_meta[fitz == b] for b in BANDS]
        parts = ax.violinplot([d if len(d) else [np.nan] for d in data], positions=BANDS, showmeans=True)
        means = [d.mean() if len(d) else np.nan for d in data]
        ax.plot(BANDS, means, "o-", color="#DD8452", lw=2, label="mean metadata-weight")
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_title(f"{r}" + (" (primary)" if r == "triage" else ""))
        ax.set_xlabel("Fitzpatrick band"); ax.set_xticks(BANDS)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("gate METADATA weight")
    plt.suptitle("Figure 4 — Gate metadata-weight by skin tone (does it rise for darker skin?)\n"
                 "FST 6 n≈4–5: interpret with caution", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "figure4_gate_weight_by_tone.png", dpi=150)
    plt.close()
    print(f"wrote {OUT}/figure4_gate_weight_by_tone.png and PREMISE_CHECK.md")


if __name__ == "__main__":
    main()
