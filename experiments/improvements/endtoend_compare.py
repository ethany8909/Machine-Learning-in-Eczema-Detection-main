"""End-to-end comparison: fine-tuned ResNet-50 image-only (A), late fusion (A averaged
50/50 with the metadata-only model, the paper's design), and end-to-end mid fusion (B),
all on harmonized labels, with the same paired tests as the frozen-feature screening."""

from __future__ import annotations

import numpy as np
from common import RESULTS, paired_delta, write_rows
from lab import load_bank, summarize
from run_experiments import cluster_delta, paper_predictions


def main():
    bank = load_bank()
    mh = ~np.isnan(bank.yh)
    yh = np.nan_to_num(bank.yh).astype(int)
    A = np.load(RESULTS / "endtoend/A_image_harmonized_mild/predictions.npz")
    B = np.load(RESULTS / "endtoend/B_midfusion_harmonized_mild/predictions.npz")
    z = np.load(RESULTS / "fusion_dinov2_harmonized_preds.npz")  # metadata-only model (image-independent)
    m_oof, m_ext = z["metadata only|oof"], z["metadata only|ext"]
    late_oof, late_ext = 0.5 * A["oof"] + 0.5 * m_oof, 0.5 * A["ext"] + 0.5 * m_ext
    p_oof, p_ext = paper_predictions(bank)
    runs = [
        ("paper: fine-tuned ResNet-50 (original labels)", p_oof, p_ext),
        ("A: fine-tuned ResNet-50, image only (harmonized)", A["oof"], A["ext"]),
        ("late fusion: A + metadata model, equal weights", late_oof, late_ext),
        ("B: fine-tuned ResNet-50, end-to-end mid fusion", B["oof"], B["ext"]),
    ]
    rows = [summarize(bank, n, o, e, [], eval_mask=mh, y_eval=yh, n_boot=2000) for n, o, e in runs]
    write_rows(RESULTS / "endtoend.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    for r in rows:
        print(
            f"{r['config']:<52} int {r['int_auroc']:.3f}  ext {r['ext_auroc']:.3f} [{r['ext_lo']:.3f}-{r['ext_hi']:.3f}]  "
            f"I-IV {r['ext_I_IV']:.3f}  V-VI {r['ext_V_VI']:.3f}  gap {r['gap']:+.3f}"
        )
    dark = bank.fst_ext >= 5
    tests = [
        ("B mid vs late fusion (end-to-end)", B["oof"], B["ext"], late_oof, late_ext),
        ("B mid fusion vs A image only (end-to-end)", B["oof"], B["ext"], A["oof"], A["ext"]),
        ("A image only vs paper (label change under fine-tuning)", A["oof"], A["ext"], p_oof, p_ext),
    ]
    out = []
    for name, o1, e1, o0, e0 in tests:
        d, lo, hi = cluster_delta(bank, mh, yh, o1, o0)
        de, (elo, ehi) = paired_delta(bank.y_ext, e1, e0)
        dd, (dlo, dhi) = paired_delta(bank.y_ext[dark], e1[dark], e0[dark])
        out.append(
            {
                "test": name,
                "int_delta": d,
                "int_lo": lo,
                "int_hi": hi,
                "ext_delta": de,
                "ext_lo": elo,
                "ext_hi": ehi,
                "dark_delta": dd,
                "dark_lo": dlo,
                "dark_hi": dhi,
            }
        )
        print(
            f"{name:<56} int {d:+.3f} [{lo:+.3f},{hi:+.3f}]  ext {de:+.3f} [{elo:+.3f},{ehi:+.3f}]  V-VI {dd:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
        )
    write_rows(RESULTS / "endtoend_tests.csv", out)


if __name__ == "__main__":
    main()
