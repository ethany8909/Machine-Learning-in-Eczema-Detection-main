"""Does cropping remove locational information SELECTIVELY, or degrade everything?

Extends the crop experiment by probing TWO targets from the SAME features at each
crop level:
    A) body site  - the information that makes metadata redundant
    B) diagnosis  - the information the model actually needs

If site recoverability falls while diagnostic signal holds, cropping strips the
redundant channel while preserving the diagnostic one. That is the structural reason
dermoscopic imaging pairs productively with location metadata while clinical
photography does not. If both fall together, the "selective" claim is wrong.

Both probes use the same ImageNet-pretrained ResNet-50 features and the same
logistic-regression head, so absolute diagnostic AUROC is lower than our fine-tuned
model's - only the SHAPE of the curves is being compared.

    python experiments/external/crop_selectivity_experiment.py
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dermafair.paths import DATA_DIR, output_dir

REPORT_DIR = output_dir("external")
from crop_redundancy_experiment import AGES, SEXES, SITES, canon, features, items, probe  # noqa: E402

from dermafair.models import build_image_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="+", type=float, default=[1.0, 0.7, 0.5, 0.35, 0.25])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = {
        r["image_name"]: canon(r)
        for r in csv.DictReader(open(DATA_DIR / "internal_canonical_manifest.csv", encoding="utf-8"))
    }
    lab = {
        r["image_name"]: int(r["label"])
        for r in csv.DictReader(open(DATA_DIR / "internal_canonical_manifest.csv", encoding="utf-8"))
    }
    tr, te = items("train", meta), items("test", meta)
    Mtr = np.array([m for _, m in tr])
    Mte = np.array([m for _, m in te])
    ytr = np.array([lab[p.name] for p, _ in tr])
    yte = np.array([lab[p.name] for p, _ in te])
    print(f"train {len(tr)} test {len(te)} | psoriasis: train {ytr.sum()} test {yte.sum()} | {dev}")

    model = build_image_model("resnet50", num_classes=2, pretrained=True).to(dev).eval()
    off = len(AGES) + len(SEXES)
    rows = []
    for frac in args.crops:
        print(f"\ncrop {frac:.2f} - extracting...", flush=True)
        Ftr, Fte = features(model, tr, frac, dev), features(model, te, frac, dev)
        site = [
            a for a in (probe(Ftr, Mtr[:, off + i], Fte, Mte[:, off + i]) for i in range(len(SITES))) if a is not None
        ]
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ftr, ytr)
        dis = roc_auc_score(yte, clf.predict_proba(Fte)[:, 1])
        rows.append({"frac": frac, "site": float(np.mean(site)), "disease": float(dis)})
        print(f"  site {rows[-1]['site']:.3f} | disease {dis:.3f}")

    f0, f1 = rows[0], rows[-1]
    ds, dd = f1["site"] - f0["site"], f1["disease"] - f0["disease"]
    L = [
        "# Is cropping selective? Locational vs diagnostic information\n",
        "Two probes trained on identical ImageNet-pretrained features at each crop level. Absolute",
        "diagnostic AUROC is below our fine-tuned model's (0.792) because this is a linear probe on",
        "generic features; only the SHAPE of the two curves is compared.\n",
        "| Crop retained | Body site (locational) | Diagnosis (diagnostic) |",
        "|---|---|---|",
    ]
    for r in rows:
        tag = " (full frame)" if r["frac"] == 1.0 else ""
        L.append(f"| {r['frac']:.0%}{tag} | **{r['site']:.3f}** | **{r['disease']:.3f}** |")
    L += [
        f"\n- body-site recoverability change, full frame to {f1['frac']:.0%}: **{ds:+.3f}**",
        f"- diagnostic signal change over the same range: **{dd:+.3f}**",
        f"- selectivity (locational loss minus diagnostic loss): **{ds - dd:+.3f}**\n",
        "## Interpretation\n",
    ]
    if ds < -0.05 and dd > ds + 0.05:
        L.append(
            "Locational information degrades substantially faster than diagnostic information, "
            "supporting the claim that tight framing strips the redundant channel while preserving "
            "the diagnostic one - the structural reason dermoscopic imaging leaves location metadata "
            "informative whereas clinical photography does not."
        )
    elif abs(ds - dd) <= 0.05:
        L.append(
            "Both signals degrade at comparable rates, so cropping removes information broadly rather "
            "than selectively. The redundancy argument should accordingly be stated as a consequence "
            "of general information loss, not of selective removal."
        )
    else:
        L.append(
            "Diagnostic signal degrades faster than locational signal, indicating that diagnosis here "
            "depends on wide-field context such as lesion distribution. This complicates rather than "
            "supports the dermoscopy account and warrants explicit discussion."
        )
    L.append(
        "\n**Caveat.** Centre-cropping assumes the lesion is centred; where it is not, tighter crops may "
        "exclude the lesion entirely. A decline in diagnostic signal therefore cannot be fully separated "
        "from lesions being cropped out of frame."
    )
    (REPORT_DIR / "CROP_SELECTIVITY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    np.savez(
        REPORT_DIR / "crop_selectivity_results.npz",
        fracs=np.array([r["frac"] for r in rows]),
        site=np.array([r["site"] for r in rows]),
        disease=np.array([r["disease"] for r in rows]),
    )
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
