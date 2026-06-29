# DermaFair

**A reusable protocol for Fitzpatrick-stratified fairness evaluation of multimodal deep learning in dermatology — applied to eczema–psoriasis differential diagnosis.**

DermaFair benchmarks four image architectures (CNN, ResNet-50, ViT-B/16, Hybrid CNN-Transformer), a metadata-only model, and two multimodal fusion strategies (fixed late-fusion and a learned **gate network**), then evaluates every model for fairness across Fitzpatrick skin-tone bands. It ships the full evaluation engine — per-tone metrics, fairness gaps, significance testing, bootstrapped confidence intervals, and Grad-CAM explainability — as a configurable, citable toolkit you can point at other datasets and conditions.

> Built on the [DermaCon-IN](https://arxiv.org/abs/2506.06099) dataset (514 images, Fitzpatrick + Monk labels, metadata; CC BY-NC-SA 4.0).

---

## Why this exists

Prior work (Groh et al., 2024) showed dermatology AI underperforms on darker skin but did not identify *which architectural and fusion choices mitigate this for a specific diagnosis*. DermaFair answers that question for eczema–psoriasis differential diagnosis and releases the measurement protocol so others can extend it.

---

## Install

```bash
git clone https://github.com/<your-username>/dermafair.git
cd dermafair
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Python 3.10+ recommended. See `requirements.txt` for pinned versions.

## Quickstart

1. Download DermaCon-IN from Harvard Dataverse (DOI `10.7910/DVN/W7OUZM`) into `data/raw/`.
2. Edit `configs/dermacon.yaml` to point at your data path.
3. Run the pipeline:

```bash
# 1. Build leakage-free stratified splits
python scripts/prepare_data.py    --config configs/dermacon.yaml

# 2. Train all seven models under identical conditions
python scripts/train_all.py       --config configs/dermacon.yaml

# 3. Run the full fairness evaluation + Grad-CAM + figures
python scripts/run_fairness.py    --config configs/dermacon.yaml
```

Outputs (tables, figures, `fairness_report.md`) land in `results/<run_name>/`.

## What you get

| Output | Description |
|---|---|
| `tables/fairness_master.csv` | Per-model × per-tone metrics, gaps, p-values, bootstrapped CIs |
| `figures/fairness_heatmap.png` | Architecture × Fitzpatrick × metric |
| `figures/accuracy_fairness_pareto.png` | Accuracy vs. fairness trade-off |
| `figures/gate_weights_by_tone.png` | Learned image/metadata weighting per skin tone |
| `gradcam/` | Explainability atlas, organized by model × tone |
| `fairness_report.md` | Auto-compiled narrative summary |

## Repo layout

```
dermafair/
├── dermafair/
│   ├── data/            # dataset, stratified splitting, leakage audit
│   ├── models/          # image backbones, metadata model, fusion strategies
│   ├── fairness/        # per-tone metrics, gaps, KW test, bootstrap
│   ├── explainability/  # Grad-CAM
│   ├── visualization/   # figures + report compiler
│   └── utils/           # seeding, config, logging
├── scripts/             # prepare_data, train_all, run_fairness
├── notebooks/           # dermacon_example.ipynb
├── configs/             # dermacon.yaml
├── tests/               # unit tests for the fairness math
└── docs/                # interpretation guide
```

## The fairness protocol (portable)

The core contribution is `dermafair.fairness`, usable standalone:

```python
from dermafair.fairness import FairnessEvaluator

evaluator = FairnessEvaluator(sensitive_attr="fitzpatrick")
report = evaluator.evaluate(y_true, y_pred, groups, n_bootstrap=1000)
print(report.fairness_score, report.max_gap, report.kruskal_p)
```

Point it at any classifier's predictions plus a group vector — it is not tied to dermatology.

## Reproducibility

- All seeds fixed via `dermafair.utils.set_seed` (NumPy, PyTorch, CUDA).
- Split indices are written to disk and never re-randomized.
- A `LEAKAGE_AUDIT.md` is generated during `prepare_data` documenting duplicate checks and patient-level splitting.

## Citing

If you use DermaFair, please cite the accompanying paper (in preparation) and this software release (Zenodo DOI: _TBD_). See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The DermaCon-IN dataset is licensed separately under CC BY-NC-SA 4.0 and is **not** redistributed here.
