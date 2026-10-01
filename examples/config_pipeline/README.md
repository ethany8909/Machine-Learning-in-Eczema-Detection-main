# Config-driven template

A generic, configuration-driven version of the fairness protocol for applying it to a **new**
dataset: prepare stratified splits, train the image, metadata and fusion models, then run the
per-tone evaluation.

```bash
python examples/config_pipeline/prepare_data.py --config examples/config_pipeline/dermacon.yaml
python examples/config_pipeline/train_all.py    --config examples/config_pipeline/dermacon.yaml
python examples/config_pipeline/run_fairness.py --config examples/config_pipeline/dermacon.yaml
```

Adapt the column names in `dermacon.yaml` and the feature block in `dermafair/data/dermacon.py`
to your metadata schema.

This template did not produce the reported results; those come from `scripts/` and
`experiments/` (see [docs/REPRODUCING.md](../../docs/REPRODUCING.md)). The evaluation itself can
also be used on its own with any classifier's predictions:

```python
from dermafair.fairness import FairnessEvaluator

report = FairnessEvaluator(sensitive_attr="fitzpatrick").evaluate(y_true, y_pred, groups, n_bootstrap=1000)
print(report.fairness_score, report.max_gap, report.kruskal_p)
```
