# Metrics and statistics

How every number in this repository is computed, and how to read it.

## Discrimination

| Metric | Definition | Notes |
|---|---|---|
| **AUROC** | Area under the ROC curve, psoriasis as the positive class | The headline metric. 0.5 is chance, 1.0 is perfect. Threshold-free, so it is unaffected by class balance and by reweighting the loss. |
| **Balanced accuracy** | Mean of sensitivity and specificity at a 0.5 threshold | Reported instead of raw accuracy, because majority-class guessing already scores 0.692 on the internal test split. |
| **Psoriasis sensitivity** | Share of psoriasis cases predicted as psoriasis | The operating-point quantity that class weighting changes. |
| **ECE** | Expected calibration error (10 equal-width bins) | Whether predicted probabilities can be trusted as probabilities. |

## Skin tone

- **Fitzpatrick I–IV vs V–VI.** Per-band counts in the external cohorts are small, so skin tone is analysed as this binary contrast (following Groh et al.). Per-band numbers are shown for pattern only.
- **Gap** = AUROC(I–IV) − AUROC(V–VI). Its confidence interval comes from bootstrapping each group independently.
- **ITA (individual typology angle)** = atan((L\* − 50) / b\*) in degrees, measured on skin pixels in CIELAB. Lower is darker; Fitzpatrick V–VI photographs typically fall below about 10°. ITA is used only to check and steer the skin-tone transformation in the synthetic-image experiment (`dermafair/data/skin_tone.py`).

## Fairness summaries (internal benchmarking)

`dermafair.fairness.FairnessEvaluator` reports, for any predictions and group vector:

| Column | Meaning |
|---|---|
| `max_accuracy_gap` | Best-band minus worst-band accuracy |
| `tpr_gap`, `fpr_gap`, `equalized_odds_diff` | The same idea for sensitivity, false-positive rate, and their maximum |
| `fairness_score` | 1 − max_accuracy_gap / overall_accuracy (1 is perfectly even; can go negative) |
| `kruskal_p` | Kruskal–Wallis test of per-sample correctness across bands |
| `acc_gap_ci_low/high` | Bootstrapped 95% interval for the accuracy gap |

With a few hundred test images across four bands, these are descriptive. A non-significant test is not evidence of fairness.

## Confidence intervals and comparisons

| Situation | Method |
|---|---|
| One model, one test set | Percentile bootstrap over cases, 2,000 resamples |
| Two models, internal data | **Paired** difference, bootstrapping **patients** rather than photos, since photos of one person are correlated |
| Two models, SCIN | Paired bootstrap over cases |
| Two models, Fitzpatrick17k | Paired bootstrap over near-duplicate clusters (DINOv2 cosine ≥ 0.95) |
| Cross-validated means | Mean and SD over the five folds; fold-paired *t* tests for the fusion comparisons |

## Decision rules

1. **Selection rule.** Every choice between variants is made on internal cross-validated results. External cohorts are reported for every configuration but never used to choose.
2. **Adoption rule.** A technique enters the final model only if its paired internal AUROC gain has a 95% interval above zero. This rule was added after the first four groups of improvement experiments. The two exceptions are documented in [RESULTS.md](RESULTS.md).
3. **One-shot external test.** The Fitzpatrick17k analysis was written down before any image was scored and was run once ([PREREGISTRATION.md](PREREGISTRATION.md)).

## Claims these numbers support

- Report every subgroup estimate with its interval, and state the sample size behind it.
- "No significant gap" means the study could not detect one, not that the model is fair.
- Darker-skin results rest on 40–106 images per cohort. They differed between the two external cohorts, so a skin-tone claim needs more than one external dataset.
