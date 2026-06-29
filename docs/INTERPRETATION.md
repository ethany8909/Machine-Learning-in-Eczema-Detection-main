# Interpreting DermaFair Outputs

A guide to reading the fairness results and turning them into manuscript claims.

## The fairness score

```
fairness_score = 1 − (max_accuracy_gap / overall_accuracy)
```

- **1.0** — identical accuracy across all Fitzpatrick bands (perfectly fair).
- **0.90–0.99** — mild disparity; acceptable with monitoring.
- **< 0.90** — substantial disparity; investigate before any deployment claim.
- **Can go negative** — when the gap exceeds overall accuracy (severe disparity on a hard task). This is informative, not a bug.

Always read it next to the **bootstrapped CI** on the gap. With ~514 images across 4 bands × 2 classes, per-cell counts are small and point estimates are noisy.

## The master table columns

| Column | Meaning |
|---|---|
| `overall_accuracy` | Sample-size-weighted accuracy across bands |
| `max_accuracy_gap` | max(band acc) − min(band acc) |
| `tpr_gap` / `fpr_gap` | Same gap idea for sensitivity / false-positive rate |
| `fairness_score` | Normalized fairness (above) |
| `kruskal_h`, `kruskal_p` | Kruskal-Wallis test for accuracy differing by band |
| `significant_gap` | True if `kruskal_p < 0.05` |
| `acc_gap_ci_low/high` | Bootstrapped 95% CI on the accuracy gap |

## Reading the gate-weight analysis

The signature result. For the gate network, `gate_weights_by_tone.png` shows the mean
image-vs-metadata weighting per Fitzpatrick band, and the pipeline logs:

```
Corr(Fitzpatrick band, metadata weight) = <r>
```

- **Positive r (e.g. > 0.3)** — the gate leans more on metadata as skin tone darkens.
  This supports the hypothesis that the gate compensates for lower image–lesion
  contrast in darker tones. This is the interpretable, novel finding.
- **r ≈ 0 / flat bars near 0.5** — the gate did not learn tone-adaptive weighting.
  Report this honestly: any fairness gains come from multimodal integration per se,
  not from adaptive fusion. Still publishable; reframe the contribution.

## Claims you can and cannot make

**Can:**
- "Architecture X showed the smallest accuracy gap across Fitzpatrick bands (gap, CI)."
- "The gate network reduced the max accuracy gap relative to fixed late-fusion."
- "Gate weighting correlated with skin tone (r=…), consistent with contrast-compensation."

**Cannot (with N≈514, single cohort, no FST 1–2):**
- "Model X is fair." → say "showed no statistically significant gap in this cohort."
- "These results generalize." → they are specific to DermaCon-IN (Indian cohort, FST 3–6).
- Strong causal claims about *why* an architecture is fairer without the Grad-CAM evidence.

## Pre-empting reviewers

1. Report CIs everywhere; state power limits explicitly.
2. Cite `LEAKAGE_AUDIT.md` in Methods.
3. Show calibration (fused-model reliability) in supplementary.
4. Devote a Limitations paragraph to: small N, single source, absent FST 1–2,
   Fitzpatrick subjectivity, no external validation.
