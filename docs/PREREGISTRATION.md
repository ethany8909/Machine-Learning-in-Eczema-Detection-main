# Pre-specified analysis plans

Both plans below were written on **2026-09-24, before any synthetic-image classifier or any
Fitzpatrick17k image was scored**. Deviations are listed at the end.

## A. Image-level synthetic data

**Recipe.** The final model is used unchanged:
- frozen DINOv2 ViT-B/14 features
- logistic regression with C = 0.003 (chosen once by inner, patient-grouped CV)
- harmonized labels
- class weights, recomputed on the real plus synthetic training images

**Leakage control.** A synthetic image derived from a photo is used only when that source photo
is in the training folds. Synthetic images have weight 1. Test folds, SCIN and Fitzpatrick17k
contain real images only.

**Adoption.** A synthetic set is adopted only if its paired internal AUROC gain over the final
model has a patient-bootstrap 95% CI above zero.

**Also reported.** The internal Fitzpatrick V–VI subgroup, and SCIN overall and V–VI.

**Quality metrics.**
- Skin ITA against the real Fitzpatrick V–VI images
- DINOv2 cosine similarity to the source photo
- A real-vs-synthetic classifier two-sample test (C2ST)
- Train-on-synthetic, test-on-real AUROC

**Generator setting.** The image-to-image strength is the strongest setting at which lesions
remain visible on a contact sheet. This rule was fixed before the pilot that applied it.

## B. Fitzpatrick17k, second external test (run once; nothing tuned or refit on it)

### Images

Every relevant image obtainable at scoring time. Images whose quality-control label is
"wrongly labelled" are excluded; there were none in the subset.

### Labels

**Primary: the clinical eczema spectrum.**
- Psoriasis: psoriasis, pustular psoriasis.
- Eczema: eczema, dyshidrotic eczema, allergic contact dermatitis, seborrheic dermatitis,
  neurodermatitis.
- Excluded, because their names match the keyword rule but they are not eczema:
  acrodermatitis enteropathica (zinc deficiency), factitial dermatitis (self-inflicted) and
  perioral dermatitis (rosacea-like).

**Sensitivity analyses.**
- S1, the mechanical SCIN keyword rule: all ten matching labels.
- S2, the strict core: psoriasis vs eczema, dyshidrotic eczema and allergic contact dermatitis.

### Skin tone

`fitzpatrick_scale`, grouped as I–IV vs V–VI. Unknown (−1) is excluded from tone analyses only.
`fitzpatrick_centaur` is a sensitivity analysis.

### Duplicates and leakage

- Exact duplicates (same MD5) are removed.
- Near-duplicates (DINOv2 ViT-B cosine ≥ 0.95) are resampled together as clusters in the bootstrap.
- Any image with cosine ≥ 0.95 to an internal or SCIN image is excluded and counted.

### Models (all fixed before the test)

| ID | Model | Role |
|---|---|---|
| M0 | Main analysis' fine-tuned ResNet-50, 5-fold ensemble | Reference |
| M1 | Final model: DINOv2 ViT-B + matched labels + class weights, 5 fold models averaged | Primary |
| M2 | Same recipe on DINOv2 ViT-L | Secondary |
| M3 | The synthetic-image set with the best internal AUROC, adopted or not | Secondary |

### Endpoints

- **Primary:** paired ΔAUROC M1 − M0, overall, primary labels.
- **Key secondary:** ΔAUROC on Fitzpatrick V–VI, and each model's I–IV minus V–VI gap.
- **Other:** M2 − M1, M3 − M1, the label sensitivity analyses, and balanced accuracy at 0.5.

### Intervals and decision rule

2,000 paired bootstrap resamples over near-duplicate clusters. An improvement is claimed only
if its 95% CI excludes zero.

## Deviations and events, logged as they happened

1. **One source site was offline.** DermaAmin, which hosts 1,580 of the 1,837 relevant
   Fitzpatrick17k images, no longer resolved, so the test used the 255 Atlas Dermatológico
   images, 204 of them in the primary label set.
2. **The first synthetic-image pilot failed quality control.** Prompt-only re-rendering erased
   the lesions, so darkening was made explicit before any classifier was trained.
3. **Pilot 2 also erased lesions.** Every generator setting tried (strengths 0.2 and 0.34)
   removed lesions too. Pilot 3 kept them at strength 0.1, which became the setting by the
   rule above.
4. **Arm used as M3.** By the pre-specified rule this was the darkening-only set: internal
   AUROC 0.877, the best of the four sets, though none was adopted.
