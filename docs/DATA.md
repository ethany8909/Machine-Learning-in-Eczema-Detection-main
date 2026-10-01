# Data

No images, metadata or derived data are stored in this repository. You obtain each cohort from its
publisher, under its own license, and point the code at your local copy.

## Cohorts

| Role | Dataset | What we use | Access and license |
|---|---|---|---|
| Development (internal) | **DermaCon-IN** (Bhuwalka et al., 2025) | Eczema-and-dermatitis vs psoriasis-and-lichenoid images: 1,125 images from 645 patients after cleaning, Fitzpatrick III–VI, with age, sex, body site and clinical descriptors | [Harvard Dataverse, doi:10.7910/DVN/W7OUZM](https://doi.org/10.7910/DVN/W7OUZM); CC BY-NC-SA 4.0 |
| External test 1 | **SCIN** (Ward et al., 2024) | 1,128 crowd-sourced cases (113 psoriasis) labelled eczema or psoriasis by dermatologist differentials, Fitzpatrick I–VI | [google-research-datasets/scin](https://github.com/google-research-datasets/scin); downloaded by `experiments/external/scin_*.py` |
| External test 2 | **Fitzpatrick17k** (Groh et al., 2021) | Atlas images of psoriasis and eczema-spectrum dermatitis; 204 usable in the primary label set (57 Fitzpatrick V–VI) | [mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k); CC BY-NC-SA 3.0; downloaded by `experiments/improvements/fitz17k_download.py` |

All three are for non-commercial research. Do not redistribute the images or anything derived from them.

## Pointing the code at your data

```bash
export DERMAFAIR_DATA_DIR=/path/to/data        # Windows PowerShell: $env:DERMAFAIR_DATA_DIR = "D:\data"
export DERMAFAIR_RESULTS_DIR=/path/to/results  # optional; defaults to ./results
```

Both default to folders inside the repository (`./data`, `./results`), which git ignores.

## Expected layout of `DERMAFAIR_DATA_DIR`

```text
DERMAFAIR_DATA_DIR/
├── Skin_Metadata-1.csv               DermaCon-IN metadata                     (download)
├── DATASET_0/  DATASET_1/            DermaCon-IN images, two source folders   (download)
├── dataset_split/                    single stratified split                  ← scripts/data/split_dataset.py
├── dataset_split_clean/              the same, after de-duplication           ← scripts/data/clean_dataset.py
├── manifest_clean.csv                one row per image, with its CV fold      ← scripts/data/clean_dataset.py
├── scin_metadata/                    scin_cases.csv, scin_labels.csv          ← experiments/external/scin_inventory.py
├── external_scin_manifest.csv        SCIN eczema/psoriasis cases              ← experiments/external/scin_adapter.py
├── internal_canonical_manifest.csv   both cohorts in one feature space        ← experiments/external/scin_adapter.py
├── scin_images/                      SCIN photographs                         ← experiments/external/scin_download.py
└── fitzpatrick17k/                   label CSV and images                     ← experiments/improvements/fitz17k_download.py
```

## Labels

- **Original classes** are DermaCon-IN's subclass groups: "eczema and dermatitis" vs "psoriasis and lichenoid disorders".
- **Harmonized labels** apply the keyword rule used to label SCIN to each image's specific diagnosis: "psoriasis" → psoriasis, "eczema" or "dermatitis" → eczema, anything else excluded (`dermafair/data/labels.py`). Only 219 of the 406 images in the original psoriasis group were psoriasis; the rest were mostly lichen planus and other lichenoid disorders. Relabeling keeps 872 images (653 eczema, 219 psoriasis).
- **Fitzpatrick17k** uses a clinically defined eczema group. Three diagnoses that match the keyword but are not eczema are excluded: acrodermatitis enteropathica, factitial dermatitis and perioral dermatitis ([PREREGISTRATION.md](PREREGISTRATION.md)).

## Splits

- **Patient level.** Every split groups images by `Subject_ID`, so no patient appears in both training and test data.
- **Jointly stratified** by class and Fitzpatrick band.
- **Single split** (70/15/15): the architecture benchmark and the metadata regimes.
- **Five-fold CV** (`manifest_clean.csv`, column `fold`): every cross-validated number, including all improvement experiments.

## Outputs that contain patient images

These stay on your machine:
- the Grad-CAM atlas (`scripts/evaluate/gradcam_atlas.py`)
- misclassification montages (`scripts/evaluate/misclassified.py`)
- the resized-image cache and feature banks (`results/improvements/cache`, `features`)
- synthetic images derived from patient photos (`results/improvements/synthetic`)

They are written under `DERMAFAIR_RESULTS_DIR`, which git ignores. Patterns in `.gitignore` also block images, arrays and checkpoints anywhere in the tree.
