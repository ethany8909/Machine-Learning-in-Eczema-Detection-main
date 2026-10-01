# Reproducing the study

Every stage is a plain Python script. The [Makefile](../Makefile) runs them in order with the
exact arguments used for the reported results. Times are for an 8-core CPU without a GPU;
everything was developed on such a machine.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                # add ,explain for Grad-CAM and ,synthesis for SD-Turbo
make test                              # 31 unit tests, no data needed
```

`requirements.txt` pins the exact versions used for the September 2026 analyses (Python 3.14).

## 2. Data

Obtain the cohorts and set `DERMAFAIR_DATA_DIR` as described in [DATA.md](DATA.md). Outputs go to
`DERMAFAIR_RESULTS_DIR` (default `./results`).

## 3. Stages

| Stage | Command | Time | Produces | Results section |
|---|---|---|---|---|
| Data preparation | `make data` | minutes | patient-level splits, de-duplication, CV folds, cohort table | [§1](RESULTS.md#1-internal-benchmarking) |
| Image models | `make internal` | ~12 h | five backbones (single split and 5-fold CV), fusion models, baselines | [§1](RESULTS.md#1-internal-benchmarking) |
| Clinical-information spectrum | `make spectrum` | ~6 h | three metadata regimes, fusion CV, fairness reports, calibration | [§1](RESULTS.md#1-internal-benchmarking) |
| External validation | `make external` | hours | SCIN download and harmonization, external tests, follow-up analyses | [§2](RESULTS.md#2-external-validation-on-scin) |
| Improvement experiments | `make improvements` | ~10 h | foundation-model features, ten-point improvement study, end-to-end checks | [§3](RESULTS.md#3-model-improvement-experiments) |
| Generated images | `make synthetic-images` | ~4 h | SD-Turbo image sets and their evaluation | [§3](RESULTS.md#image-level-synthetic-data-sd-turbo) |
| Second external test | `make fitzpatrick17k` | minutes | Fitzpatrick17k download and the one-shot test | [§4](RESULTS.md#4-second-external-test-fitzpatrick17k) |
| README figures | `make figures` | seconds | `docs/figures/*_light.png`, `*_dark.png` | — |

Run `make help` to list the targets. On Windows without `make`, copy the commands from the
Makefile; every script also accepts `--help`.

## 4. Where the outputs land

```text
DERMAFAIR_RESULTS_DIR/
├── image_models_clean/  fusion_clean/  cv_clean/  cv_triage/  spectrum/  baselines/
├── week4_fairness/  week5_spectrum/  week5_gate/  paper_figures/
├── reports/          cleaning report, cohort composition table
├── external/         EXTERNAL_VALIDATION_*.md, predictions, follow-up reports
└── improvements/
    ├── features/     frozen-backbone feature banks          (derived from patient images)
    ├── cache/        resized-image cache                     (derived from patient images)
    ├── synthetic/    generated images                        (derived from patient images)
    └── results/      every improvement table as CSV, plus TABLES.md
```

## 5. Determinism

- Splits and CV folds are written to disk once, under `DERMAFAIR_DATA_DIR`, and are never redrawn.
- Every random step uses a fixed seed: NumPy, PyTorch, the bootstrap and the generator seeds.
- Reported means over seeds use seeds 0, 1 and 2.
- CPU results are reproducible run to run. On a GPU, PyTorch kernels can differ in the last
  digits.

## 6. Checks done before release

- **Lint and format:** `ruff check` and `ruff format --check` pass.
- **Unit tests:** 31 tests covering the fairness metrics, the label rule, the skin-tone
  transforms and the path configuration.
- **Command-line scripts:** every script runs `--help` (also in CI).
- **Reproduction against real data:** after the repository was reorganized, three stages
  produced byte-identical output to the original analysis:
  - `make_tables.py` (all improvement tables)
  - `make_table_s1.py` (cohort composition)
  - `relative_fairness.py`
