# Pushing DermaFair to GitHub

This repo is fully scaffolded and the fairness engine is tested. To publish it
under your own account, run the following from inside the `dermafair/` folder.

## 1. Create the remote

On GitHub, create a new **empty** repository named `dermafair` (no README/license —
this repo already has them). Then:

```bash
cd dermafair
git init
git add .
git commit -m "Initial commit: DermaFair fairness evaluation protocol"
git branch -M main
git remote add origin https://github.com/<your-username>/dermafair.git
git push -u origin main
```

## 2. Verify the install + tests on your machine

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # the fairness math tests should pass
```

## 3. Get a citable DOI (for the paper's Code Availability statement)

1. Sign in to https://zenodo.org with your GitHub account.
2. In Zenodo > Settings > GitHub, flip the toggle **on** for `dermafair`.
3. On GitHub, create a release (e.g. tag `v0.1.0`). Zenodo auto-archives it and
   mints a DOI.
4. Paste the DOI into `CITATION.cff` and the README "Citing" section.

## 4. Reference manager

Import `docs/references.bib` into Zotero (`File > Import`) or add it to your
Overleaf project. It is pre-loaded with the anchor papers (Groh 2024, DermaCon-IN,
the fairness/architecture/multimodal literature) organized by theme.

---

**What's already done for you:**
- Full package with working models, fusion strategies (incl. gate network), training harness, and the tested fairness engine
- Three runnable pipeline scripts
- Config, license, .gitignore, pyproject, CITATION.cff
- Example notebook, interpretation guide, BibTeX library

**What you do:**
- Run the three auth/credential steps above (GitHub push, Zenodo DOI, Zotero import)
- Adapt `dermafair/data/dermacon.py` column names to the exact DermaCon-IN schema
