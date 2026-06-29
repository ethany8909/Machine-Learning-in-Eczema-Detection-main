#!/usr/bin/env python
"""Step 1 of the pipeline: build leakage-free stratified splits + Table S1.

Usage:
    python scripts/prepare_data.py --config configs/dermacon.yaml
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from dermafair.data import composition_table, leakage_audit, make_splits
from dermafair.utils import get_logger, load_config, set_seed

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    dcfg = cfg["data"]
    df = pd.read_csv(dcfg["metadata_csv"])

    # binarize label
    label_col = dcfg["label_col"]
    df = df[df[label_col].isin([dcfg["positive_class"], dcfg["negative_class"]])].copy()
    df["label"] = (df[label_col] == dcfg["positive_class"]).astype(int)

    sens = dcfg["sensitive_attr"]
    splits = make_splits(
        df,
        label_col="label",
        stratify_cols=["label", sens],
        patient_col=dcfg.get("patient_col"),
        train=cfg["split"]["train"],
        val=cfg["split"]["val"],
        test=cfg["split"]["test"],
        seed=cfg["seed"],
    )
    for k, v in splits.items():
        log.info(f"{k}: {len(v)} samples")

    out_dir = Path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    df.to_parquet(out_dir / "dataset.parquet")

    # Table S1
    table_s1 = composition_table(df, "label", sens)
    table_s1.to_csv(out_dir / "table_s1_composition.csv")
    log.info(f"Table S1 written:\n{table_s1}")

    # Leakage audit -> LEAKAGE_AUDIT.md
    findings = leakage_audit(
        df, splits,
        image_col=dcfg["image_col"],
        patient_col=dcfg.get("patient_col"),
        image_root=Path(dcfg["raw_dir"]),
        out_path=out_dir / "LEAKAGE_AUDIT.md",
    )
    log.info(f"Leakage audit: {findings}")
    log.info(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
