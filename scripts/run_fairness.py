#!/usr/bin/env python
"""Step 3 of the pipeline: fairness evaluation + figures + report.

Consumes saved model predictions (written by train_all.py) of the form:
    results/<run>/predictions/<model>.npz   with keys y_true, y_pred, fitzpatrick
                                            and optionally gate_weights [N,2]

Produces the master fairness table, all figures, and fairness_report.md.

Usage:
    python scripts/run_fairness.py --config configs/dermacon.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dermafair.fairness import FairnessEvaluator
from dermafair.utils import get_logger, load_config, set_seed
from dermafair.visualization import (
    accuracy_fairness_pareto,
    compile_report,
    fairness_heatmap,
    fusion_comparison,
    gate_weights_by_tone,
)

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    run_dir = Path(cfg["output"]["dir"]) / cfg["run_name"]
    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        raise SystemExit(f"No predictions found at {pred_dir}. Run train_all.py first.")

    evaluator = FairnessEvaluator(sensitive_attr=cfg["data"]["sensitive_attr"])
    rows = []
    gate_weights_df = None

    for npz_path in sorted(pred_dir.glob("*.npz")):
        model_name = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)
        report = evaluator.evaluate(
            data["y_true"], data["y_pred"], data["fitzpatrick"],
            n_bootstrap=cfg["fairness"]["n_bootstrap"],
            alpha=cfg["fairness"]["alpha"],
            seed=cfg["seed"],
        )
        rows.append(report.to_row(model_name))
        log.info(f"{model_name}: acc={report.overall_accuracy:.3f} "
                 f"fairness={report.fairness_score:.3f} p={report.kruskal_p:.3f}")

        # capture gate weights for the signature figure
        if "gate_weights" in data.files and model_name == "gate_network":
            gw = data["gate_weights"]  # [N, 2]
            gate_weights_df = pd.DataFrame({
                "fitzpatrick": data["fitzpatrick"],
                "w_image": gw[:, 0],
                "w_metadata": gw[:, 1],
            })

    master = pd.DataFrame(rows)
    tables_dir = run_dir / "tables"
    figs_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(tables_dir / "fairness_master.csv", index=False)
    log.info(f"Master table written ({len(master)} models)")

    # Figures
    fairness_heatmap(master, figs_dir / "fairness_heatmap.png")
    accuracy_fairness_pareto(master, figs_dir / "accuracy_fairness_pareto.png")
    fusion_comparison(master, figs_dir / "fusion_comparison.png")
    if gate_weights_df is not None:
        gate_weights_df.to_csv(tables_dir / "gate_weights_by_sample.csv", index=False)
        gate_weights_by_tone(gate_weights_df, figs_dir / "gate_weights_by_tone.png")
        # quick correlation: does metadata weight rise with Fitzpatrick band?
        corr = np.corrcoef(
            gate_weights_df["fitzpatrick"].astype(float),
            gate_weights_df["w_metadata"].astype(float),
        )[0, 1]
        log.info(f"Corr(Fitzpatrick band, metadata weight) = {corr:.3f}")

    compile_report(master, run_dir / "fairness_report.md", cfg["run_name"])
    log.info(f"All outputs in {run_dir}")


if __name__ == "__main__":
    main()
