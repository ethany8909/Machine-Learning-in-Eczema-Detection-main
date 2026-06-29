"""Fairness evaluation engine.

Framework-agnostic: operates on prediction arrays, not models. Computes per-group
performance, fairness gaps, a normalized fairness score, Kruskal-Wallis significance,
and bootstrapped confidence intervals.

Example
-------
>>> from dermafair.fairness import FairnessEvaluator
>>> ev = FairnessEvaluator(sensitive_attr="fitzpatrick")
>>> report = ev.evaluate(y_true, y_pred, groups, n_bootstrap=1000)
>>> print(report.fairness_score, report.max_gap, report.kruskal_p)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import kruskal


# --------------------------------------------------------------------------- #
# Per-group metric primitives
# --------------------------------------------------------------------------- #
def _binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, sensitivity (TPR), specificity (TNR), precision, F1 for binary labels.

    Positive class is 1. Returns NaN for undefined rates (empty denominators).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else np.nan
    sens = tp / (tp + fn) if (tp + fn) else np.nan          # TPR / recall
    spec = tn / (tn + fp) if (tn + fp) else np.nan          # TNR
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (
        2 * prec * sens / (prec + sens)
        if (prec is not np.nan and sens is not np.nan and (prec + sens) > 0)
        else np.nan
    )
    fpr = fp / (fp + tn) if (fp + tn) else np.nan

    return {
        "n": n,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "tpr": sens,
        "fpr": fpr,
    }


def per_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict:
    """Compute binary metrics for each unique group value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    out: dict = {}
    for g in sorted(np.unique(groups), key=lambda x: str(x)):
        mask = groups == g
        out[g] = _binary_rates(y_true[mask], y_pred[mask])
    return out


# --------------------------------------------------------------------------- #
# Gap + fairness-score computation
# --------------------------------------------------------------------------- #
def _gap(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return np.nan
    return float(max(vals) - min(vals))


def fairness_gaps(group_metrics: dict) -> dict[str, float]:
    """Max gaps across groups for accuracy, TPR, FPR; plus overall accuracy."""
    accs = [m["accuracy"] for m in group_metrics.values()]
    tprs = [m["tpr"] for m in group_metrics.values()]
    fprs = [m["fpr"] for m in group_metrics.values()]

    # sample-size-weighted overall accuracy
    total_n = sum(m["n"] for m in group_metrics.values())
    overall_acc = (
        sum(m["accuracy"] * m["n"] for m in group_metrics.values() if not np.isnan(m["accuracy"]))
        / total_n
        if total_n
        else np.nan
    )

    max_acc_gap = _gap(accs)
    fairness_score = (
        1.0 - (max_acc_gap / overall_acc)
        if (not np.isnan(max_acc_gap) and overall_acc and not np.isnan(overall_acc))
        else np.nan
    )

    return {
        "overall_accuracy": overall_acc,
        "max_accuracy_gap": max_acc_gap,
        "tpr_gap": _gap(tprs),
        "fpr_gap": _gap(fprs),
        "fairness_score": fairness_score,
    }


# --------------------------------------------------------------------------- #
# Significance testing
# --------------------------------------------------------------------------- #
def kruskal_wallis(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    """Kruskal-Wallis test on per-sample correctness across groups.

    Tests whether the distribution of correct/incorrect predictions differs by
    group. Returns (H statistic, p-value). NaN if <2 groups have data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)

    correctness_by_group = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        correct = (y_pred[mask] == y_true[mask]).astype(float)
        correctness_by_group.append(correct)

    if len(correctness_by_group) < 2:
        return np.nan, np.nan
    # The test is only ill-defined when *every* value in *every* group is identical
    # (zero total variance). Groups that are each internally constant but differ
    # from one another are valid and informative.
    pooled = np.concatenate(correctness_by_group)
    if np.all(pooled == pooled[0]):
        return np.nan, 1.0
    try:
        h, p = kruskal(*correctness_by_group)
        return float(h), float(p)
    except ValueError:
        return np.nan, np.nan


# --------------------------------------------------------------------------- #
# Bootstrapped confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_gap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric: str = "accuracy",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for the max across-group gap of a metric.

    Essential for small-N fairness work: a single point estimate of a gap is
    not credible without an interval.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    n = len(y_true)

    gaps = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gm = per_group_metrics(y_true[idx], y_pred[idx], groups[idx])
        vals = [m[metric] for m in gm.values()]
        gaps.append(_gap(vals))

    gaps = np.array([g for g in gaps if not np.isnan(g)])
    if gaps.size == 0:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    point = _gap([m[metric] for m in per_group_metrics(y_true, y_pred, groups).values()])
    lo = float(np.percentile(gaps, 100 * alpha / 2))
    hi = float(np.percentile(gaps, 100 * (1 - alpha / 2)))
    return {"point": float(point), "ci_low": lo, "ci_high": hi}


# --------------------------------------------------------------------------- #
# Report container + evaluator
# --------------------------------------------------------------------------- #
@dataclass
class FairnessReport:
    """Structured results for a single model's fairness evaluation."""

    sensitive_attr: str
    group_metrics: dict
    overall_accuracy: float
    max_gap: float
    tpr_gap: float
    fpr_gap: float
    fairness_score: float
    kruskal_h: float
    kruskal_p: float
    bootstrap: dict = field(default_factory=dict)

    @property
    def is_fair(self, threshold: float = 0.05) -> bool:
        """Heuristic: no statistically significant gap at alpha=0.05."""
        if np.isnan(self.kruskal_p):
            return False
        return self.kruskal_p >= threshold

    def to_row(self, model_name: str) -> dict:
        """Flatten into a single dict row for a results table."""
        row = {
            "model": model_name,
            "overall_accuracy": self.overall_accuracy,
            "max_accuracy_gap": self.max_gap,
            "tpr_gap": self.tpr_gap,
            "fpr_gap": self.fpr_gap,
            "fairness_score": self.fairness_score,
            "kruskal_h": self.kruskal_h,
            "kruskal_p": self.kruskal_p,
            "significant_gap": (not self.is_fair),
        }
        if "accuracy" in self.bootstrap:
            b = self.bootstrap["accuracy"]
            row["acc_gap_ci_low"] = b["ci_low"]
            row["acc_gap_ci_high"] = b["ci_high"]
        return row


class FairnessEvaluator:
    """Top-level entry point for fairness evaluation of one model's predictions."""

    def __init__(self, sensitive_attr: str = "fitzpatrick"):
        self.sensitive_attr = sensitive_attr

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        groups: Sequence,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> FairnessReport:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)

        gm = per_group_metrics(y_true, y_pred, groups)
        gaps = fairness_gaps(gm)
        h, p = kruskal_wallis(y_true, y_pred, groups)

        boot = {}
        if n_bootstrap and n_bootstrap > 0:
            boot["accuracy"] = bootstrap_gap_ci(
                y_true, y_pred, groups, "accuracy", n_bootstrap, alpha, seed
            )

        return FairnessReport(
            sensitive_attr=self.sensitive_attr,
            group_metrics=gm,
            overall_accuracy=gaps["overall_accuracy"],
            max_gap=gaps["max_accuracy_gap"],
            tpr_gap=gaps["tpr_gap"],
            fpr_gap=gaps["fpr_gap"],
            fairness_score=gaps["fairness_score"],
            kruskal_h=h,
            kruskal_p=p,
            bootstrap=boot,
        )
