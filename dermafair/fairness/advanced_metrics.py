"""Advanced evaluation metrics used in recent fairness / medical-imaging studies.

This module complements ``dermafair.fairness`` (which covers per-group accuracy/TPR/FPR
gaps, Kruskal-Wallis, and bootstrap CIs) with the probability-aware and group-fairness
metrics that reviewers in this space now expect:

Overall (probability-aware):
  - AUROC, AUPRC (average precision)
  - Balanced accuracy, Matthews correlation coefficient (MCC), Cohen's kappa
  - Brier score, Expected/Maximum Calibration Error (ECE/MCE)

Group fairness (Hardt et al. 2016; Sagawa et al. 2020 worst-group; Fairlearn conventions):
  - Equal Opportunity Difference   (max across-group TPR gap)
  - Equalized Odds Difference       (max of TPR-gap and FPR-gap)
  - Demographic Parity Difference   (max gap in positive-prediction rate)
  - Predictive Parity Difference    (max gap in precision / PPV)
  - Worst-group accuracy / F1 / AUROC
  - AUROC gap across groups
  - Between-group accuracy std and coefficient of variation

Convention: positive class == 1. With the ImageFolder ordering used here that is
``psoriasis_lichenoid`` (eczema_dermatitis == 0). ``y_prob`` may be passed either as
the [N, 2] softmax matrix or the [N] positive-class probability vector.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _pos_prob(y_prob) -> np.ndarray:
    """Return the positive-class (class 1) probability vector from [N,2] or [N]."""
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 2:
        return y_prob[:, 1]
    return y_prob


def _safe_auroc(y_true, p_pos) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:  # AUROC undefined with one class present
        return float("nan")
    return float(roc_auc_score(y_true, p_pos))


def _safe_auprc(y_true, p_pos) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, p_pos))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(y_true, p_pos, n_bins: int = 10) -> dict[str, float]:
    """Expected and Maximum Calibration Error using equal-width confidence bins.

    Confidence is the predicted probability of the argmax class. Returns
    {"ece", "mce", "brier"}.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pos = _pos_prob(p_pos)
    y_pred = (p_pos >= 0.5).astype(int)
    confidence = np.where(y_pred == 1, p_pos, 1.0 - p_pos)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, n = 0.0, 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        # last bin is closed on the right so confidence == 1.0 is included
        in_bin = (confidence > lo) & (confidence <= hi) if hi < 1.0 else (confidence > lo) & (confidence <= hi + 1e-9)
        m = in_bin.sum()
        if m == 0:
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = confidence[in_bin].mean()
        gap = abs(acc_bin - conf_bin)
        ece += (m / n) * gap
        mce = max(mce, gap)

    brier = float(brier_score_loss(y_true, p_pos)) if len(np.unique(y_true)) > 0 else float("nan")
    return {"ece": float(ece), "mce": float(mce), "brier": brier}


# --------------------------------------------------------------------------- #
# Overall metrics
# --------------------------------------------------------------------------- #
def overall_metrics(y_true, y_pred, y_prob, n_bins: int = 10) -> dict[str, float]:
    """Probability-aware overall metrics for a binary classifier."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p_pos = _pos_prob(y_prob)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    cal = expected_calibration_error(y_true, p_pos, n_bins=n_bins)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": _safe_auroc(y_true, p_pos),
        "auprc": _safe_auprc(y_true, p_pos),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "brier": cal["brier"],
        "ece": cal["ece"],
        "mce": cal["mce"],
    }


# --------------------------------------------------------------------------- #
# Group fairness metrics
# --------------------------------------------------------------------------- #
def _rates(y_true, y_pred, p_pos):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + tn + fp + fn
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "selection_rate": (tp + fp) / n if n else float("nan"),
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
        "auroc": _safe_auroc(y_true, p_pos),
    }


def _spread(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return {"gap": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"gap": float(max(vals) - min(vals)), "min": float(min(vals)), "max": float(max(vals))}


def group_fairness_metrics(y_true, y_pred, y_prob, groups) -> dict:
    """Per-group table plus summary group-fairness metrics.

    Returns {"per_group": {g: {...}}, "summary": {...}}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_pos = _pos_prob(y_prob)
    groups = np.asarray(groups)

    per_group = {}
    for g in sorted(np.unique(groups), key=lambda x: str(x)):
        mask = groups == g
        per_group[str(g)] = _rates(y_true[mask], y_pred[mask], p_pos[mask])

    accs = [m["accuracy"] for m in per_group.values()]
    tprs = [m["tpr"] for m in per_group.values()]
    fprs = [m["fpr"] for m in per_group.values()]
    ppvs = [m["ppv"] for m in per_group.values()]
    sels = [m["selection_rate"] for m in per_group.values()]
    f1s = [m["f1"] for m in per_group.values()]
    aucs = [m["auroc"] for m in per_group.values()]

    tpr_gap = _spread(tprs)["gap"]
    fpr_gap = _spread(fprs)["gap"]
    valid_accs = [a for a in accs if not np.isnan(a)]
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    valid_f1 = [a for a in f1s if not np.isnan(a)]

    summary = {
        "equal_opportunity_diff": tpr_gap,                       # TPR gap
        "equalized_odds_diff": float(np.nanmax([tpr_gap, fpr_gap]))
        if not (np.isnan(tpr_gap) and np.isnan(fpr_gap)) else float("nan"),
        "demographic_parity_diff": _spread(sels)["gap"],          # selection-rate gap
        "predictive_parity_diff": _spread(ppvs)["gap"],           # PPV gap
        "accuracy_gap": _spread(accs)["gap"],
        "auroc_gap": _spread(aucs)["gap"],
        "worst_group_accuracy": float(min(valid_accs)) if valid_accs else float("nan"),
        "worst_group_auroc": float(min(valid_aucs)) if valid_aucs else float("nan"),
        "worst_group_f1": float(min(valid_f1)) if valid_f1 else float("nan"),
        "accuracy_std": float(np.std(valid_accs)) if len(valid_accs) > 1 else float("nan"),
        "accuracy_cv": float(np.std(valid_accs) / np.mean(valid_accs))
        if len(valid_accs) > 1 and np.mean(valid_accs) else float("nan"),
    }
    return {"per_group": per_group, "summary": summary}


def evaluate_all(y_true, y_pred, y_prob, groups=None, n_bins: int = 10) -> dict:
    """One-call bundle: overall metrics + (optional) group fairness metrics."""
    result = {"overall": overall_metrics(y_true, y_pred, y_prob, n_bins=n_bins)}
    if groups is not None:
        result["fairness"] = group_fairness_metrics(y_true, y_pred, y_prob, groups)
    return result
