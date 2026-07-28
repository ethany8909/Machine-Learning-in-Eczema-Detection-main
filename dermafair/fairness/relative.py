"""Relative fairness comparison via PAIRED bootstrap.

Small per-tone cells make absolute significance testing hopeless (everything is
p>0.05). Instead of asking "is model X's gap significant?", this asks the
answerable RELATIVE question: across bootstrap resamples of the shared test set,
how often does model A have a SMALLER fairness gap than model B?

Because every model is evaluated on the same test patients, we resample patient
indices ONCE per bootstrap iteration and apply them to every model (paired) — this
cancels shared test-set variance and isolates the between-model difference.

Yields claims of the form "model A is fairer than B with probability 0.87 on the
accuracy gap, consistently across accuracy/TPR/FPR" — robust to low statistical
power, which is the honest framing for small-N fairness work.
"""
from __future__ import annotations

import numpy as np

METRICS = ("accuracy", "tpr", "fpr")


def _group_gap(y_true, y_pred, groups, metric):
    """Max-minus-min of a per-group rate across groups (NaN if <2 usable groups)."""
    vals = []
    for g in np.unique(groups):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        tp = int(np.sum((yp == 1) & (yt == 1)))
        tn = int(np.sum((yp == 0) & (yt == 0)))
        fp = int(np.sum((yp == 1) & (yt == 0)))
        fn = int(np.sum((yp == 0) & (yt == 1)))
        if metric == "accuracy":
            n = tp + tn + fp + fn
            v = (tp + tn) / n if n else np.nan
        elif metric == "tpr":
            v = tp / (tp + fn) if (tp + fn) else np.nan
        elif metric == "fpr":
            v = fp / (fp + tn) if (fp + tn) else np.nan
        else:
            raise ValueError(metric)
        if not np.isnan(v):
            vals.append(v)
    return (max(vals) - min(vals)) if len(vals) >= 2 else np.nan


def paired_bootstrap_gaps(models, metric="accuracy", n_boot=2000, seed=42):
    """models: {name: (y_true, y_pred, groups)} all aligned to the same test set.

    Returns {name: ndarray[n_boot]} of the metric's gap under shared resamples.
    """
    rng = np.random.default_rng(seed)
    names = list(models)
    n = len(next(iter(models.values()))[0])
    dist = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)                      # shared across models (paired)
        for name, (yt, yp, g) in models.items():
            dist[name][b] = _group_gap(yt[idx], yp[idx], g[idx], metric)
    return dist


def prob_fairer(dist, a, b):
    """P(gap_a < gap_b) over paired resamples (ignoring NaN pairs)."""
    ga, gb = dist[a], dist[b]
    mask = ~(np.isnan(ga) | np.isnan(gb))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(ga[mask] < gb[mask]))


def gap_estimate(dist, name, alpha=0.05):
    """Median gap + percentile CI for one model (the estimation view)."""
    g = dist[name][~np.isnan(dist[name])]
    if g.size == 0:
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "median": float(np.median(g)),
        "ci_low": float(np.percentile(g, 100 * alpha / 2)),
        "ci_high": float(np.percentile(g, 100 * (1 - alpha / 2))),
    }


def relative_report(models, reference, metrics=METRICS, n_boot=2000, seed=42):
    """For each model vs ``reference``: P(fairer) per metric + cross-metric consistency.

    Consistency = mean over metrics of P(model's gap < reference's gap). A model
    that is fairer than the reference on every metric and every resample scores 1.0.
    """
    dists = {m: paired_bootstrap_gaps(models, m, n_boot, seed) for m in metrics}
    out = {}
    for name in models:
        if name == reference:
            continue
        per_metric = {m: prob_fairer(dists[m], name, reference) for m in metrics}
        valid = [p for p in per_metric.values() if not np.isnan(p)]
        out[name] = {
            "prob_fairer": per_metric,
            "consistency": float(np.mean(valid)) if valid else float("nan"),
            "gap_estimate": {m: gap_estimate(dists[m], name) for m in metrics},
        }
    out["_reference_gap_estimate"] = {m: gap_estimate(dists[m], reference) for m in metrics}
    return out
