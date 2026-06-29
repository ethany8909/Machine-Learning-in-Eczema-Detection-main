"""Unit tests for the fairness engine — verifying the math is correct."""
import numpy as np

from dermafair.fairness import (
    FairnessEvaluator,
    _binary_rates,
    bootstrap_gap_ci,
    fairness_gaps,
    kruskal_wallis,
    per_group_metrics,
)


def test_binary_rates_perfect():
    y = np.array([0, 1, 0, 1])
    r = _binary_rates(y, y)
    assert r["accuracy"] == 1.0
    assert r["sensitivity"] == 1.0
    assert r["specificity"] == 1.0


def test_binary_rates_known():
    # TP=1, FN=1, TN=1, FP=1
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    r = _binary_rates(y_true, y_pred)
    assert r["accuracy"] == 0.5
    assert r["sensitivity"] == 0.5
    assert r["specificity"] == 0.5
    assert r["fpr"] == 0.5


def test_per_group_and_gap():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    # group A perfect, group B all wrong
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    gm = per_group_metrics(y_true, y_pred, groups)
    assert gm["A"]["accuracy"] == 1.0
    assert gm["B"]["accuracy"] == 0.0
    gaps = fairness_gaps(gm)
    assert gaps["max_accuracy_gap"] == 1.0
    assert gaps["overall_accuracy"] == 0.5
    # fairness_score = 1 - (1.0 / 0.5) = -1.0
    assert abs(gaps["fairness_score"] - (-1.0)) < 1e-9


def test_kruskal_runs():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = y_true.copy()
    y_pred[100:] = 1 - y_pred[100:]  # second group worse
    groups = np.array(["A"] * 100 + ["B"] * 100)
    h, p = kruskal_wallis(y_true, y_pred, groups)
    assert not np.isnan(h)
    assert 0.0 <= p <= 1.0


def test_bootstrap_ci_bounds():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 120)
    y_pred = y_true.copy()
    groups = rng.choice(["A", "B", "C"], size=120)
    ci = bootstrap_gap_ci(y_true, y_pred, groups, n_bootstrap=200)
    assert ci["ci_low"] <= ci["ci_high"]


def test_evaluator_end_to_end():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 300)
    y_pred = y_true.copy()
    flip = rng.random(300) < 0.2
    y_pred[flip] = 1 - y_pred[flip]
    groups = rng.choice([3, 4, 5, 6], size=300)
    report = FairnessEvaluator("fitzpatrick").evaluate(
        y_true, y_pred, groups, n_bootstrap=200
    )
    row = report.to_row("test_model")
    assert "fairness_score" in row
    assert "acc_gap_ci_low" in row
    assert 0.0 <= report.overall_accuracy <= 1.0
