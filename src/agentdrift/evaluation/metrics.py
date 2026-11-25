"""Calibration metrics: Brier score, expected calibration error (ECE), reliability table.

Written by Lot B for the v0.1 calibration study (KAN-20, Step 2.3); Lot D (KAN-28 / KAN-29) owns this module and
extends it with the first-alert metrics. Ranking metrics (PR-AUC, ROC-AUC, recall at a fixed FPR) live in
`evaluation/bootstrap.py`.

Why these and how the bins are chosen:
    brier   mean squared error of the probability: rewards calibration AND discrimination; its reference is the
            constant model at the prevalence (Brier = p (1 - p): 0.105 for the scanner, 0.143 for the gate).
    ece     mean |observed rate - mean predicted probability| over bins, weighted by bin size. Bins have EQUAL MASS
            (quantiles of the predicted probability), not equal width: with ~10 % positives most scores sit near 0,
            and equal-width bins would leave the high-score bins nearly empty (one or two rows, pure noise) while
            pooling most rows in the first bin. Ten bins of ~1,000 rows per fold keep the standard error of each
            observed rate below 0.016 (worst case p = 0.5). Ties at a bin edge stay in one bin (edges are unique).
            The ECE depends on the binning: compare models with the SAME `n_bins` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BINS = 10


def _check(prob, y) -> tuple[np.ndarray, np.ndarray]:
    prob, y = np.asarray(prob, dtype=float), np.asarray(y).astype(int)
    if prob.shape != y.shape or prob.ndim != 1:
        raise ValueError("prob and y must be 1-D arrays of the same length")
    if np.isnan(prob).any() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("calibration metrics need probabilities in [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary 0 / 1")
    return prob, y


def is_probability(score) -> bool:
    s = np.asarray(score, dtype=float)
    return bool(len(s)) and bool(np.isfinite(s).all()) and s.min() >= 0 and s.max() <= 1


def brier_score(prob, y) -> float:
    prob, y = _check(prob, y)
    return float(np.mean((prob - y) ** 2))


def reliability_table(prob, y, n_bins: int = DEFAULT_BINS) -> pd.DataFrame:
    """Equal-mass bins: n, mean predicted probability, observed positive rate, and their gap, per bin."""
    prob, y = _check(prob, y)
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:  # constant score: one bin
        edges = np.array([prob.min(), prob.max()])
    b = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    t = (pd.DataFrame({"bin": b, "prob": prob, "y": y}).groupby("bin")
         .agg(n=("y", "size"), mean_prob=("prob", "mean"), observed=("y", "mean")).reset_index())
    t["lower"], t["upper"] = edges[t["bin"]], edges[t["bin"] + 1]
    t["gap"] = t["observed"] - t["mean_prob"]
    return t[["bin", "lower", "upper", "n", "mean_prob", "observed", "gap"]]


def expected_calibration_error(prob, y, n_bins: int = DEFAULT_BINS) -> float:
    t = reliability_table(prob, y, n_bins)
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())


def calibration_metrics(prob, y, n_bins: int = DEFAULT_BINS) -> dict[str, float]:
    return {"brier": brier_score(prob, y), "ece": expected_calibration_error(prob, y, n_bins)}
