"""Cluster bootstrap for ranking metrics (CONTEXT 8.3, `configs/split.yaml` > bootstrap).

Steps of a trajectory, and trajectories of a template, are dependent: a confidence interval resamples
templates (clusters), never steps. `iid_bootstrap` resamples steps independently; it exists only to show
how much too narrow such an interval would be (design effect).

Vectorised: scores are grouped into buckets of tied values, and per template the positives and negatives of
each bucket are counted once. A cluster resample is a vector of template multiplicities, so the bucket counts
of all resamples are a single matrix product, and every metric below is read off those counts:

    roc_auc         P(score+ > score-) + 0.5 P(tie), as sklearn's roc_auc_score
    pr_auc          average precision with tied scores as one threshold, as sklearn's average_precision_score
    prevalence      positives / rows
    recall@<name>   share of positives with score > threshold[name]
    fpr@<name>      share of negatives with score > threshold[name]

Thresholds are given, never fitted here: they come from training or validation data (CONTEXT 7.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BucketCounts:
    """Positives and negatives per (cluster, tied-score bucket); buckets sorted by decreasing score."""
    pos: np.ndarray      # (n_clusters, n_buckets)
    neg: np.ndarray      # (n_clusters, n_buckets)
    scores: np.ndarray   # (n_buckets,) decreasing
    clusters: np.ndarray  # (n_clusters,) cluster labels


def bucket_counts(score, y, groups) -> BucketCounts:
    score, y = np.asarray(score, dtype=float), np.asarray(y).astype(int)
    if np.isnan(score).any():
        raise ValueError("scores contain NaN")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary 0 / 1")
    uniq, b = np.unique(-score, return_inverse=True)  # ascending in -score = decreasing score
    g, clusters = pd.factorize(pd.Series(groups).astype(str), sort=True)
    nb, ng = len(uniq), len(clusters)
    pos = np.bincount(g * nb + b, weights=y, minlength=ng * nb).reshape(ng, nb)
    neg = np.bincount(g * nb + b, weights=1 - y, minlength=ng * nb).reshape(ng, nb)
    return BucketCounts(pos, neg, -uniq, np.asarray(clusters))


def metrics_from_counts(pos: np.ndarray, neg: np.ndarray, scores: np.ndarray,
                        thresholds: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """Metrics for each row of `pos` / `neg` (shape (R, n_buckets), buckets sorted by decreasing score)."""
    pos, neg = np.atleast_2d(pos).astype(float), np.atleast_2d(neg).astype(float)
    P, N = pos.sum(1), neg.sum(1)
    tp, fp = pos.cumsum(1), neg.cumsum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "roc_auc": (neg * (tp - pos + 0.5 * pos)).sum(1) / (P * N),
            "pr_auc": (pos * np.where(tp + fp > 0, tp / (tp + fp), 0.0)).sum(1) / P,
            "prevalence": P / (P + N),
        }
        for name, thr in (thresholds or {}).items():
            alert = scores > thr
            out[f"recall@{name}"] = pos[:, alert].sum(1) / P
            out[f"fpr@{name}"] = neg[:, alert].sum(1) / N
    return out


def point_estimate(score, y, groups=None, thresholds: dict[str, float] | None = None) -> dict[str, float]:
    c = bucket_counts(score, y, np.zeros(len(score)) if groups is None else groups)
    m = metrics_from_counts(c.pos.sum(0), c.neg.sum(0), c.scores, thresholds)
    return {k: float(v[0]) for k, v in m.items()}


def cluster_bootstrap(score, y, groups, thresholds: dict[str, float] | None = None, n_resamples: int = 2000,
                      size: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Resample clusters with replacement (`size` draws, default: as many as there are clusters).
    A `size` other than the number of clusters estimates the spread of the metric on a sample of `size`
    clusters drawn from the same population (e.g. a test set of a given number of templates)."""
    c = bucket_counts(score, y, groups)
    ng = len(c.clusters)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(size or ng, np.full(ng, 1 / ng), size=n_resamples)  # (R, n_clusters)
    return pd.DataFrame(metrics_from_counts(w @ c.pos, w @ c.neg, c.scores, thresholds))


def iid_bootstrap(score, y, thresholds: dict[str, float] | None = None, n_resamples: int = 2000,
                  seed: int = 42) -> pd.DataFrame:
    """Resample rows independently (NOT valid for steps; for the design-effect comparison only).
    Rows in the same (bucket, class) cell are interchangeable for these metrics, so a row resample is a
    multinomial draw over the cells."""
    c = bucket_counts(score, y, np.zeros(len(score)))
    cells = np.concatenate([c.pos[0], c.neg[0]])
    rng = np.random.default_rng(seed)
    draw = rng.multinomial(int(cells.sum()), cells / cells.sum(), size=n_resamples)
    nb = len(c.scores)
    return pd.DataFrame(metrics_from_counts(draw[:, :nb], draw[:, nb:], c.scores, thresholds))


def interval(resamples: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Percentile interval per metric: columns low, high, width (NaN resamples, e.g. no positive, dropped)."""
    a = (1 - level) / 2
    q = resamples.quantile([a, 1 - a]).T
    q.columns = ["low", "high"]
    return q.assign(width=q["high"] - q["low"])
