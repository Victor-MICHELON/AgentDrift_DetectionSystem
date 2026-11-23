"""Cluster bootstrap: metrics read off bucket counts must equal sklearn's, resamples must move whole clusters."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from agentdrift.evaluation.bootstrap import (
    bucket_counts,
    cluster_bootstrap,
    iid_bootstrap,
    interval,
    metrics_from_counts,
    point_estimate,
)


def _toy(n=600, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = y * 0.8 + rng.normal(size=n)
    if ties:
        s = np.round(s, 1)  # many tied scores, like a score from categorical features
    groups = rng.integers(0, 25, n)
    return s, y, groups


@pytest.mark.parametrize("ties", [True, False])
def test_point_estimate_matches_sklearn(ties):
    s, y, g = _toy(ties=ties)
    m = point_estimate(s, y, g, thresholds={"t": 0.5})
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))
    assert m["recall@t"] == pytest.approx(((s > 0.5) & (y == 1)).sum() / y.sum())
    assert m["fpr@t"] == pytest.approx(((s > 0.5) & (y == 0)).sum() / (y == 0).sum())


def test_a_cluster_resample_equals_the_metric_on_the_resampled_rows():
    s, y, g = _toy()
    c = bucket_counts(s, y, g)
    w = np.bincount([0, 3, 3, 7], minlength=len(c.clusters))
    m = metrics_from_counts(w @ c.pos, w @ c.neg, c.scores)
    rows = np.concatenate([np.flatnonzero(pd.Series(g).astype(str) == c.clusters[k]) for k in [0, 3, 3, 7]])
    assert m["roc_auc"][0] == pytest.approx(roc_auc_score(y[rows], s[rows]))
    assert m["pr_auc"][0] == pytest.approx(average_precision_score(y[rows], s[rows]))


def test_bootstrap_is_seeded_and_sized():
    s, y, g = _toy()
    a = cluster_bootstrap(s, y, g, n_resamples=50, seed=3)
    pd.testing.assert_frame_equal(a, cluster_bootstrap(s, y, g, n_resamples=50, seed=3))
    assert len(a) == 50 and set(a.columns) == {"roc_auc", "pr_auc", "prevalence"}
    ci = interval(a)
    assert (ci["low"] <= ci["high"]).all() and (ci["width"] >= 0).all()


def test_fewer_clusters_give_a_wider_interval_and_clustering_widens_it():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 50)
    effect = rng.normal(scale=1.0, size=40)[g]  # strong cluster effect on the score
    y = rng.integers(0, 2, len(g))
    s = y + effect + rng.normal(scale=0.5, size=len(g))
    full = interval(cluster_bootstrap(s, y, g, n_resamples=400)).loc["roc_auc", "width"]
    small = interval(cluster_bootstrap(s, y, g, n_resamples=400, size=10)).loc["roc_auc", "width"]
    naive = interval(iid_bootstrap(s, y, n_resamples=400)).loc["roc_auc", "width"]
    assert small > full > naive


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        bucket_counts([0.1, np.nan], [0, 1], [0, 0])
    with pytest.raises(ValueError, match="binary"):
        bucket_counts([0.1, 0.2], [0, 2], [0, 0])
