"""Split protocol (CONTEXT 8.1, leak L1): grouped by template, test untouched, reproducible, balanced.

A synthetic test runs everywhere; corpus tests are skipped until data/interim/ exists.
"""

import numpy as np
import pandas as pd
import pytest

from agentdrift.config import INTERIM
from agentdrift.splits.grouped import (
    TEST_FOLD, attach_folds, folds_hash, inner_splits, lodo_splits, make_folds, outer_splits,
)


def _synthetic(n_templates=60, per_template=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_templates):
        dom = ["email", "banking", "web"][k % 3]
        split = "test" if k < 10 else "train"
        for j in range(per_template):
            rows.append({
                "traj_id": f"{dom}_{k}_{j}", "origin_split": split, "template_id": f"{dom}_{k:03d}", "domain": dom,
                "world_id": f"w{rng.integers(40)}", "source_category": ["benign", "attacked_full"][j % 2],
            })
    return pd.DataFrame(rows)


def test_synthetic_grouping_and_test_isolation():
    traj = _synthetic()
    f = make_folds(traj)
    assert (f.loc[f["origin_split"] == "test", "fold"] == TEST_FOLD).all()
    assert (f.loc[f["origin_split"] != "test", "fold"] >= 0).all()
    assert (f[f["fold"] >= 0].groupby("template_id")["fold"].nunique() == 1).all()
    assert (f[f["world_fold"] >= 0].groupby("world_id")["world_fold"].nunique() == 1).all()
    assert folds_hash(f) == folds_hash(make_folds(traj))  # reproducible with the configured seed


def test_outer_and_lodo_splits_never_yield_test_rows():
    f = make_folds(_synthetic())
    test_pos = set(np.flatnonzero(f["fold"] == TEST_FOLD))
    dev_pos = set(np.flatnonzero(f["fold"] != TEST_FOLD))
    for _, tr, va in outer_splits(f):
        assert not set(tr) & set(va)
        assert set(tr) | set(va) == dev_pos
        assert not (set(tr) | set(va)) & test_pos
    for dom, tr, va in lodo_splits(f):
        assert set(f.iloc[va]["domain"]) == {dom} and dom not in set(f.iloc[tr]["domain"])
        assert not (set(tr) | set(va)) & test_pos


def test_inner_splits_are_grouped_by_template():
    f = make_folds(_synthetic())
    _, tr, _ = next(outer_splits(f))
    part = f.iloc[tr].reset_index(drop=True)
    for itr, iva in inner_splits(part):
        assert set(part.iloc[itr]["template_id"]).isdisjoint(part.iloc[iva]["template_id"])


# ---------------------------------------------------------------- corpus

needs_interim = pytest.mark.skipif(
    not (INTERIM / "trajectories.parquet").exists(), reason="run `python -m agentdrift.data.load` first"
)


@pytest.fixture(scope="module")
def corpus():
    from agentdrift.data.load import read_steps, read_trajectories

    traj = read_trajectories(include_test=True)
    return traj, read_steps(include_test=True), make_folds(traj)


@needs_interim
def test_corpus_l1_no_template_across_folds_or_test(corpus):
    traj, _, f = corpus
    dev = f[f["fold"] != TEST_FOLD]
    assert (dev.groupby("template_id")["fold"].nunique() == 1).all()
    assert set(dev["template_id"]).isdisjoint(f.loc[f["fold"] == TEST_FOLD, "template_id"])
    assert dev["template_id"].nunique() == 215 and len(f[f["fold"] == TEST_FOLD]) == 1722


@needs_interim
def test_corpus_folds_are_balanced(corpus):
    traj, steps, f = corpus
    s = attach_folds(steps, f)
    dev = s[s["fold"] != TEST_FOLD]
    rate = dev.groupby("fold")["y_gate"].mean()
    assert (rate - dev["y_gate"].mean()).abs().max() < 0.02
    n_tpl = f[f["fold"] != TEST_FOLD].groupby("fold")["template_id"].nunique()
    assert n_tpl.min() >= 0.8 * n_tpl.mean() and n_tpl.max() <= 1.2 * n_tpl.mean()
    # every source category is present in every fold
    t = traj.merge(f[["traj_id", "fold"]], on="traj_id")
    assert (pd.crosstab(t["fold"], t["source_category"]) > 0).all().all()


@needs_interim
def test_corpus_steps_inherit_their_trajectory_fold(corpus):
    _, steps, f = corpus
    s = attach_folds(steps, f)
    assert s["fold"].notna().all()
    assert (s.groupby("traj_id")["fold"].nunique() == 1).all()
