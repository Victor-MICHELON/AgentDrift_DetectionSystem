"""Experiment harness (models/experiment.py, KAN-61): the leakage rules, each with an automatic test.

L2  a spy estimator records every row it is fitted on: always inside the training part of its split, never a row
    or a template of the evaluated part (main fit and the inner fits used for thresholds).
L5  flipping the labels of an evaluated fold changes nothing in that fold's scores and thresholds: neither the
    model nor the threshold has seen them. The fixed test set is refused.
L12 non-feature and metadata columns are refused as model inputs; the step index only for a declared shortcut.
Plus: threshold rule, output contract, a corpus run of the majority baseline.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from agentdrift.config import CONFIGS, PROCESSED
from agentdrift.features.views import feature_columns, view_path
from agentdrift.models import experiment as E

# The toy has 4 trajectories per template: some inner strata (source_category x domain) are tiny, and
# StratifiedGroupKFold warns. Toy-only: the corpus run raises no warning (checked with -W error::UserWarning).
pytestmark = pytest.mark.filterwarnings("ignore:The least populated class:UserWarning")

# ----------------------------------------------------------------------------- toy data

CATEGORIES = ["benign", "hard_negative", "failed_attack", "attacked_full", "attacked_partial"]


def toy(n_templates: int = 40, traj_per_template: int = 4, steps: int = 4, seed: int = 0):
    """A gate-like view + folds table. Attacked trajectories are hijacked from step 2; the args of hijacked steps
    usually contain 'evil' (learnable signal). Folds by template, domains crossing folds, world folds by trajectory."""
    rng = np.random.default_rng(seed)
    rows, folds = [], []
    for g in range(n_templates):
        tpl, dom = f"tpl_{g:02d}", ["email", "banking", "web", "coding", "medical"][(g // 2) % 5]
        for j in range(traj_per_template):
            cat = CATEGORIES[(g + j) % len(CATEGORIES)]
            tid = f"{dom}_{cat}_{g:02d}{j}"
            folds.append({"traj_id": tid, "origin_split": "train", "template_id": tpl, "domain": dom,
                          "world_id": f"w{j}", "source_category": cat, "fold": g % 5, "world_fold": int(rng.integers(5))})
            for t in range(steps):
                y = int(cat.startswith("attacked") and t >= 2)
                evil = y if rng.random() < 0.85 else 1 - y
                rows.append({
                    "trajectory_id": tid, "step": t, "label": y, "domain": dom, "template_id": tpl,
                    "world_id": f"w{j}", "origin_split": "train", "is_first_h": bool(y and t == 2),
                    "history_n_steps": t, "history_truncated": False,
                    "task": "do the task", "history_text": "<call> x {}", "last_content": "ok",
                    "proposed_call_tool": "send", "proposed_call_args": "evil" if evil else "fine",
                })
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("Int8")
    df["is_first_h"] = df["is_first_h"].astype("boolean")
    return df, pd.DataFrame(folds)


FITS: list[np.ndarray] = []


class Spy(ClassifierMixin, BaseEstimator):
    """Learns P(y=1 | 'evil' in args) on its training rows and records the row positions it was fitted on."""

    def fit(self, X, y):
        FITS.append(np.asarray(X.index))
        evil = X["proposed_call_args"].str.contains("evil").to_numpy()
        y = np.asarray(y)
        self.rate_ = {True: y[evil].mean() if evil.any() else 0.0, False: y[~evil].mean() if (~evil).any() else 0.0}
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.where(X["proposed_call_args"].str.contains("evil"), self.rate_[True], self.rate_[False])
        p = p + 1e-6 * X["proposed_call_args"].str.len().to_numpy()  # break ties a little
        return np.column_stack([1 - p, p])


def spy() -> Spy:
    return Spy()


def cfg(**kw) -> E.ExperimentConfig:
    base = {"version": "toy", "kind": "model", "views": ["gate"], "model": {"factory": "tests.test_experiment:spy"},
            "bootstrap": {"n_resamples": 50, "seed": 42}}
    return E.ExperimentConfig(**{**base, **kw})


@pytest.fixture
def data():
    df, folds = toy()
    return E.make_dataset(df, "gate", cfg(), folds)


# ----------------------------------------------------------------------------- L2: fits see training rows only


@pytest.mark.parametrize("evaluation", E.EVALUATIONS)
@pytest.mark.parametrize("threshold_source", ["inner_oof", "train"])  # `estimator`: tested below with its own spy
def test_every_fit_sees_training_rows_only(data, evaluation, threshold_source):
    FITS.clear()
    c = cfg(evaluation=evaluation, threshold_source=threshold_source)
    E.run_view(c, data)
    splits = list(E.outer(data.meta, evaluation))
    per_split = 1 + (3 if threshold_source == "inner_oof" else 0)  # main fit + inner fits for the thresholds
    assert len(FITS) == len(splits) * per_split
    tpl = data.meta["template_id"].to_numpy()
    for i, fitted in enumerate(FITS):
        _, train, ev = splits[i // per_split]
        assert set(fitted) <= set(train), f"fit {i} saw rows outside the training part"
        assert not set(fitted) & set(ev)
        if evaluation != "world_fold":  # world folds share templates by design (control split)
            assert not set(tpl[fitted]) & set(tpl[ev]), f"fit {i} saw a template of the evaluated part"
        if i % per_split:  # inner fit: strictly inside the training part
            assert len(fitted) < len(train)


INNER: list[list] = []


class InnerSpy(Spy):
    """Declares `inner_cv` (as a calibrated or tuned model would) and records the splits it receives."""

    def __init__(self, inner_cv=None):
        self.inner_cv = inner_cv

    def fit(self, X, y):
        INNER.append((np.asarray(X.index), self.inner_cv))
        return super().fit(X, y)


def inner_spy(inner_cv=None) -> InnerSpy:
    return InnerSpy(inner_cv)


def test_inner_splits_keep_trajectories_and_templates_together(data):
    tr = next(E.outer(data.meta, "cv"))[1]
    meta = data.meta.iloc[tr].reset_index(drop=True)
    splits = E.inner_cv(meta, 42)
    assert len(splits) == 3
    assert np.array_equal(np.sort(np.concatenate([va for _, va in splits])), np.arange(len(meta)))  # a partition
    for itr, iva in splits:
        for key in ("trajectory_id", "template_id"):
            assert not set(meta[key].iloc[itr]) & set(meta[key].iloc[iva])


def test_a_factory_declaring_inner_cv_gets_splits_of_its_own_training_rows(data):
    INNER.clear()
    E.run_view(cfg(model={"factory": "tests.test_experiment:inner_spy"}), data)
    assert len(INNER) == 5 * 4  # main fit + 3 threshold fits per outer split
    tpl = data.meta["template_id"].to_numpy()
    for fitted, splits in INNER:
        assert splits is not None and len(splits) == 3
        for itr, iva in splits:  # positions relative to the rows this estimator is fitted on
            assert max(itr.max(), iva.max()) < len(fitted)
            assert not set(tpl[fitted[itr]]) & set(tpl[fitted[iva]])


# ----------------------------------------------------------------------------- factory arguments filled by the harness

SEEN: list[dict] = []


class ArgSpy(Spy):
    """Records the harness-filled arguments of its factory (`view`, `memory`)."""

    def __init__(self, view=None, memory=None):
        self.view, self.memory = view, memory

    def fit(self, X, y):
        SEEN.append({"view": self.view, "memory": self.memory})
        return super().fit(X, y)


def arg_spy(view=None, memory=None) -> ArgSpy:
    return ArgSpy(view, memory)


def test_a_factory_declaring_view_and_memory_receives_them(data, tmp_path):
    SEEN.clear()
    c = cfg(model={"factory": "tests.test_experiment:arg_spy"}, threshold_source="train")
    E.run_view(c, data, cache_dir=tmp_path)
    assert len(SEEN) == 5 and {s["view"] for s in SEEN} == {"gate"} and {s["memory"] for s in SEEN} == {str(tmp_path)}
    SEEN.clear()
    E.run_view(c, data)  # no cache requested: the factory gets None, never a stale directory
    assert {s["memory"] for s in SEEN} == {None}


def test_a_factory_declaring_nothing_receives_nothing_extra(data):
    E.run_view(cfg(threshold_source="train"), data)  # `spy()` takes no argument: would raise on an unexpected one


# ----------------------------------------------------------------------------- threshold_source: estimator

FIT_ROWS: list[np.ndarray] = []


class OofSpy(Spy):
    """What a tuned model does: declares `inner_cv`, fits on its inner splits and exposes `inner_oof_scores_`."""

    def __init__(self, inner_cv=None, mode="ok"):
        self.inner_cv, self.mode = inner_cv, mode

    def fit(self, X, y):
        FIT_ROWS.append(np.asarray(X.index))
        super().fit(X, y)
        y = np.asarray(y)
        oof = np.full(len(X), np.nan)
        for itr, iva in self.inner_cv:
            inner = Spy().fit(X.iloc[itr], y[itr])
            oof[iva] = inner.predict_proba(X.iloc[iva])[:, 1]
        if self.mode == "ok":
            self.inner_oof_scores_ = oof
        elif self.mode == "short":
            self.inner_oof_scores_ = oof[:-1]
        elif self.mode == "nan":
            self.inner_oof_scores_ = np.where(np.arange(len(oof)) == 0, np.nan, oof)
        return self  # mode == "missing": no attribute


def oof_spy(inner_cv=None, mode="ok") -> OofSpy:
    return OofSpy(inner_cv, mode)


def test_estimator_source_uses_the_scores_of_the_estimator_and_fits_once_per_split(data):
    FIT_ROWS.clear()
    c = cfg(model={"factory": "tests.test_experiment:oof_spy"}, threshold_source="estimator")
    oof, met = E.run_view(c, data)
    splits = list(E.outer(data.meta, "cv"))
    assert len(FIT_ROWS) == len(splits)  # no extra fits for the thresholds: the estimator did them itself
    tpl = data.meta["template_id"].to_numpy()
    for fitted, (_, train, ev) in zip(FIT_ROWS, splits):
        assert set(fitted) <= set(train) and not set(tpl[fitted]) & set(tpl[ev])
    # the thresholds are the ones of the TRAINING part's out-of-fold scores: same value as the default source on a
    # model whose inner fits are the same
    base, met_inner = E.run_view(cfg(model={"factory": "tests.test_experiment:inner_spy"}), data)
    assert [f["thresholds"] for f in met["folds"]] == [f["thresholds"] for f in met_inner["folds"]]
    pd.testing.assert_frame_equal(oof.drop(columns=["score"]), base.drop(columns=["score"]), check_exact=False)


@pytest.mark.parametrize("mode, message", [("missing", "inner_oof_scores_"), ("short", "one finite score"),
                                            ("nan", "one finite score")])
def test_estimator_source_refuses_unusable_scores(data, mode, message):
    c = cfg(model={"factory": "tests.test_experiment:oof_spy", "params": {"mode": mode}}, threshold_source="estimator")
    with pytest.raises(ValueError, match=message):
        E.run_view(c, data)


def test_labels_of_the_evaluated_fold_never_reach_the_estimator_thresholds(data):
    c = cfg(model={"factory": "tests.test_experiment:oof_spy"}, threshold_source="estimator")
    _, met = E.run_view(c, data)
    flipped = E.Dataset(data.view, data.X, data.y.copy(), data.meta, data.is_first_h)
    in_fold0 = (data.meta["fold"] == 0).to_numpy()
    flipped.y[in_fold0] = 1 - flipped.y[in_fold0]
    _, met2 = E.run_view(c, flipped)
    assert met["folds"][0]["thresholds"] == met2["folds"][0]["thresholds"]


# ----------------------------------------------------------------------------- paired difference


def test_paired_difference_of_a_run_with_itself_is_exactly_zero(data):
    oof, _ = E.run_view(cfg(), data)
    d = E.paired_difference(oof, oof, [0.01, 0.002], 100, 42)
    for m, v in d.items():
        assert v["point"] == 0 and v["low"] == 0 and v["high"] == 0, m


def test_paired_difference_sees_a_better_model_and_ignores_row_order(data):
    oof, _ = E.run_view(cfg(), data)
    rng = np.random.default_rng(5)
    worse = oof.assign(score=rng.random(len(oof)))  # random scores: PR-AUC near the prevalence
    d = E.paired_difference(oof, worse.sample(frac=1, random_state=1), [0.01], 200, 42)
    assert d["pr_auc"]["point"] > 0.1 and d["pr_auc"]["low"] > 0  # the signal model is clearly better
    assert d["pr_auc"]["low"] <= d["pr_auc"]["point"] <= d["pr_auc"]["high"]
    rev = E.paired_difference(worse, oof, [0.01], 200, 42)
    assert rev["pr_auc"]["point"] == pytest.approx(-d["pr_auc"]["point"])


def test_paired_difference_refuses_runs_that_do_not_hold_the_same_rows(data):
    oof, _ = E.run_view(cfg(), data)
    with pytest.raises(ValueError, match="same rows"):
        E.paired_difference(oof, oof.iloc[:-1], [0.01], 50, 42)
    other = oof.assign(label=1 - oof["label"])
    with pytest.raises(ValueError, match="same rows"):
        E.paired_difference(oof, other, [0.01], 50, 42)


# ----------------------------------------------------------------------------- L5: evaluated labels never used


def test_labels_of_the_evaluated_fold_never_reach_its_scores_or_thresholds(data):
    c = cfg()
    oof, met = E.run_view(c, data)
    flipped = E.Dataset(data.view, data.X, data.y.copy(), data.meta, data.is_first_h)
    in_fold0 = (data.meta["fold"] == 0).to_numpy()
    flipped.y[in_fold0] = 1 - flipped.y[in_fold0]
    oof2, met2 = E.run_view(c, flipped)
    a, b = oof[oof["split"] == "cv0"], oof2[oof2["split"] == "cv0"]
    np.testing.assert_array_equal(a["score"].to_numpy(), b["score"].to_numpy())
    assert met["folds"][0]["thresholds"] == met2["folds"][0]["thresholds"]
    # sanity: the other folds train on the flipped labels, so their scores do change
    assert not np.array_equal(oof.loc[oof["split"] == "cv1", "score"], oof2.loc[oof2["split"] == "cv1", "score"])


def test_fixed_test_rows_are_refused():
    df, folds = toy()
    folds.loc[0, "fold"] = -1
    with pytest.raises(ValueError, match="fixed test set"):
        E.make_dataset(df, "gate", cfg(), folds)


# ----------------------------------------------------------------------------- L12: input columns


@pytest.mark.parametrize("bad", ["template_id", "label", "trajectory_id", "domain", "world_id", "is_first_h",
                                 "history_n_steps", "source_category", "step"])
def test_non_feature_columns_are_refused(bad):
    with pytest.raises(ValueError):
        E.check_columns(["proposed_call_args", bad], "gate", False, uses_position=False)


def test_step_only_for_a_declared_shortcut(data):
    with pytest.raises(ValueError, match="shortcut"):
        cfg(uses_position=True)  # kind: model
    E.check_columns(["step"], "gate", False, uses_position=True)
    df, folds = toy()
    d = E.make_dataset(df, "gate", cfg(kind="shortcut", uses_position=True, columns=["proposed_call_args"]), folds)
    assert list(d.X.columns) == ["proposed_call_args", "step"]


def test_thought_column_only_in_the_thought_variant():
    with pytest.raises(ValueError, match="not features"):
        E.check_columns(["proposed_call_thought"], "gate", False, uses_position=False)
    E.check_columns(["proposed_call_thought"], "gate", True, uses_position=False)


def test_default_input_is_every_feature_column_and_nothing_else(data):
    assert list(data.X.columns) == feature_columns("gate")
    assert not set(data.X.columns) & set(E.META_COLUMNS)


# ----------------------------------------------------------------------------- thresholds and outputs


def test_threshold_at_fpr_keeps_reference_fpr_at_or_below_target():
    rng = np.random.default_rng(1)
    neg = rng.normal(size=1000)
    for fpr in (0.01, 0.002, 0.05):
        thr = E.threshold_at_fpr(neg, fpr)
        assert np.mean(neg > thr) <= fpr
        assert np.mean(neg >= thr) > fpr - 1e-9  # the threshold is the tightest such value
    assert np.mean(np.zeros(100) > E.threshold_at_fpr(np.zeros(100), 0.01)) == 0  # all tied: no alert


def test_output_contract_and_majority_values():
    df, folds = toy()
    c = cfg(model={"factory": "agentdrift.models.baselines:majority"}, kind="trivial")
    d = E.make_dataset(df, "gate", c, folds)
    oof, met = E.run_view(c, d)
    assert list(oof.columns[: len(E.OOF_COLUMNS)]) == list(E.OOF_COLUMNS)
    assert {"thr@fpr0.01", "thr@fpr0.002"} <= set(oof.columns)
    assert len(oof) == len(df) and not oof.duplicated(["trajectory_id", "step"]).any()
    assert met["estimate"]["roc_auc"]["point"] == pytest.approx(0.5)
    assert met["estimate"]["roc_auc"]["low"] == pytest.approx(0.5) == met["estimate"]["roc_auc"]["high"]
    assert met["fold_summary"]["roc_auc"]["mean"] == pytest.approx(0.5)
    assert met["estimate"]["recall@fpr0.01"]["point"] == 0  # constant score: never above its own threshold
    for f in met["folds"]:
        assert f["pr_auc"] == pytest.approx(f["prevalence"])
    rec = met["subgroups"]["fpr0.01"]["recall_by_attack_type"]
    assert set(rec) == {"attacked_full", "attacked_partial"}


def test_headline_does_not_pool_scores_of_different_models():
    """A constant model scores its training prevalence: pooling the five folds' scores into one ranking orders the
    folds, not the rows (pooled ROC-AUC 0.41 on this toy). The headline estimate averages per-split metrics."""
    df, folds = toy()
    c = cfg(model={"factory": "agentdrift.models.baselines:majority"}, kind="trivial")
    _, met = E.run_view(c, E.make_dataset(df, "gate", c, folds))
    assert met["pooled_oof"]["roc_auc"] != pytest.approx(0.5)  # the bias exists on this data...
    assert met["estimate"]["roc_auc"]["point"] == pytest.approx(0.5)  # ...and the headline is free of it


def test_resamples_are_shared_between_models_for_paired_comparisons(data):
    """Same view, splits and seed -> same template draws: the CI of a difference is a paired one."""
    oof, _ = E.run_view(cfg(), data)
    other = oof.assign(score=oof["score"] + np.random.default_rng(3).normal(0, 0.01, len(oof)))
    _, r1 = E.split_resamples(oof, [0.01], 200, 42)
    _, r2 = E.split_resamples(other, [0.01], 200, 42)
    np.testing.assert_array_equal(r1["prevalence"].to_numpy(), r2["prevalence"].to_numpy())  # same draws


def test_metrics_are_json_safe(data):
    _, met = E.run_view(cfg(), data)
    json.dumps(E._clean(met), allow_nan=False)


def test_config_file_loads_and_unknown_keys_are_refused(tmp_path):
    c = E.ExperimentConfig.from_yaml(CONFIGS / "models" / "v0_0_majority.yaml")
    assert c.text == "anon" and c.include_thought is False and c.kind == "trivial"
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: x\nmodel: {factory: a:b}\nlearning_rate: 3\n")
    with pytest.raises(ValueError, match="unknown keys"):
        E.ExperimentConfig.from_yaml(bad)


# ----------------------------------------------------------------------------- corpus (needs views and folds)

needs_views = pytest.mark.skipif(
    not (view_path("gate").exists() and (PROCESSED / "folds.parquet").exists()),
    reason="run `python -m agentdrift.splits.grouped` and `python -m agentdrift.pipelines.build_views` first",
)


@needs_views
def test_majority_on_the_corpus():
    c = E.ExperimentConfig.from_yaml(CONFIGS / "models" / "v0_0_majority.yaml")
    c.bootstrap = {"n_resamples": 100, "seed": 42}
    res = E.run_experiment(c, write=False)
    expected_rows = {"scanner": 50_414, "gate": 61_228}  # dev rows (train + val) of each view
    for view, r in res.items():
        assert r["data"]["rows"] == expected_rows[view]
        assert r["data"]["folds_hash_lf"] == "f69d2d09f08332a9"
        assert r["estimate"]["roc_auc"]["point"] == pytest.approx(0.5)
        assert r["estimate"]["pr_auc"]["point"] == pytest.approx(r["data"]["prevalence"], abs=0.01)
        assert len(r["folds"]) == 5


# ----------------------------------------------------------------------------- comparison table (run_baseline)


def test_comparison_table_has_headline_metrics_and_paired_differences(data):
    from agentdrift.pipelines.run_baseline import comparison_rows

    def report(version, oof, metrics, kind="model"):
        return {"version": version, "kind": kind, "view": "gate", "evaluation": "cv",
                "data": {"text": "anon", "rows": len(oof), "prevalence": float(oof["label"].mean())},
                "config": {"target_fpr": [0.01, 0.002], "bootstrap": {"n_resamples": 50, "seed": 42}}, **metrics}

    ref_oof, ref_met = E.run_view(cfg(model={"factory": "agentdrift.models.baselines:majority"}, kind="trivial"), data)
    oof, met = E.run_view(cfg(), data)
    table = comparison_rows([(report("ref", ref_oof, ref_met, "trivial"), ref_oof), (report("m", oof, met), oof)],
                            "ref", {("gate", "cv"): ref_oof})
    assert list(table["version"]) == ["ref", "m"]
    assert {"pr_auc", "pr_auc_low", "pr_auc_high", "recall@fpr0.01", "fpr@fpr0.002"} <= set(table)
    assert table.loc[0, ["delta_pr_auc"]].isna().all()  # the reference is not compared with itself
    assert table.loc[1, "delta_pr_auc"] > 0 and table.loc[1, "delta_pr_auc_low"] > 0
    assert table.loc[1, "delta_pr_auc"] == pytest.approx(table.loc[1, "pr_auc"] - table.loc[0, "pr_auc"])


def test_comparison_without_reference_has_no_difference_columns(data):
    from agentdrift.pipelines.run_baseline import comparison_rows
    oof, met = E.run_view(cfg(), data)
    rep = {"version": "m", "kind": "model", "view": "gate", "evaluation": "cv",
           "data": {"text": "anon", "rows": len(oof), "prevalence": 0.1},
           "config": {"target_fpr": [0.01, 0.002], "bootstrap": {"n_resamples": 50, "seed": 42}}, **met}
    assert not [c for c in comparison_rows([(rep, oof)]).columns if c.startswith("delta_")]
