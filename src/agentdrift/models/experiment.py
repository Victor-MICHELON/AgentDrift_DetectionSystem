"""Experiment harness (B.0, KAN-61): one leakage-safe loop to train and evaluate every model version.

    python -m agentdrift.pipelines.run_experiment --config configs/models/<version>.yaml

A version is ONE YAML file (configs/models/), read by `ExperimentConfig.from_yaml`:

    version: v0_0_majority          # results/runs/<version>/<view>[_<evaluation>]/
    kind: trivial                   # trivial | shortcut | model ("shortcut" = reads a generator regularity)
    views: [scanner, gate]
    text: anon                      # anon (default) | masked | raw (H3 ablation only)
    include_thought: false
    columns: null                   # subset of views.feature_columns(view); null = all of them
    uses_position: false            # only a kind: shortcut may read the step index t
    evaluation: cv                  # cv (5 folds by template) | lodo (leave one domain out) | world_fold (control)
    model:
      factory: agentdrift.models.baselines:majority    # "module:function" returning an sklearn estimator
      params: {}
    threshold_source: inner_oof     # inner_oof | train (see below)
    target_fpr: [0.01, 0.002]       # per-step FPR operating points (configs/split.yaml)
    bootstrap: {n_resamples: 2000, seed: 42}
    seed: 42

Contract of the loop, for each outer split (k, train rows, evaluated rows):

    estimator   factory(**params) is called afresh and fitted on the TRAINING rows only (L2). A factory that declares
                an `inner_cv` argument receives `inner_splits` of the training part (grouped by template, stratified
                like the outer folds): calibration and tuning happen there and nowhere else (L5).
    thresholds  the score threshold of each target FPR is set on scores of the TRAINING part, never on the
                evaluated rows (L5). `inner_oof` (default): out-of-fold scores of the training part over its inner
                splits, an honest estimate of how unseen negatives score. `train`: in-sample scores of the fitted
                model, cheaper but optimistic (a model that memorises its negatives gets a threshold that is too low,
                hence an observed FPR above target); kept for cheap baselines only.
    inputs      the matrix X holds only feature columns of the view (`check_columns`): keys, labels, groups,
                diagnostics and generator metadata (NON_FEATURE_COLUMNS, FORBIDDEN_FIELDS) are refused; the step
                index t only for a declared shortcut (L3, L12). The fixed test set is never read: `read_view` hides
                it and any test row is refused (L5).

Outputs per view (and evaluation mode):

    oof.parquet   one row per decision point (not versioned): trajectory_id, step, label, score, view, split,
                  version, domain, template_id, is_first_h (gate; NA for scanner), thr@<target> (threshold of the
                  split that produced the row). Enough for D's first-alert metrics without re-running anything.
    metrics.json  config, seed, git hash, folds fingerprint, library versions, duration; per split metrics and
                  mean +- std over splits; `estimate` = mean over splits with a 95 % cluster-bootstrap CI over
                  templates stratified by split (`split_resamples`, paired across models); train / evaluation gap;
                  recall by attack type and alert rate by negative class, joined AFTER prediction for analysis only.

`pooled_oof` (diagnostic only) ranks all out-of-fold rows together: that mixes the score scales of five models
and is biased (a constant model gets a pooled ROC-AUC of 0.41 on the toy test, AI_AUDIT). The headline is `estimate`.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import platform
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

from agentdrift.config import ROOT, RUNS, load_config
from agentdrift.evaluation.bootstrap import bucket_counts, interval, metrics_from_counts, point_estimate
from agentdrift.features.views import (
    FORBIDDEN_FIELDS, NON_FEATURE_COLUMNS, TEXT_VARIANTS, VIEWS, feature_columns, read_view,
)
from agentdrift.splits.grouped import TEST_FOLD, folds_hash, inner_splits, load_folds, lodo_splits, outer_splits
from agentdrift.utils.seed import set_seed

log = logging.getLogger(__name__)

KINDS = ("trivial", "shortcut", "model")
EVALUATIONS = ("cv", "lodo", "world_fold")
THRESHOLD_SOURCES = ("inner_oof", "train")
POSITION_COLUMN = "step"
# Per-row metadata kept next to X for splitting, bootstrap and subgroup analysis. NEVER model inputs.
META_COLUMNS = ("trajectory_id", "step", "label", "domain", "template_id", "source_category", "fold", "world_fold")
OOF_COLUMNS = ("trajectory_id", "step", "label", "score", "view", "split", "version", "domain", "template_id",
               "is_first_h")
NEGATIVE_CLASSES = ("benign", "hard_negative", "failed_attack")
LIBRARIES = ("numpy", "pandas", "scipy", "sklearn", "imblearn", "optuna")


# ----------------------------------------------------------------------------- configuration


@dataclass
class ExperimentConfig:
    version: str
    model: dict
    kind: str = "model"
    description: str = ""
    views: list[str] = field(default_factory=lambda: list(VIEWS))
    text: str = "anon"
    include_thought: bool = False
    columns: list[str] | None = None
    uses_position: bool = False
    evaluation: str = "cv"
    threshold_source: str = "inner_oof"
    target_fpr: list[float] = field(default_factory=lambda: list(load_config("split")["target_fpr_per_step"]))
    bootstrap: dict = field(default_factory=lambda: {"n_resamples": load_config("split")["bootstrap"]["n_resamples"],
                                                      "seed": 42})
    seed: int = 42

    def __post_init__(self) -> None:
        checks = [
            (self.kind in KINDS, f"kind must be one of {KINDS}"),
            (self.text in TEXT_VARIANTS, f"text must be one of {TEXT_VARIANTS}"),
            (self.evaluation in EVALUATIONS, f"evaluation must be one of {EVALUATIONS}"),
            (self.threshold_source in THRESHOLD_SOURCES, f"threshold_source must be one of {THRESHOLD_SOURCES}"),
            (set(self.views) <= set(VIEWS) and self.views, f"views must be a non-empty subset of {VIEWS}"),
            ("factory" in self.model, "model.factory is required ('module:function')"),
            (all(0 < f < 1 for f in self.target_fpr), "target_fpr values must be in (0, 1)"),
            (not self.uses_position or self.kind == "shortcut",
             "only a kind: shortcut may read the step index (uses_position: true)"),
        ]
        bad = [msg for ok, msg in checks if not ok]
        if bad:
            raise ValueError(f"invalid config {self.version!r}: " + "; ".join(bad))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        return cls(**raw)

    def output_dir(self, view: str, root: Path = RUNS) -> Path:
        return root / self.version / (view if self.evaluation == "cv" else f"{view}_{self.evaluation}")


def resolve_factory(spec: str) -> Callable[..., Any]:
    """'package.module:function' -> the function (estimator factory)."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"model.factory must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module), name)


def needs_inner_cv(cfg: ExperimentConfig) -> bool:
    return "inner_cv" in inspect.signature(resolve_factory(cfg.model["factory"])).parameters


def make_estimator(cfg: ExperimentConfig, meta_train: pd.DataFrame, seed: int):
    """A fresh, unfitted estimator for these training rows. A factory that declares `inner_cv` receives the inner
    splits of exactly these rows (positions relative to them); the others never trigger their computation."""
    factory = resolve_factory(cfg.model["factory"])
    params = dict(cfg.model.get("params") or {})
    if needs_inner_cv(cfg):
        params["inner_cv"] = inner_cv(meta_train, seed)
    return clone(factory(**params))


# ----------------------------------------------------------------------------- data


def check_columns(columns: list[str], view: str, include_thought: bool, uses_position: bool) -> None:
    """Refuse any input column that is not a feature of the view (L3, L11, L12)."""
    allowed = set(feature_columns(view, include_thought)) | ({POSITION_COLUMN} if uses_position else set())  # type: ignore[arg-type]
    forbidden = (set(NON_FEATURE_COLUMNS) | set(FORBIDDEN_FIELDS)) - ({POSITION_COLUMN} if uses_position else set())
    leaked = sorted(set(columns) & forbidden)
    unknown = sorted(set(columns) - allowed - set(leaked))
    if leaked:
        raise ValueError(f"non-feature / forbidden columns in the model input: {leaked}")
    if unknown:
        raise ValueError(f"columns {unknown} are not features of the {view} view (thought={include_thought})")


@dataclass
class Dataset:
    """Model input X and per-row metadata (same row order). `meta` is never given to the estimator."""
    view: str
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    is_first_h: pd.Series


def attach_meta(df: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Folds and source_category of each row, by trajectory (join key only)."""
    cols = ["traj_id", "source_category", "fold", "world_fold"]
    out = df.merge(folds[cols].rename(columns={"traj_id": "trajectory_id"}), on="trajectory_id", how="left",
                   validate="many_to_one")
    if out["fold"].isna().any():
        raise ValueError(f"{int(out['fold'].isna().sum())} rows have no fold: rebuild folds / views")
    if (out["fold"] == TEST_FOLD).any() or (out.get("origin_split", pd.Series(dtype=str)) == "test").any():
        raise ValueError("the fixed test set reached the harness: it is never read in development (L5)")
    return out


def make_dataset(view_df: pd.DataFrame, view: str, cfg: ExperimentConfig, folds: pd.DataFrame) -> Dataset:
    cols = list(cfg.columns) if cfg.columns else feature_columns(view, cfg.include_thought)  # type: ignore[arg-type]
    cols = cols + ([POSITION_COLUMN] if cfg.uses_position and POSITION_COLUMN not in cols else [])
    check_columns(cols, view, cfg.include_thought, cfg.uses_position)
    df = attach_meta(view_df, folds).reset_index(drop=True)
    if df["label"].isna().any():
        raise ValueError("rows without a label cannot be used for training or evaluation")
    first_h = df["is_first_h"] if "is_first_h" in df else pd.Series(pd.NA, index=df.index, dtype="boolean")
    return Dataset(view, df[cols].copy(), df["label"].astype(int).to_numpy(), df[list(META_COLUMNS)].copy(), first_h)


def load_dataset(view: str, cfg: ExperimentConfig) -> Dataset:
    """Development rows of a persisted view (fixed test set hidden by `read_view`)."""
    return make_dataset(read_view(view, cfg.include_thought, text=cfg.text), view, cfg, load_folds())  # type: ignore[arg-type]


def outer(meta: pd.DataFrame, evaluation: str) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """(split name, train positions, evaluated positions) of the chosen evaluation mode."""
    if evaluation == "lodo":
        yield from ((str(d), tr, ev) for d, tr, ev in lodo_splits(meta))
    else:
        yield from ((f"{evaluation}{k}", tr, ev) for k, tr, ev in outer_splits(meta, "fold" if evaluation == "cv"
                                                                               else "world_fold"))


def inner_cv(meta: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner splits of a training part (positions relative to `meta`): `splits.grouped.inner_splits` applied to its
    TRAJECTORIES (grouped by template, stratified by source_category x domain, like the outer folds, which are made
    at the trajectory level), then expanded to their steps. A trajectory's steps always stay together."""
    meta = meta.reset_index(drop=True)
    traj = meta.drop_duplicates("trajectory_id").reset_index(drop=True)
    rows = meta.groupby("trajectory_id", sort=False).indices
    ids = traj["trajectory_id"].to_numpy()

    def expand(pos: np.ndarray) -> np.ndarray:
        return np.sort(np.concatenate([rows[t] for t in ids[pos]]))

    return [(expand(tr), expand(va)) for tr, va in inner_splits(traj, seed=seed)]


# ----------------------------------------------------------------------------- scores and thresholds


def score(estimator, X: pd.DataFrame) -> np.ndarray:
    """Positive-class score: predict_proba[:, 1], else decision_function."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return np.asarray(proba[:, list(estimator.classes_).index(1)] if proba.shape[1] > 1 else proba[:, 0], float)
    return np.asarray(estimator.decision_function(X), dtype=float)


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float) -> float:
    """Threshold such that at most `fpr` of these negatives score strictly above it (alert = score > threshold).

    With n negatives, k = floor(fpr * n) may exceed it: the threshold is the (k+1)-th largest score. Ties at that
    value are all below the alert line, so the reference FPR is <= fpr."""
    s = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    if len(s) == 0:
        raise ValueError("no negative score to set a threshold on")
    return float(s[min(int(np.floor(fpr * len(s))), len(s) - 1)])


def fit_fold(cfg: ExperimentConfig, data: Dataset, train: np.ndarray, seed: int):
    """Fit on the training rows only; return (estimator, reference scores of the training part for thresholds)."""
    X_tr, y_tr, meta_tr = data.X.iloc[train], data.y[train], data.meta.iloc[train]
    est = make_estimator(cfg, meta_tr, seed)
    est.fit(X_tr, y_tr)
    if cfg.threshold_source == "train":
        return est, score(est, X_tr)
    ref = np.full(len(train), np.nan)
    for itr, iva in inner_cv(meta_tr, seed):
        e = make_estimator(cfg, meta_tr.iloc[itr], seed)
        e.fit(X_tr.iloc[itr], y_tr[itr])
        ref[iva] = score(e, X_tr.iloc[iva])
    if np.isnan(ref).any():
        raise RuntimeError("inner splits did not cover the training part")
    return est, ref


# ----------------------------------------------------------------------------- metrics


def _target_name(fpr: float) -> str:
    return f"fpr{fpr:g}"


def _clean(obj):
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def split_resamples(oof: pd.DataFrame, targets: list[float], n_resamples: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """Metrics averaged over the evaluated splits, and their cluster-bootstrap resamples.

    Each split is scored by its own model, so its metrics are computed on its own rows; the estimate is the mean
    over splits (as the per-split mean +- std). One resample draws, in every split independently, as many of its
    templates as it holds, with replacement, then averages the per-split metrics: a template bootstrap stratified
    by split. Pooling all out-of-fold scores into one ranking would compare scores of different models (a constant
    model gets a pooled ROC-AUC below 0.5 when the training prevalence moves against the evaluated one; see
    `pooled_oof`). The draws depend only on the seed and on the templates of each split: two models evaluated on
    the same view and splits get the same resamples, so their difference is paired."""
    rng = np.random.default_rng(seed)
    names = [_target_name(f) for f in targets]
    points, draws = [], []
    for _, d in oof.groupby("split", sort=False):
        c = bucket_counts(d["score"], d["label"], d["template_id"])
        thr = {n: float(d[f"thr@{n}"].iloc[0]) for n in names}
        ng = len(c.clusters)
        w = rng.multinomial(ng, np.full(ng, 1 / ng), size=n_resamples)
        points.append(pd.DataFrame(metrics_from_counts(c.pos.sum(0), c.neg.sum(0), c.scores, thr)))
        draws.append(pd.DataFrame(metrics_from_counts(w @ c.pos, w @ c.neg, c.scores, thr)))
    point = pd.concat(points).mean().to_dict()
    resamples = sum(draws) / len(draws)  # NaN (e.g. no positive in a resample of a split) propagates, then dropped
    return point, resamples


def estimate_with_ci(oof: pd.DataFrame, targets: list[float], n_resamples: int, seed: int) -> dict:
    """{metric: {point, low, high}}: mean over splits with a 95 % template bootstrap stratified by split."""
    point, res = split_resamples(oof, targets, n_resamples, seed)
    ci = interval(res)
    return {m: {"point": point[m], "low": ci.loc[m, "low"], "high": ci.loc[m, "high"],
                "nan_resamples": int(res[m].isna().sum())} for m in res.columns}


def pooled_oof(oof: pd.DataFrame, targets: list[float]) -> dict:
    """Diagnostic only: metrics of all out-of-fold rows pooled into one ranking (mixes the scales of the split
    models; recall / FPR use per-row alerts, which do not mix scales)."""
    out = point_estimate(oof["score"], oof["label"])
    for fpr in targets:
        n = _target_name(fpr)
        alert = (oof["score"] > oof[f"thr@{n}"]).astype(float)
        pt = point_estimate(alert, oof["label"], thresholds={n: 0.5})
        out |= {f"recall@{n}": pt[f"recall@{n}"], f"fpr@{n}": pt[f"fpr@{n}"]}
    return out


def subgroup_metrics(oof: pd.DataFrame, source_category: pd.Series, targets: list[float]) -> dict:
    """Recall by attack type (positives) and alert rate by negative class, at each target. Joined after prediction,
    for analysis only. Every category of the data is listed; NEGATIVE_CLASSES are the trajectory-level negatives."""
    d = oof.assign(source_category=source_category.to_numpy())
    out = {}
    for fpr in targets:
        name = _target_name(fpr)
        alert = d["score"] > d[f"thr@{name}"]
        pos = d["label"] == 1
        rec = alert[pos].groupby(d.loc[pos, "source_category"]).agg(["mean", "size"])
        neg = alert[~pos].groupby(d.loc[~pos, "source_category"]).agg(["mean", "size"])
        out[name] = {
            "recall_by_attack_type": {k: {"recall": v["mean"], "n_positive": v["size"]} for k, v in rec.iterrows()},
            "alert_rate_by_class_of_negatives": {k: {"alert_rate": v["mean"], "n_negative": v["size"]}
                                                 for k, v in neg.iterrows()},
        }
    return out


def fold_summary(folds: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = np.array([f[k] for f in folds if f.get(k) is not None], dtype=float)
        out[k] = {"mean": float(np.mean(vals)) if len(vals) else None,
                  "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    return out


# ----------------------------------------------------------------------------- run


def run_view(cfg: ExperimentConfig, data: Dataset) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold predictions and metrics of one view under the config's evaluation mode."""
    set_seed(cfg.seed)
    names = [_target_name(f) for f in cfg.target_fpr]
    parts, folds = [], []
    for split, train, ev in outer(data.meta, cfg.evaluation):
        start = time.perf_counter()
        est, ref = fit_fold(cfg, data, train, cfg.seed)
        thr = {n: threshold_at_fpr(ref[data.y[train] == 0], f) for n, f in zip(names, cfg.target_fpr)}
        s_ev = score(est, data.X.iloc[ev])
        s_tr = score(est, data.X.iloc[train])
        m_ev = point_estimate(s_ev, data.y[ev], thresholds=thr)
        m_tr = point_estimate(s_tr, data.y[train])
        folds.append({
            "split": split, "n_train": int(len(train)), "n_eval": int(len(ev)),
            "positives_eval": int(data.y[ev].sum()), "templates_eval": int(data.meta["template_id"].iloc[ev].nunique()),
            **m_ev, "train_roc_auc": m_tr["roc_auc"], "train_pr_auc": m_tr["pr_auc"],
            "gap_pr_auc": m_tr["pr_auc"] - m_ev["pr_auc"],
            "thresholds": thr, "reference_fpr": {n: float(np.mean(ref[data.y[train] == 0] > thr[n])) for n in names},
            "seconds": round(time.perf_counter() - start, 2),
        })
        m = data.meta.iloc[ev]
        part = pd.DataFrame({
            "trajectory_id": m["trajectory_id"].to_numpy(), "step": m["step"].to_numpy(), "label": data.y[ev],
            "score": s_ev, "view": data.view, "split": split, "version": cfg.version,
            "domain": m["domain"].to_numpy(), "template_id": m["template_id"].to_numpy(),
            "is_first_h": data.is_first_h.iloc[ev].to_numpy(),
        })
        for n in names:
            part[f"thr@{n}"] = thr[n]
        parts.append(part)
        log.info("%s %s %s: PR-AUC %.3f (%.1fs)", cfg.version, data.view, split, m_ev["pr_auc"], folds[-1]["seconds"])

    oof = pd.concat(parts, ignore_index=True)
    oof["is_first_h"] = oof["is_first_h"].astype("boolean")
    src = data.meta.set_index(["trajectory_id", "step"])["source_category"]
    source = src.reindex(pd.MultiIndex.from_frame(oof[["trajectory_id", "step"]])).reset_index(drop=True)
    keys = ["roc_auc", "pr_auc", *[f"{p}@{n}" for n in names for p in ("recall", "fpr")],
            "train_pr_auc", "gap_pr_auc"]
    metrics = {
        "folds": folds,
        "fold_summary": fold_summary(folds, keys),
        "estimate": estimate_with_ci(oof, cfg.target_fpr, cfg.bootstrap["n_resamples"], cfg.bootstrap["seed"]),
        "pooled_oof": pooled_oof(oof, cfg.target_fpr),
        "subgroups": subgroup_metrics(oof, source, cfg.target_fpr),
    }
    return oof, metrics


def _git_hash() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout
    return out.stdout.strip() + ("-dirty" if dirty.strip() else "")


def folds_fingerprints(folds: pd.DataFrame) -> dict[str, str]:
    """`folds_hash` of Lot A as is, and the same hash with '\\n' line endings: the first depends on the OS (to_csv
    writes '\\r\\n' on Windows, JOURNAL 23/11), the second only on the folds (f69d2d09f08332a9 for the frozen folds)."""
    s = folds.sort_values("traj_id")[["traj_id", "fold", "world_fold"]].to_csv(index=False, lineterminator="\n")
    return {"folds_hash": folds_hash(folds), "folds_hash_lf": hashlib.sha256(s.encode()).hexdigest()[:16]}


def run_info(seconds: float) -> dict:
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = importlib.import_module(lib).__version__
        except ImportError:
            versions[lib] = None
    return {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": _git_hash(),
            "python": platform.python_version(), "libraries": versions, "seconds": round(seconds, 1)}


def run_experiment(cfg: ExperimentConfig, views: list[str] | None = None, out_root: Path = RUNS,
                   write: bool = True) -> dict[str, dict]:
    """Run the config on each view; write oof.parquet + metrics.json per view. Returns the metrics by view."""
    folds = load_folds()
    fingerprints = folds_fingerprints(folds)
    results = {}
    for view in views or cfg.views:
        start = time.perf_counter()
        data = load_dataset(view, cfg)
        oof, metrics = run_view(cfg, data)
        report = {
            "version": cfg.version, "kind": cfg.kind, "description": cfg.description, "view": view,
            "evaluation": cfg.evaluation, "seed": cfg.seed,
            "data": {"text": cfg.text, "include_thought": cfg.include_thought, "columns": list(data.X.columns),
                     "rows": int(len(data.y)), "positives": int(data.y.sum()), "prevalence": float(data.y.mean()),
                     "templates": int(data.meta["template_id"].nunique()), **fingerprints},
            "config": asdict(cfg),
            **metrics,
            "run": run_info(time.perf_counter() - start),
        }
        report = _clean(report)
        if write:
            out = cfg.output_dir(view, out_root)
            out.mkdir(parents=True, exist_ok=True)
            oof.to_parquet(out / "oof.parquet", index=False)
            (out / "metrics.json").write_text(json.dumps(report, indent=2))
        results[view] = report
    return results
