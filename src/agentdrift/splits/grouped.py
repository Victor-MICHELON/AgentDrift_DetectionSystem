"""Evaluation splits (CONTEXT 8.1, configs/split.yaml): fixed test set, grouped CV, leave-one-domain-out.

    python -m agentdrift.splits.grouped     # write data/processed/folds.parquet + per-fold summary

Every split is made at the TRAJECTORY level and grouped by task template, so the steps of a trajectory and the
trajectories of a template always fall on the same side (leak L1). Steps inherit the fold of their trajectory
(`attach_folds`).

folds.parquet, one row per trajectory of the whole corpus:
    traj_id, origin_split, template_id, domain, world_id
    fold        outer fold 0..K-1 for dev trajectories (train + val merged), -1 for the fixed test set
    world_fold  control split: GroupKFold by world (dev only, -1 for test)

Leave-one-domain-out needs no stored column: the held-out part is `domain == d` among dev trajectories.
Inner folds (nested CV) are generated on demand by `inner_splits`, grouped by template inside a training part.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from agentdrift.config import PROCESSED, ROOT, load_config
from agentdrift.data.load import read_steps, read_trajectories

FOLDS_PATH = PROCESSED / "folds.parquet"
TEST_FOLD = -1


def _cfg() -> dict:
    return load_config("split")


def strata(traj: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """Stratification label: source_category x domain (stratification only, never a model input)."""
    cfg = cfg or _cfg()
    return traj[cfg["strata"]].astype(str).agg("|".join, axis=1)


def make_folds(traj: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Assign outer folds and world-control folds. Deterministic for a given seed and corpus."""
    cfg = cfg or _cfg()
    g, w = cfg["group_key"], cfg["world_key"]
    keep = ["traj_id", "origin_split", g, cfg["domain_key"], w] + [c for c in cfg["strata"] if c != cfg["domain_key"]]
    out = traj[keep].copy().reset_index(drop=True)
    out["origin_split"] = out["origin_split"].astype(str)
    out["fold"] = TEST_FOLD
    out["world_fold"] = TEST_FOLD
    dev = out.index[out["origin_split"] != "test"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=cfg["outer_folds"], shuffle=True, random_state=cfg["seed"])
    y = strata(out.loc[dev], cfg)
    for k, (_, val) in enumerate(sgkf.split(dev, y, groups=out.loc[dev, g])):
        out.loc[dev[val], "fold"] = k

    # GroupKFold has no random_state: shuffle the group order with the seed for a reproducible permutation.
    rng = np.random.default_rng(cfg["seed"])
    worlds = out.loc[dev, w].unique()
    order = dict(zip(worlds, rng.permutation(len(worlds))))
    for k, (_, val) in enumerate(GroupKFold(n_splits=cfg["outer_folds"]).split(dev, groups=out.loc[dev, w].map(order))):
        out.loc[dev[val], "world_fold"] = k
    return out


def folds_hash(folds: pd.DataFrame) -> str:
    """Short fingerprint of the assignment, to check that every member uses the same folds."""
    s = folds.sort_values("traj_id")[["traj_id", "fold", "world_fold"]].to_csv(index=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_folds() -> pd.DataFrame:
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"{FOLDS_PATH} missing: run `python -m agentdrift.splits.grouped` first")
    return pd.read_parquet(FOLDS_PATH)


def attach_folds(df: pd.DataFrame, folds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `fold` and `world_fold` to a step or trajectory table (by traj_id)."""
    folds = load_folds() if folds is None else folds
    return df.merge(folds[["traj_id", "fold", "world_fold"]], on="traj_id", how="left", validate="many_to_one")


def outer_splits(df: pd.DataFrame, column: str = "fold") -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (k, train_positions, val_positions) over dev rows of a table that has `column` (see attach_folds).
    Rows of the fixed test set are never yielded."""
    f = df[column].to_numpy()
    for k in sorted(set(f[f != TEST_FOLD])):
        yield int(k), np.flatnonzero((f != k) & (f != TEST_FOLD)), np.flatnonzero(f == k)


def lodo_splits(df: pd.DataFrame, domain_key: str = "domain") -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-domain-out over dev rows: train on four domains, evaluate on the fifth."""
    dev = (df["fold"] != TEST_FOLD).to_numpy() if "fold" in df else (df["origin_split"].astype(str) != "test").to_numpy()
    dom = df[domain_key].to_numpy()
    for d in sorted(set(dom[dev])):
        yield d, np.flatnonzero(dev & (dom != d)), np.flatnonzero(dev & (dom == d))


def inner_splits(df: pd.DataFrame, n_splits: int | None = None, seed: int | None = None,
                 group_key: str = "template_id") -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Nested CV inside a training part: StratifiedGroupKFold by template, stratified like the outer folds.
    Positions are relative to `df` (pass the training part only)."""
    cfg = _cfg()
    sgkf = StratifiedGroupKFold(n_splits=n_splits or cfg["inner_folds"], shuffle=True,
                                random_state=cfg["seed"] if seed is None else seed)
    y = strata(df, cfg)
    yield from sgkf.split(np.zeros(len(df)), y, groups=df[group_key])


def fold_summary(folds: pd.DataFrame, traj: pd.DataFrame, steps: pd.DataFrame, column: str = "fold") -> pd.DataFrame:
    """Size and prevalence per fold (trajectory and step level)."""
    t = traj.merge(folds[["traj_id", column]], on="traj_id")
    s = steps.merge(folds[["traj_id", column]], on="traj_id")
    out = t.groupby(column).agg(
        trajectories=("traj_id", "size"), templates=("template_id", "nunique"), worlds=("world_id", "nunique"),
        y_traj=("y_traj", "mean"),
    ).join(s.groupby(column).agg(steps=("t", "size"), y_scan=("y_scan", "mean"), y_gate=("y_gate", "mean"),
                                 first_h=("is_first_h", "sum")))
    cats = pd.crosstab(t[column], t["source_category"], normalize="index")
    doms = pd.crosstab(t[column], t["domain"], normalize="index")
    return out.join(cats.add_prefix("share ")).join(doms.add_prefix("share "))


def main() -> int:
    traj = read_trajectories(include_test=True)
    steps = read_steps(include_test=True)
    folds = make_folds(traj)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(FOLDS_PATH, index=False)
    print(f"Wrote {FOLDS_PATH.relative_to(ROOT)} (hash {folds_hash(folds)})")
    summ = fold_summary(folds, traj, steps)
    cols = ["trajectories", "templates", "worlds", "steps", "y_traj", "y_scan", "y_gate", "first_h"]
    print(summ[cols].round(3).to_string())
    print("\nWorld-control folds:\n" + fold_summary(folds, traj, steps, "world_fold")[cols].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
