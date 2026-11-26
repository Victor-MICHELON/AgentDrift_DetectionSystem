"""Study of the v0.1 baseline (KAN-20): calibration, threshold source, fields, gate "action only", learning curves,
coefficients. Every variant is the official config `configs/models/v0_1_baseline.yaml` with ONE change.

    python -m agentdrift.pipelines.run_v01_study                 # run the missing variants, then tables + figures
    python -m agentdrift.pipelines.run_v01_study --only cal fields --force
    python -m agentdrift.pipelines.run_v01_study --tables-only   # rebuild tables and figures from the runs on disk

Runs: results/runs/v0_1_study/<variant>/<view>[_lodo]/ (same layout as any version). Tables in results/runs/v0_1_study/:

    calibration.csv     none / sigmoid (official) / isotonic / unweighted: ranking metrics, Brier, ECE, C per fold
    thresholds.csv      scanner: thresholds from the estimator's inner scores (official) vs refitted inner folds
    fields.csv          one field at a time, all fields, gate action only (CV and leave one domain out)
    learning_curve.csv  10 / 25 / 50 / 100 % of the training templates of each fold
    reliability.csv     equal-mass reliability bins of each calibration variant (out-of-fold, pooled over folds:
                        calibration concerns the probability values, not the ranking across models)
    coefficients_<view>.csv  top 30 + / - coefficients per field, fitted on the training part of fold cv0, to READ
Figures: results/figures/b_v01_*.png (+ PDF in report/latex/figures/).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
from dataclasses import dataclass, field, replace

import pandas as pd

from agentdrift.config import CONFIGS, RUNS
from agentdrift.evaluation.metrics import reliability_table
from agentdrift.features.views import feature_columns
from agentdrift.models.baselines import top_coefficients
from agentdrift.models.experiment import ExperimentConfig, load_dataset, make_estimator, outer, run_experiment

log = logging.getLogger(__name__)

BASE = CONFIGS / "models" / "v0_1_baseline.yaml"
STUDY = "v0_1_study"
OUT = RUNS / STUDY
ACTION = ["task", "proposed_call_tool", "proposed_call_args"]
FRACTIONS = (0.1, 0.25, 0.5)
GROUPS = ("cal", "thr", "fields", "action", "lc")


@dataclass
class Variant:
    tag: str
    group: str
    views: list[str]
    evaluations: list[str] = field(default_factory=lambda: ["cv"])
    params: dict = field(default_factory=dict)  # merged into model.params
    changes: dict = field(default_factory=dict)  # ExperimentConfig fields


def variants() -> list[Variant]:
    out = [
        Variant("cal_none", "cal", ["scanner", "gate"], params={"calibration": "none"}),
        Variant("cal_isotonic", "cal", ["scanner", "gate"], params={"calibration": "isotonic"}),
        Variant("unweighted_none", "cal", ["scanner", "gate"], params={"class_weight": None, "calibration": "none"}),
        Variant("thr_inner_oof", "thr", ["scanner"], changes={"threshold_source": "inner_oof"}),
        Variant("action", "action", ["gate"], ["cv", "lodo"], changes={"columns": ACTION}),
    ]
    for view in ("scanner", "gate"):
        out += [Variant(f"field_{view}_{f}", "fields", [view], changes={"columns": [f]}) for f in feature_columns(view)]
    out += [Variant(f"lc_{int(p * 100)}", "lc", ["scanner", "gate"], changes={"train_fraction": p}) for p in FRACTIONS]
    return out


def config_of(base: ExperimentConfig, v: Variant) -> ExperimentConfig:
    model = copy.deepcopy(base.model)
    model["params"] = {**model.get("params", {}), **v.params}
    return replace(base, version=f"{STUDY}/{v.tag}", views=v.views, model=model,
                   description=f"v0.1 study: {v.tag} ({v.group})", **v.changes)


def metrics_path(version: str, view: str, evaluation: str = "cv"):
    return RUNS / version / (view if evaluation == "cv" else f"{view}_{evaluation}") / "metrics.json"


def run_variants(base: ExperimentConfig, groups: set[str], force: bool, n_resamples: int | None) -> None:
    for v in variants():
        if v.group not in groups:
            continue
        for ev in v.evaluations:
            cfg = replace(config_of(base, v), evaluation=ev)
            if n_resamples:
                cfg = replace(cfg, bootstrap={**cfg.bootstrap, "n_resamples": n_resamples})
            todo = [view for view in v.views if force or not metrics_path(cfg.version, view, ev).exists()]
            if todo:
                log.info("running %s %s on %s", cfg.version, ev, todo)
                run_experiment(cfg, todo)


# ----------------------------------------------------------------------------- tables


def _load(version: str, view: str, evaluation: str = "cv") -> dict | None:
    p = metrics_path(version, view, evaluation)
    return json.loads(p.read_text()) if p.exists() else None


def _row(m: dict) -> dict:
    e, fs, sub = m["estimate"], m["fold_summary"], m["subgroups"]["fpr0.01"]["recall_by_attack_type"]
    row = {"rows": m["data"]["rows"], "prevalence": m["data"]["prevalence"]}
    for k in ("pr_auc", "roc_auc", "recall@fpr0.01", "fpr@fpr0.01", "recall@fpr0.002", "fpr@fpr0.002"):
        row |= {k: e[k]["point"], f"{k}_low": e[k]["low"], f"{k}_high": e[k]["high"]}
    row |= {"brier": fs.get("brier", {}).get("mean"), "ece": fs.get("ece", {}).get("mean"),
            "train_pr_auc": fs["train_pr_auc"]["mean"], "gap_pr_auc": fs["gap_pr_auc"]["mean"],
            "C_per_fold": ",".join(str((f.get("selected") or {}).get("C", "")) for f in m["folds"])}
    for cat in ("attacked_full", "attacked_partial", "attacked_delayed", "failed_attack"):
        if cat in sub:
            row[f"recall@fpr0.01_{cat}"] = sub[cat]["recall"]
    return row


def calibration_table(base: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for view in ("scanner", "gate"):
        for name, version in [("sigmoid (official)", base.version), ("none", f"{STUDY}/cal_none"),
                              ("isotonic", f"{STUDY}/cal_isotonic"), ("unweighted, none", f"{STUDY}/unweighted_none")]:
            if (m := _load(version, view)) is not None:
                rows.append({"view": view, "variant": name, **_row(m)})
    return pd.DataFrame(rows)


def thresholds_table(base: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for name, version in [("estimator (official)", base.version), ("inner_oof", f"{STUDY}/thr_inner_oof")]:
        if (m := _load(version, "scanner")) is not None:
            r = _row(m)
            rows.append({"view": "scanner", "threshold_source": name, "seconds": m["run"]["seconds"],
                         **{k: r[k] for k in r if k.startswith(("recall@", "fpr@"))}})
    return pd.DataFrame(rows)


def fields_table(base: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for view in ("scanner", "gate"):
        for ev in ("cv", "lodo"):
            if (m := _load(base.version, view, ev)) is not None:
                rows.append({"view": view, "evaluation": ev, "fields": "all fields", **_row(m)})
        for f in feature_columns(view):
            if (m := _load(f"{STUDY}/field_{view}_{f}", view)) is not None:
                rows.append({"view": view, "evaluation": "cv", "fields": f, **_row(m)})
    for ev in ("cv", "lodo"):
        if (m := _load(f"{STUDY}/action", "gate", ev)) is not None:
            rows.append({"view": "gate", "evaluation": ev, "fields": "action only", **_row(m)})
    return pd.DataFrame(rows)


def learning_curve_table(base: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for view in ("scanner", "gate"):
        for p, version in [(p, f"{STUDY}/lc_{int(p * 100)}") for p in FRACTIONS] + [(1.0, base.version)]:
            if (m := _load(version, view)) is not None:
                tpl = sum(f["templates_train"] for f in m["folds"] if "templates_train" in f) / len(m["folds"])
                rows.append({"view": view, "train_fraction": p, "templates_train": tpl, **_row(m)})
    return pd.DataFrame(rows)


def reliability(base: ExperimentConfig) -> pd.DataFrame:
    parts = []
    for view in ("scanner", "gate"):
        for name, version in [("sigmoid (official)", base.version), ("none", f"{STUDY}/cal_none"),
                              ("isotonic", f"{STUDY}/cal_isotonic"), ("unweighted, none", f"{STUDY}/unweighted_none")]:
            p = metrics_path(version, view).parent / "oof.parquet"
            if p.exists():
                oof = pd.read_parquet(p)
                parts.append(reliability_table(oof["score"], oof["label"]).assign(view=view, variant=name))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def coefficients(base: ExperimentConfig, k: int = 30) -> dict[str, pd.DataFrame]:
    """v0.1 fitted on the training part of fold cv0 (the evaluated fold is never used here either)."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="agentdrift_cache_") as cache:
        for view in base.views:
            data = load_dataset(view, base)
            _, train, _ = next(outer(data.meta, "cv"))
            est = make_estimator(base, data.meta.iloc[train], base.seed, view, cache)
            est.fit(data.X.iloc[train], data.y[train])
            out[view] = top_coefficients(est, k).assign(C=est.selected_["C"])
    return out


def figures(tables: dict[str, pd.DataFrame]) -> None:
    from agentdrift.viz import plots as P
    P.setup()
    comp = RUNS / "comparisons" / "v0_0_v0_1.csv"
    if comp.exists():
        P.save(P.plot_baselines(pd.read_csv(comp)), "b_v0_baselines")
    if len(tables["reliability"]):
        P.save(P.plot_reliability(tables["reliability"]), "b_v01_reliability")
    f = tables["fields"]
    if len(f):
        P.save(P.plot_field_ablation(f[f["evaluation"] == "cv"]), "b_v01_fields")
    if len(tables["learning_curve"]):
        P.save(P.plot_learning_curve(tables["learning_curve"]), "b_v01_learning_curve")


def build_tables(base: ExperimentConfig, with_coefficients: bool = True) -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"calibration": calibration_table(base), "thresholds": thresholds_table(base),
              "fields": fields_table(base), "learning_curve": learning_curve_table(base),
              "reliability": reliability(base)}
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
    if with_coefficients:
        for view, t in coefficients(base).items():
            t.to_csv(OUT / f"coefficients_{view}.csv", index=False)
    figures(tables)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v0.1 study: one change at a time from v0_1_baseline.yaml.")
    parser.add_argument("--only", nargs="+", choices=GROUPS, help="variant groups to run (default: all)")
    parser.add_argument("--force", action="store_true", help="rerun variants that already have results")
    parser.add_argument("--tables-only", action="store_true", help="no run: tables and figures from disk")
    parser.add_argument("--no-coefficients", action="store_true", help="skip the coefficient fit (~1 min per view)")
    parser.add_argument("--n-resamples", type=int, help="override the bootstrap resamples (quick checks only)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base = ExperimentConfig.from_yaml(BASE)
    if not args.tables_only:
        run_variants(base, set(args.only or GROUPS), args.force, args.n_resamples)
    tables = build_tables(base, not args.no_coefficients)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        for name in ("calibration", "thresholds", "fields", "learning_curve"):
            t = tables[name]
            cols = [c for c in t.columns if c in ("view", "variant", "threshold_source", "evaluation", "fields",
                                                  "train_fraction", "pr_auc", "roc_auc", "recall@fpr0.01",
                                                  "fpr@fpr0.01", "brier", "ece", "gap_pr_auc", "C_per_fold",
                                                  "recall@fpr0.01_attacked_partial", "seconds")]
            print(f"\n=== {name}\n" + (t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(t)
                                       else "(no run yet)"))
    print(f"\ntables in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
