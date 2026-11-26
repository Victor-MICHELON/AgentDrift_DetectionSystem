"""AutoML v0.2 (B.6, KAN-21): nested search on both views, 5 folds by template and leave one domain out, then report.

    python -m agentdrift.pipelines.run_automl                 # run (≈ 2 h), then tables + figure
    python -m agentdrift.pipelines.run_automl --tables-only

Runs: results/runs/v0_2_automl/<view>[_lodo]/ (harness layout). Tables in results/runs/v0_2_study/:
    choices.csv           per outer split: chosen family, blocks, TF-IDF configuration, parameters, inner PR-AUC,
                          trials complete / pruned, timeout hit, search time
    best_by_family.csv    per outer split and family: inner PR-AUC of its best trial (how close the families are)
    recall_same_fpr.csv   DIAGNOSTIC: recall at 1 % FPR with the threshold set on each evaluated split, by attack type,
                          v0.1 against v0.2 (separates ranking from threshold transfer, see run_h1)
and results/runs/comparisons/v0_2.csv: position only, v0.1, v0.2 in both modes, paired difference to v0.1.
Figure: results/figures/b_v02_comparison.png (+ PDF).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from agentdrift.config import CONFIGS, RUNS
from agentdrift.models.experiment import ExperimentConfig, run_experiment
from agentdrift.pipelines.run_baseline import comparison_rows
from agentdrift.splits.grouped import load_folds

log = logging.getLogger(__name__)
CONFIG = CONFIGS / "models" / "v0_2_automl.yaml"
OUT = RUNS / "v0_2_study"
MODES = ("cv", "lodo")
COMPARED = ("v0_0_position", "v0_1_baseline", "v0_2_automl")


def run_dir(version: str, view: str, mode: str):
    return RUNS / version / (view if mode == "cv" else f"{view}_{mode}")


def _metrics(version: str, view: str, mode: str) -> dict | None:
    p = run_dir(version, view, mode) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def choices_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fam_rows = [], []
    for view in ("scanner", "gate"):
        for mode in MODES:
            m = _metrics("v0_2_automl", view, mode)
            if m is None:
                continue
            for f in m["folds"]:
                s = f["selected"]
                rows.append({"view": view, "mode": mode, "split": f["split"], "family": s["family"],
                             "blocks": s["blocks"], "text": s["text"], "inner_pr_auc": s["inner_pr_auc"],
                             "eval_pr_auc": f["pr_auc"], "complete": s["trials"]["complete"],
                             "pruned": s["trials"]["pruned"], "failed": s["trials"]["failed"],
                             "timeout_hit": s["timeout_hit"], "search_seconds": s["search_seconds"],
                             "fold_seconds": f["seconds"],
                             "params": json.dumps({k: v for k, v in s["params"].items()
                                                   if k not in ("family", "blocks", "text")})})
                for fam, b in s["best_by_family"].items():
                    fam_rows.append({"view": view, "mode": mode, "split": f["split"], "family": fam,
                                     "inner_pr_auc": b["inner_pr_auc"], "blocks": b["params"]["blocks"]})
    return pd.DataFrame(rows), pd.DataFrame(fam_rows)


def comparison() -> pd.DataFrame:
    entries, ref = [], {}
    for version in COMPARED:
        for view in ("scanner", "gate"):
            for mode in MODES:
                m = _metrics(version, view, mode)
                if m is None:
                    continue
                oof = pd.read_parquet(run_dir(version, view, mode) / "oof.parquet")
                entries.append((m, oof))
                if version == "v0_1_baseline":
                    ref[(view, mode)] = oof
    return comparison_rows(entries, "v0_1_baseline", ref)


def recall_same_fpr(fpr: float = 0.01) -> pd.DataFrame:
    src = load_folds().set_index("traj_id")["source_category"]
    rows = []
    for version in ("v0_1_baseline", "v0_2_automl"):
        for view in ("scanner", "gate"):
            for mode in MODES:
                p = run_dir(version, view, mode) / "oof.parquet"
                if not p.exists():
                    continue
                o = pd.read_parquet(p)
                o["cat"] = o["trajectory_id"].map(src)
                acc: dict[str, list[float]] = {k: [] for k in ("all", "full", "partial", "delayed")}
                for _, d in o.groupby("split"):
                    neg = np.sort(d.loc[d["label"] == 0, "score"].to_numpy())[::-1]
                    thr = neg[min(int(fpr * len(neg)), len(neg) - 1)]
                    pos = d[d["label"] == 1]
                    acc["all"].append(float((pos["score"] > thr).mean()))
                    for c in ("full", "partial", "delayed"):
                        q = pos[pos["cat"] == f"attacked_{c}"]
                        if len(q):
                            acc[c].append(float((q["score"] > thr).mean()))
                rows.append({"version": version, "view": view, "mode": mode,
                             **{f"recall_same_fpr_{k}": float(np.mean(v)) for k, v in acc.items()}})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoML v0.2: nested search, both views, both modes, report.")
    parser.add_argument("--tables-only", action="store_true", help="no run: tables and figure from disk")
    parser.add_argument("--force", action="store_true", help="rerun even if results exist")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = ExperimentConfig.from_yaml(CONFIG)
    if not args.tables_only:
        for mode in args.modes:
            for view in cfg.views:
                if args.force or _metrics(cfg.version, view, mode) is None:
                    log.info("running v0.2 %s %s", view, mode)
                    run_experiment(replace(cfg, evaluation=mode), [view])
    OUT.mkdir(parents=True, exist_ok=True)
    choices, families = choices_tables()
    comp = comparison()
    same = recall_same_fpr()
    choices.to_csv(OUT / "choices.csv", index=False)
    families.to_csv(OUT / "best_by_family.csv", index=False)
    same.to_csv(OUT / "recall_same_fpr.csv", index=False)
    (RUNS / "comparisons").mkdir(parents=True, exist_ok=True)
    comp.to_csv(RUNS / "comparisons" / "v0_2.csv", index=False)
    from agentdrift.viz import plots as P
    P.setup()
    if len(comp):
        P.save(P.plot_baselines(comp), "b_v02_comparison")
    fmt = {"float_format": lambda v: f"{v:.3f}", "index": False}
    with pd.option_context("display.width", 240, "display.max_columns", 30, "display.max_colwidth", 70):
        print("=== choices per outer split\n" + (choices.drop(columns=["params"]).to_string(**fmt) if len(choices)
                                                  else "(no v0.2 run yet)"))
        if len(families):
            print("\n=== best inner PR-AUC per family (mean over outer splits)\n"
                  + families.groupby(["view", "mode", "family"])["inner_pr_auc"].agg(["mean", "min", "max", "size"])
                  .reset_index().to_string(**fmt))
        cols = [c for c in ("version", "view", "evaluation", "pr_auc", "roc_auc", "recall@fpr0.01", "fpr@fpr0.01",
                            "recall@fpr0.002", "delta_pr_auc", "delta_pr_auc_low", "delta_pr_auc_high",
                            "delta_recall@fpr0.01", "delta_recall@fpr0.002") if c in comp]
        print("\n=== comparison (paired difference to v0.1)\n" + comp[cols].to_string(**fmt))
        print("\n=== recall at 1 % FPR, threshold on each evaluated split (diagnostic)\n" + same.to_string(**fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
