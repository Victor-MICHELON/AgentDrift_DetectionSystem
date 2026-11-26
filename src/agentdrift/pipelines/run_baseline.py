"""Run several model versions and write one comparison table (B.5, KAN-20; harness extension of KAN-61).

    python -m agentdrift.pipelines.run_baseline --name v0_0 \\
        --configs configs/models/v0_0_majority.yaml configs/models/v0_0_position.yaml \\
        --evaluations cv lodo --reference v0_0_majority

Each config is run on each of its views and evaluation modes through `models.experiment.run_experiment` (results in
`results/runs/<version>/...`). The table `results/runs/comparisons/<name>.csv` has one row per (version, view,
evaluation): the headline metrics with their 95 % template-bootstrap interval, and, when `--reference` names a version
that has run on the same view and evaluation, the PAIRED difference to it (same rows, same template draws:
`experiment.paired_difference`). Without a reference the difference columns are empty.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from agentdrift.config import ROOT, RUNS
from agentdrift.models.experiment import EVALUATIONS, ExperimentConfig, paired_difference, run_experiment

log = logging.getLogger(__name__)

COMPARISONS = RUNS / "comparisons"
HEADLINE = ("pr_auc", "roc_auc")


def _metric_names(report: dict) -> list[str]:
    names = [f"fpr{f:g}" for f in report["config"]["target_fpr"]]
    return [*HEADLINE, *[f"{p}@{n}" for n in names for p in ("recall", "fpr")]]


def comparison_rows(entries: list[tuple[dict, pd.DataFrame]], reference: str | None = None,
                    reference_oof: dict[tuple[str, str], pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per (report, out-of-fold predictions) pair.

    `reference_oof` maps (view, evaluation) to the out-of-fold predictions of the reference version. Entries of the
    reference version itself get no difference. A pair that cannot be paired (other rows) gets an empty difference
    and a warning, never a silent one."""
    rows = []
    for report, oof in entries:
        cfg = report["config"]
        row = {"version": report["version"], "kind": report["kind"], "text": report["data"]["text"],
               "view": report["view"], "evaluation": report["evaluation"], "rows": report["data"]["rows"],
               "prevalence": report["data"]["prevalence"]}
        for m in _metric_names(report):
            e = report["estimate"][m]
            row |= {m: e["point"], f"{m}_low": e["low"], f"{m}_high": e["high"]}
        ref = (reference_oof or {}).get((report["view"], report["evaluation"]))
        if reference and report["version"] != reference and ref is not None:
            try:
                diff = paired_difference(oof, ref, cfg["target_fpr"], cfg["bootstrap"]["n_resamples"],
                                         cfg["bootstrap"]["seed"])
                for m in (*HEADLINE, *[f"recall@fpr{f:g}" for f in cfg["target_fpr"]]):
                    row |= {f"delta_{m}": diff[m]["point"], f"delta_{m}_low": diff[m]["low"],
                            f"delta_{m}_high": diff[m]["high"]}
            except ValueError as err:
                log.warning("%s %s %s not paired with %s: %s", report["version"], report["view"],
                            report["evaluation"], reference, err)
        rows.append(row)
    return pd.DataFrame(rows)


def read_reference(version: str, views: list[str], evaluations: list[str],
                   root: Path = RUNS) -> dict[tuple[str, str], pd.DataFrame]:
    """Out-of-fold predictions of an earlier run (`results/runs/<version>/<view>[_<evaluation>]/oof.parquet`)."""
    out = {}
    for view in views:
        for ev in evaluations:
            path = root / version / (view if ev == "cv" else f"{view}_{ev}") / "oof.parquet"
            if path.exists():
                out[(view, ev)] = pd.read_parquet(path)
            else:
                log.warning("reference %s has no run at %s", version, path)
    return out


def _show(table: pd.DataFrame) -> str:
    cols = [c for c in ("version", "kind", "view", "evaluation", "pr_auc", "roc_auc", "recall@fpr0.01",
                        "fpr@fpr0.01", "recall@fpr0.002", "delta_pr_auc", "delta_pr_auc_low", "delta_pr_auc_high")
            if c in table]
    return table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run model versions and write a comparison table.")
    parser.add_argument("--configs", nargs="+", required=True, help="YAML files of configs/models/")
    parser.add_argument("--name", required=True, help="table name: results/runs/comparisons/<name>.csv")
    parser.add_argument("--evaluations", nargs="+", choices=EVALUATIONS, default=["cv"])
    parser.add_argument("--reference", help="version the paired differences are taken against (it must have run "
                                            "on the same views and evaluations, here or earlier)")
    parser.add_argument("--n-resamples", type=int, help="override the bootstrap resamples (quick checks only)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    configs = [ExperimentConfig.from_yaml(p) for p in args.configs]
    if args.n_resamples:
        configs = [replace(c, bootstrap={**c.bootstrap, "n_resamples": args.n_resamples}) for c in configs]
    runs: list[tuple[dict, Path]] = []
    for cfg in configs:
        for ev in args.evaluations:
            c = replace(cfg, evaluation=ev)
            for view, report in run_experiment(c).items():
                runs.append((report, c.output_dir(view)))
    views = sorted({r["view"] for r, _ in runs})
    reference_oof = read_reference(args.reference, views, args.evaluations) if args.reference else None
    entries = [(report, pd.read_parquet(out / "oof.parquet")) for report, out in runs]
    table = comparison_rows(entries, args.reference, reference_oof)
    COMPARISONS.mkdir(parents=True, exist_ok=True)
    path = COMPARISONS / f"{args.name}.csv"
    table.to_csv(path, index=False)
    print(_show(table))
    print(f"\nwritten to {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
