"""Run one model version through the experiment harness (B.0, KAN-61).

    python -m agentdrift.pipelines.run_experiment --config configs/models/v0_0_majority.yaml
    python -m agentdrift.pipelines.run_experiment --config <yaml> --views gate --evaluation lodo

Writes results/runs/<version>/<view>[_<evaluation>]/{oof.parquet, metrics.json} (the parquet is not versioned).
Requires the views (`python -m agentdrift.pipelines.build_views`) and the folds (`python -m agentdrift.splits.grouped`).
See `agentdrift.models.experiment` for the contract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from agentdrift.config import ROOT
from agentdrift.features.views import VIEWS
from agentdrift.models.experiment import EVALUATIONS, ExperimentConfig, run_experiment


def _fmt(m: dict) -> str:
    return f"{m['point']:.3f} [{m['low']:.3f}, {m['high']:.3f}]" if m["point"] is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a model version (configs/models/<version>.yaml).")
    parser.add_argument("--config", required=True, help="path to the version's YAML file")
    parser.add_argument("--views", nargs="+", choices=VIEWS, help="override the config's views")
    parser.add_argument("--evaluation", choices=EVALUATIONS, help="override the config's evaluation mode")
    parser.add_argument("--n-resamples", type=int, help="override the bootstrap resamples (quick checks only)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.evaluation:
        cfg = replace(cfg, evaluation=args.evaluation)
    if args.n_resamples:
        cfg = replace(cfg, bootstrap={**cfg.bootstrap, "n_resamples": args.n_resamples})
    results = run_experiment(cfg, args.views)
    for view, r in results.items():
        p, fs = r["estimate"], r["fold_summary"]
        print(f"\n=== {cfg.version} [{cfg.kind}] {view} ({cfg.evaluation}, text {cfg.text}): "
              f"{r['data']['rows']:,} rows, {r['run']['seconds']}s")
        print(f"PR-AUC  {_fmt(p['pr_auc'])} (mean over splits, 95 % CI) | std over splits {fs['pr_auc']['std'] or 0:.3f}")
        print(f"ROC-AUC {_fmt(p['roc_auc'])} | std over splits {fs['roc_auc']['std'] or 0:.3f}")
        for f in cfg.target_fpr:
            n = f"fpr{f:g}"
            print(f"recall@{n} {_fmt(p[f'recall@{n}'])} | observed FPR {_fmt(p[f'fpr@{n}'])}")
        print(f"written to {cfg.output_dir(view).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
