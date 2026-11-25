"""Test of H1 (B.7, KAN-62): do word-free features generalise better than words? And do they add to the words?

    python -m agentdrift.pipelines.run_h1                  # run the missing variants, then tables + figure
    python -m agentdrift.pipelines.run_h1 --tables-only

Same estimator for every variant (the v0.1 estimator, `configs/models/v0_1_baseline.yaml`: same C grid, Platt
calibration, thresholds from the inner scores), only the feature blocks change:

    text              TF-IDF per field: v0.1 itself (its runs are reused, not recomputed)
    struct            risk-category structure (+ argument provenance on the gate): NO word at all
    struct_cues       struct + injection cues (hand-written instruction patterns and text length: lexical and
                      stylistic, so not word-free; the plan's "structure" variant)
    struct_tool       struct + tool names (control: names do not exist in a left-out domain)
    struct_position   struct + step index (control, `kind: shortcut`: how much of struct is position)
    text_struct       text + struct + cues (complementarity: does structure add to the words?)

Two evaluation modes each: 5 folds by template (cv) and leave one domain out (lodo).
Tables in results/runs/h1/:
    runs.csv            variant x view x mode: PR-AUC, ROC-AUC, recall@1 % with 95 % CI, observed FPR, recall by
                        attack type, alert rate on hard negatives
    losses.csv          generalisation loss (lodo - cv) of each variant against the text one, paired template
                        bootstrap of the difference of losses (`experiment.paired_loss_difference`). ROC-AUC is in the
                        table because PR-AUC moves with the prevalence, which differs between domains (banking 34 %)
    complementarity.csv text_struct - text in each mode (`experiment.paired_difference`), with recall by attack type
    recall_same_fpr.csv DIAGNOSTIC: recall at 1 % FPR with the threshold set on the negatives of each evaluated split
                        (fold or left-out domain). Not deployable (it reads the evaluated labels): it separates the
                        ranking from the threshold transfer, since transferred thresholds give different observed FPR
                        to different variants (e.g. 0.3 % for text, 2.2 % for struct on the gate, left-out domain)
Figure: results/figures/b_h1_generalisation.png (+ PDF in report/latex/figures/).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from agentdrift.config import CONFIGS, RUNS
from agentdrift.models.experiment import ExperimentConfig, paired_difference, paired_loss_difference, run_experiment

log = logging.getLogger(__name__)

BASE = CONFIGS / "models" / "v0_1_baseline.yaml"
OUT = RUNS / "h1"
MODES = ("cv", "lodo")
METRICS = ("pr_auc", "roc_auc", "recall@fpr0.01")
STRUCT = {"scanner": ["structure"], "gate": ["structure", "provenance"]}
VARIANTS = {
    "struct": {v: b for v, b in STRUCT.items()},
    "struct_cues": {v: b + ["cues"] for v, b in STRUCT.items()},
    "struct_tool": {v: b + ["tool_name"] for v, b in STRUCT.items()},
    "struct_position": {v: b + ["position"] for v, b in STRUCT.items()},
    "text_struct": {v: ["text"] + b + ["cues"] for v, b in STRUCT.items()},
}
LABELS = {"text": "text (v0.1)", "struct": "structure (no word)", "struct_cues": "structure + cues",
          "struct_tool": "structure + tool name", "struct_position": "structure + position (control)",
          "text_struct": "text + structure + cues"}


def config_of(base: ExperimentConfig, variant: str, view: str, mode: str) -> ExperimentConfig:
    model = copy.deepcopy(base.model)
    model["params"] = {**model.get("params", {}), "blocks": VARIANTS[variant][view]}
    shortcut = variant == "struct_position"
    return replace(base, version=f"h1/{variant}", views=[view], model=model, evaluation=mode,
                   kind="shortcut" if shortcut else base.kind, uses_position=shortcut,
                   description=f"H1 {variant}: blocks {VARIANTS[variant][view]}")


def run_dir(variant: str, view: str, mode: str, base: ExperimentConfig):
    version = base.version if variant == "text" else f"h1/{variant}"
    return RUNS / version / (view if mode == "cv" else f"{view}_{mode}")


def run_variants(base: ExperimentConfig, force: bool, only: list[str] | None) -> None:
    for variant in VARIANTS:
        if only and variant not in only:
            continue
        for view in ("scanner", "gate"):
            for mode in MODES:
                if force or not (run_dir(variant, view, mode, base) / "metrics.json").exists():
                    log.info("running H1 %s %s %s", variant, view, mode)
                    run_experiment(config_of(base, variant, view, mode), [view])


def _metrics(variant: str, view: str, mode: str, base: ExperimentConfig) -> dict | None:
    p = run_dir(variant, view, mode, base) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _oof(variant: str, view: str, mode: str, base: ExperimentConfig) -> pd.DataFrame:
    return pd.read_parquet(run_dir(variant, view, mode, base) / "oof.parquet")


def runs_table(base: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for variant in ("text", *VARIANTS):
        for view in ("scanner", "gate"):
            for mode in MODES:
                m = _metrics(variant, view, mode, base)
                if m is None:
                    continue
                e, sub = m["estimate"], m["subgroups"]["fpr0.01"]
                row = {"variant": variant, "label": LABELS[variant], "view": view, "mode": mode,
                       "prevalence": m["data"]["prevalence"], "fpr@fpr0.01": e["fpr@fpr0.01"]["point"]}
                for k in (*METRICS, "recall@fpr0.002"):
                    row |= {k: e[k]["point"], f"{k}_low": e[k]["low"], f"{k}_high": e[k]["high"]}
                for cat, v in sub["recall_by_attack_type"].items():
                    row[f"recall_{cat.replace('attacked_', '')}"] = v["recall"]
                row["alert_hard_negative"] = sub["alert_rate_by_class_of_negatives"].get("hard_negative", {}).get(
                    "alert_rate")
                rows.append(row)
    return pd.DataFrame(rows)


def losses_table(base: ExperimentConfig, n_resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    for view in ("scanner", "gate"):
        if _metrics("text", view, "lodo", base) is None:
            continue
        cv_t, out_t = _oof("text", view, "cv", base), _oof("text", view, "lodo", base)
        for variant in VARIANTS:
            if any(_metrics(variant, view, m, base) is None for m in MODES):
                continue
            d = paired_loss_difference(_oof(variant, view, "cv", base), _oof(variant, view, "lodo", base), cv_t, out_t,
                                       [0.01, 0.002], n_resamples, seed)
            for metric in METRICS:
                r = d[metric]
                rows.append({"view": view, "variant": variant, "metric": metric, "loss_variant": r["loss_a"],
                             "loss_text": r["loss_b"], "loss_difference": r["point"], "low": r["low"],
                             "high": r["high"], "excludes_zero": bool(r["low"] > 0 or r["high"] < 0)})
    return pd.DataFrame(rows)


def complementarity_table(base: ExperimentConfig, runs: pd.DataFrame, n_resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    for view in ("scanner", "gate"):
        for mode in MODES:
            if _metrics("text_struct", view, mode, base) is None:
                continue
            d = paired_difference(_oof("text_struct", view, mode, base), _oof("text", view, mode, base),
                                  [0.01, 0.002], n_resamples, seed)
            r = runs.set_index(["variant", "view", "mode"])
            row = {"view": view, "mode": mode}
            for metric in (*METRICS, "recall@fpr0.002"):
                row |= {f"delta_{metric}": d[metric]["point"], f"delta_{metric}_low": d[metric]["low"],
                        f"delta_{metric}_high": d[metric]["high"]}
            for cat in ("full", "partial", "delayed"):
                col = f"recall_{cat}"
                if col in r.columns:
                    row[f"delta_{col}"] = r.loc[("text_struct", view, mode), col] - r.loc[("text", view, mode), col]
            rows.append(row)
    return pd.DataFrame(rows)


def recall_same_fpr(base: ExperimentConfig, fpr: float = 0.01) -> pd.DataFrame:
    """Recall at `fpr` with a threshold set on the negatives of each evaluated split, by attack type (diagnostic)."""
    from agentdrift.splits.grouped import load_folds
    src = load_folds().set_index("traj_id")["source_category"]
    rows = []
    for variant in ("text", *VARIANTS):
        for view in ("scanner", "gate"):
            for mode in MODES:
                if _metrics(variant, view, mode, base) is None:
                    continue
                o = _oof(variant, view, mode, base)
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
                rows.append({"variant": variant, "view": view, "mode": mode,
                             **{f"recall_same_fpr_{k}": float(np.mean(v)) if v else None for k, v in acc.items()}})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H1: word-free features against words, in and out of domain.")
    parser.add_argument("--only", nargs="+", choices=list(VARIANTS), help="variants to run (default: all)")
    parser.add_argument("--force", action="store_true", help="rerun variants that already have results")
    parser.add_argument("--tables-only", action="store_true", help="no run: tables and figure from disk")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base = ExperimentConfig.from_yaml(BASE)
    if not args.tables_only:
        run_variants(base, args.force, args.only)
    OUT.mkdir(parents=True, exist_ok=True)
    n, seed = base.bootstrap["n_resamples"], base.bootstrap["seed"]
    runs = runs_table(base)
    losses = losses_table(base, n, seed)
    comp = complementarity_table(base, runs, n, seed)
    same = recall_same_fpr(base)
    for name, t in (("runs", runs), ("losses", losses), ("complementarity", comp), ("recall_same_fpr", same)):
        t.to_csv(OUT / f"{name}.csv", index=False)
    from agentdrift.viz import plots as P
    P.setup()
    if len(runs):
        P.save(P.plot_h1(runs, [LABELS[v] for v in ("text", *VARIANTS)]), "b_h1_generalisation")
    fmt = {"float_format": lambda v: f"{v:.3f}", "index": False}
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["variant", "view", "mode", "pr_auc", "roc_auc", "recall@fpr0.01", "fpr@fpr0.01", "recall_full",
                "recall_partial", "recall_delayed"]
        print("=== runs\n" + runs[[c for c in cols if c in runs]].to_string(**fmt))
        print("\n=== losses (lodo - cv), variant against text\n" + losses.to_string(**fmt))
        print("\n=== complementarity (text_struct - text)\n" + comp.to_string(**fmt))
        print("\n=== recall at 1 % FPR, threshold on each evaluated split (diagnostic)\n" + same.to_string(**fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
