"""Test of H3 (B.4, KAN-19): how much of the score comes from generator artefacts? Ablation ladder of CONTEXT §3.4.

    python -m agentdrift.pipelines.build_views --raw          # once: the raw-text views (rungs 2 bis and 3)
    python -m agentdrift.pipelines.run_h3                     # run the missing rungs, then tables + figure
    python -m agentdrift.pipelines.run_h3 --tables-only

Same estimator at every rung (the v0.1 estimator, `configs/models/v0_1_baseline.yaml`), same rows, same folds; only
the text given to the views changes, one piece of information removed per rung:

    2 bis  raw_thought   raw text, WITH the generated reasoning (`thought`: past thoughts in the history, and thought_t
                         for the gate, written before a_t)
    3      raw           raw text, without `thought`                                    removes: the reasoning
    4      anon          world-anonymised text = v0.1 itself (its runs are reused)      removes: the world identity
    5      masked        anon + generator-artefact masking (`features/masking.py`)      removes: openers, policy codes,
                                                                                                 amounts
Rungs 0-2 (bidirectional trajectory model, future, obs_t) are outside B: the views are causal by construction.
The gate "action only" (task, proposed tool, its arguments: no text read by the agent) is run at the rungs raw,
anon (reused from the v0.1 study) and masked: what the gate knows without the injection, and whether THAT depends on
artefacts.

Two evaluation modes each: 5 folds by template (cv) and leave one domain out (lodo). Tables in results/runs/h3/:
    runs.csv            rung x view x mode: PR-AUC, ROC-AUC, recall@1 % with 95 % CI, observed FPR, recall by attack
                        type (positives), alert rate on the negatives of each class
    steps.csv           paired difference between successive rungs (and of every rung against the reference 5),
                        template bootstrap shared by the two runs (`experiment.paired_difference`)
    recall_same_fpr.csv DIAGNOSTIC: recall at 1 % FPR with the threshold set on each evaluated split, by attack type
                        (separates the ranking from the threshold transfer, see run_h1)
    probe.csv           attack-type probe (`data.audit.attack_type_probe`: can the TEXT of an injection tell whether
                        and how it will succeed?) on raw, anonymised and masked tables. A small drop of the detector
                        does NOT show the style is neutralised while this probe stays high (JOURNAL 23/11)
    shortcuts.csv       reference rows: position only and style only (v0.0)
Figure: results/figures/b_h3_ladder.png (+ PDF in report/latex/figures/).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from agentdrift.config import CONFIGS, RUNS
from agentdrift.models.experiment import ExperimentConfig, paired_difference, run_experiment

log = logging.getLogger(__name__)

BASE = CONFIGS / "models" / "v0_1_baseline.yaml"
OUT = RUNS / "h3"
MODES = ("cv", "lodo")
VIEWS = ("scanner", "gate")
METRICS = ("pr_auc", "roc_auc", "recall@fpr0.01", "recall@fpr0.002")
ACTION = ["task", "proposed_call_tool", "proposed_call_args"]
REFERENCE = "masked"


@dataclass(frozen=True)
class Rung:
    tag: str
    step: str
    label: str
    views: tuple[str, ...] = VIEWS
    changes: dict = field(default_factory=dict)  # ExperimentConfig fields
    reuse: str | None = None  # version whose runs ARE this rung (not recomputed)


RUNGS = (
    Rung("raw_thought", "2 bis", "raw + thought", changes={"text": "raw", "include_thought": True}),
    Rung("raw", "3", "raw", changes={"text": "raw"}),
    Rung("anon", "4", "anonymised (v0.1)", reuse="v0_1_baseline"),
    Rung("masked", "5", "anonymised + masked", changes={"text": "masked"}),
)
ACTION_RUNGS = (
    Rung("action_raw", "3", "action only, raw", ("gate",), {"text": "raw", "columns": ACTION}),
    Rung("action_anon", "4", "action only, anonymised", ("gate",), reuse="v0_1_study/action"),
    Rung("action_masked", "5", "action only, masked", ("gate",), {"text": "masked", "columns": ACTION}),
)
ALL = RUNGS + ACTION_RUNGS
BY_TAG = {r.tag: r for r in ALL}


def version_of(r: Rung) -> str:
    return r.reuse or f"h3/{r.tag}"


def run_dir(r: Rung, view: str, mode: str):
    return RUNS / version_of(r) / (view if mode == "cv" else f"{view}_{mode}")


def config_of(base: ExperimentConfig, r: Rung, view: str, mode: str) -> ExperimentConfig:
    return replace(base, version=version_of(r), views=[view], model=copy.deepcopy(base.model), evaluation=mode,
                   description=f"H3 rung {r.step} ({r.tag}): v0.1 estimator, {r.changes}", **r.changes)


def run_rungs(base: ExperimentConfig, force: bool, only: list[str] | None) -> None:
    for r in ALL:
        if r.reuse or (only and r.tag not in only):
            continue
        for view in r.views:
            for mode in MODES:
                if force or not (run_dir(r, view, mode) / "metrics.json").exists():
                    log.info("running H3 %s %s %s", r.tag, view, mode)
                    run_experiment(config_of(base, r, view, mode), [view])


def _metrics(r: Rung, view: str, mode: str) -> dict | None:
    p = run_dir(r, view, mode) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _oof(r: Rung, view: str, mode: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(r, view, mode) / "oof.parquet")


def _row(m: dict) -> dict:
    e, sub = m["estimate"], m["subgroups"]["fpr0.01"]
    row = {"prevalence": m["data"]["prevalence"], "fpr@fpr0.01": e["fpr@fpr0.01"]["point"],
           "text": m["data"]["text"], "include_thought": m["data"]["include_thought"]}
    for k in METRICS:
        row |= {k: e[k]["point"], f"{k}_low": e[k]["low"], f"{k}_high": e[k]["high"]}
    for cat, v in sub["recall_by_attack_type"].items():
        row[f"recall_{cat.replace('attacked_', '').replace('_attack', '')}"] = v["recall"]
    for cat, v in sub["alert_rate_by_class_of_negatives"].items():
        row[f"alert_{cat.replace('attacked_', '').replace('_attack', '')}"] = v["alert_rate"]
    return row


def runs_table() -> pd.DataFrame:
    rows = []
    for r in ALL:
        for view in r.views:
            for mode in MODES:
                if (m := _metrics(r, view, mode)) is not None:
                    rows.append({"rung": r.tag, "step": r.step, "label": r.label, "view": view, "mode": mode,
                                 **_row(m)})
    return pd.DataFrame(rows)


def _pairs() -> list[tuple[Rung, Rung, str]]:
    """(a, b, kind): successive rungs (a = the rung after b, so a - b is the effect of removing one information),
    then every rung against the reference 5 (and the same for the action-only ladder)."""
    out = []
    for ladder in (RUNGS, ACTION_RUNGS):
        out += [(ladder[i + 1], ladder[i], "successive") for i in range(len(ladder) - 1)]
        ref = ladder[-1]
        out += [(ref, r, "to_reference") for r in ladder[:-2]]
    out.append((BY_TAG["action_masked"], BY_TAG["masked"], "action_vs_full"))
    return out


def steps_table(n_resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    for a, b, kind in _pairs():
        for view in sorted(set(a.views) & set(b.views), key=VIEWS.index):
            for mode in MODES:
                if _metrics(a, view, mode) is None or _metrics(b, view, mode) is None:
                    continue
                d = paired_difference(_oof(a, view, mode), _oof(b, view, mode), [0.01, 0.002], n_resamples, seed)
                row = {"kind": kind, "view": view, "mode": mode, "from": b.tag, "to": a.tag,
                       "removed": f"{b.step} -> {a.step}"}
                for k in METRICS:
                    row |= {f"delta_{k}": d[k]["point"], f"delta_{k}_low": d[k]["low"],
                            f"delta_{k}_high": d[k]["high"]}
                row["excludes_zero_pr_auc"] = bool(d["pr_auc"]["low"] > 0 or d["pr_auc"]["high"] < 0)
                rows.append(row)
    return pd.DataFrame(rows)


def recall_same_fpr(fpr: float = 0.01) -> pd.DataFrame:
    """Recall at `fpr` with a threshold set on the negatives of each evaluated split, by attack type (diagnostic)."""
    from agentdrift.splits.grouped import load_folds
    src = load_folds().set_index("traj_id")["source_category"]
    rows = []
    for r in ALL:
        for view in r.views:
            for mode in MODES:
                if _metrics(r, view, mode) is None:
                    continue
                o = _oof(r, view, mode)
                o["cat"] = o["trajectory_id"].map(src)
                acc: dict[str, list[float]] = {k: [] for k in ("all", "full", "partial", "delayed", "failed")}
                for _, d in o.groupby("split"):
                    neg = np.sort(d.loc[d["label"] == 0, "score"].to_numpy())[::-1]
                    thr = neg[min(int(fpr * len(neg)), len(neg) - 1)]
                    pos = d[d["label"] == 1]
                    acc["all"].append(float((pos["score"] > thr).mean()))
                    for c in ("full", "partial", "delayed", "failed"):
                        q = pos[pos["cat"] == ("failed_attack" if c == "failed" else f"attacked_{c}")]
                        if len(q):
                            acc[c].append(float((q["score"] > thr).mean()))
                rows.append({"rung": r.tag, "view": view, "mode": mode,
                             **{f"recall_same_fpr_{k}": float(np.mean(v)) if v else None for k, v in acc.items()}})
    return pd.DataFrame(rows)


def probe_table() -> pd.DataFrame:
    """Attack-type probe on the injection text of raw, anonymised and masked tables (train -> val)."""
    from agentdrift.data import audit as A
    from agentdrift.data.load import read_steps

    raw = read_steps(include_test=True)
    raw["origin_split"] = pd.Categorical(raw["origin_split"], ["train", "val", "test"], ordered=True)
    da = A.load_inputs_anon()
    dm = A.load_inputs_masked(da)
    rows = {"raw": A.attack_type_probe(raw), "anon": A.attack_type_probe(da["steps"])}
    if dm is not None:
        rows["masked"] = A.attack_type_probe(dm["steps"])
    return pd.DataFrame(rows).T.rename_axis("text").reset_index()


def shortcuts_table() -> pd.DataFrame:
    rows = []
    for version in ("v0_0_position", "v0_0_style"):
        for view in VIEWS:
            p = RUNS / version / view / "metrics.json"
            if p.exists():
                m = json.loads(p.read_text())
                rows.append({"version": version, "view": view, "mode": "cv", **_row(m)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H3: ablation ladder raw + thought -> raw -> anonymised -> masked.")
    parser.add_argument("--only", nargs="+", choices=[r.tag for r in ALL if not r.reuse], help="rungs to run")
    parser.add_argument("--force", action="store_true", help="rerun rungs that already have results")
    parser.add_argument("--tables-only", action="store_true", help="no run: tables and figure from disk")
    parser.add_argument("--no-probe", action="store_true", help="skip the attack-type probe (~1 min)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base = ExperimentConfig.from_yaml(BASE)
    if not args.tables_only:
        run_rungs(base, args.force, args.only)
    OUT.mkdir(parents=True, exist_ok=True)
    n, seed = base.bootstrap["n_resamples"], base.bootstrap["seed"]
    tables = {"runs": runs_table(), "steps": steps_table(n, seed), "recall_same_fpr": recall_same_fpr(),
              "shortcuts": shortcuts_table()}
    if not args.no_probe:
        tables["probe"] = probe_table()
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
    from agentdrift.viz import plots as P
    P.setup()
    runs = tables["runs"]
    if len(runs):
        P.save(P.plot_h3(runs[runs["rung"].isin([r.tag for r in RUNGS])], [r.label for r in RUNGS],
                         tables["shortcuts"]), "b_h3_ladder")
    fmt = {"float_format": lambda v: f"{v:.3f}", "index": False}
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        cols = ["rung", "view", "mode", "pr_auc", "roc_auc", "recall@fpr0.01", "fpr@fpr0.01", "recall_full",
                "recall_partial", "recall_delayed", "recall_failed", "alert_failed", "alert_hard_negative"]
        print("=== runs\n" + runs[[c for c in cols if c in runs]].to_string(**fmt))
        s = tables["steps"]
        cols = ["kind", "view", "mode", "from", "to", "delta_pr_auc", "delta_pr_auc_low", "delta_pr_auc_high",
                "delta_recall@fpr0.01", "delta_recall@fpr0.002"]
        print("\n=== paired differences (to - from)\n" + (s[cols].to_string(**fmt) if len(s) else "(none)"))
        print("\n=== recall at 1 % FPR, threshold on each evaluated split (diagnostic)\n"
              + tables["recall_same_fpr"].to_string(**fmt))
        if "probe" in tables:
            print("\n=== attack-type probe on the injection text\n" + tables["probe"].to_string(**fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
