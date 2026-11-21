"""Build the scanner and gate views (B.1) from the interim tables and summarise them against the EDA.

    python -m agentdrift.pipelines.build_views                 # dev views (train + val)
    python -m agentdrift.pipelines.build_views --include-test  # + the fixed test set, in separate files

Writes data/processed/views/{scanner,gate}[_thought][_test].parquet (not committed) and the summary
results/runs/views/summary.json (versioned, like the EDA metrics). Both `thought` variants are built
(JOURNAL 19/11: without thought = reference). Requires `python -m agentdrift.data.load` first.

The summary is descriptive and covers every split, the fixed test set included, as in the EDA
(`data/audit.py`): views fit nothing, so building them never "uses" the test set; only writing it to
disk is opt-in, and `read_view` hides it by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from agentdrift.config import ROOT, RUNS
from agentdrift.data.load import read_steps, read_trajectories
from agentdrift.features.views import VIEWS, ViewConfig, build_view, feature_columns, view_path
from agentdrift.utils.seed import set_seed

OUT = RUNS / "views"
EDA_METRICS = RUNS / "eda" / "metrics.json"
EDA_TARGET = {"scanner": "y_scan", "gate": "y_gate"}
SPLITS = ["train", "val", "test"]


def _git_hash() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    return out.stdout.strip() + ("-dirty" if dirty.strip() else "")


def prevalence(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)["label"]
    t = pd.DataFrame({"rows": g.size(), "positives": g.sum().astype(int), "prevalence": g.mean().astype(float)})
    t.loc["all"] = [len(df), int(df["label"].sum()), float(df["label"].mean())]
    return t.astype({"rows": int, "positives": int})


def history_lengths(df: pd.DataFrame) -> dict:
    chars = df["history_text"].str.len()
    q = chars.quantile([0.5, 0.9, 0.99]).round(0)
    return {
        "chars_mean": round(float(chars.mean()), 1), "chars_p50": float(q[0.5]), "chars_p90": float(q[0.9]),
        "chars_p99": float(q[0.99]), "chars_max": int(chars.max()),
        "steps_mean": round(float(df["history_n_steps"].mean()), 2), "steps_max": int(df["history_n_steps"].max()),
        "n_truncated": int(df["history_truncated"].sum()),
    }


def compare_with_eda(view: str, by_split: pd.DataFrame, n_traj: pd.Series) -> pd.DataFrame:
    """Rows and positives of the view vs the EDA step table (results/runs/eda/metrics.json).

    Expected: gate rows = EDA steps and positives = EDA y_gate; scanner rows = EDA steps - trajectories
    (no row at t = 0) and positives = EDA y_scan (y_scan(0) = 0 by definition)."""
    eda = json.loads(EDA_METRICS.read_text())["prevalence_split"]
    e = pd.DataFrame(eda["data"], index=eda["index"], columns=eda["columns"])
    target = EDA_TARGET[view]
    rows = []
    for split in [*SPLITS, "all"]:
        steps = int(e.loc[split, "steps"])
        expected_rows = steps - int(n_traj[split]) if view == "scanner" else steps
        expected_pos = int(round(e.loc[split, target] * steps))
        rows.append({
            "split": split, "eda_steps": steps, f"eda_{target}": expected_pos,
            f"eda_{target}_prevalence": round(float(e.loc[split, target]), 4),
            "expected_rows": expected_rows, "view_rows": int(by_split.loc[split, "rows"]),
            "view_positives": int(by_split.loc[split, "positives"]),
            "view_prevalence": round(float(by_split.loc[split, "prevalence"]), 4),
            "gap_rows": int(by_split.loc[split, "rows"]) - expected_rows,
            "gap_positives": int(by_split.loc[split, "positives"]) - expected_pos,
        })
    return pd.DataFrame(rows).set_index("split")


def _clip(text: str, width: int = 600) -> str:
    return text if len(text) <= width else text[: width // 2] + " […] " + text[-width // 2:]


def examples(view: str, df: pd.DataFrame, traj: pd.DataFrame) -> str:
    """One positive and one negative train row, printed readably. The positive comes from a full attack, the
    negative from a hard negative (the realistic difficulty); source_category is used to pick, never shown."""
    cat = traj.set_index("traj_id")["source_category"]
    d = df[df["origin_split"] == "train"].assign(_cat=df["trajectory_id"].map(cat))
    pos = d[(d["label"] == 1) & (d["_cat"] == "attacked_full")].sort_values(["trajectory_id", "step"]).iloc[0]
    neg = d[(d["label"] == 0) & (d["_cat"] == "hard_negative") & (d["step"] >= 2)]
    neg = neg.sort_values(["trajectory_id", "step"]).iloc[0]
    out = []
    for name, r in (("POSITIVE", pos), ("NEGATIVE", neg)):
        out.append(f"--- {view} {name}: {r['trajectory_id']} step {r['step']} label {r['label']}")
        for c in feature_columns(view):  # type: ignore[arg-type]
            out.append(f"[{c}]\n{_clip(str(r[c]))}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the scanner / gate views into data/processed/views/.")
    parser.add_argument("--include-test", action="store_true", help="also write the fixed test set views")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    seed = set_seed()  # nothing is random here; seeded for uniformity with the other runs

    steps, traj = read_steps(include_test=True), read_trajectories(include_test=True)
    n_traj = traj["origin_split"].value_counts()
    n_traj["all"] = len(traj)
    cfg = ViewConfig()
    summary: dict = {
        "run": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed, "git": _git_hash()},
        "config": asdict(cfg),
        "views": {},
    }
    view_path(VIEWS[0]).parent.mkdir(parents=True, exist_ok=True)
    report = []
    for view in VIEWS:
        for include_thought in (False, True):
            start = time.perf_counter()
            df = build_view(steps, traj, view, include_thought, cfg)
            seconds = round(time.perf_counter() - start, 1)
            is_test = df["origin_split"] == "test"
            df[~is_test].reset_index(drop=True).to_parquet(view_path(view, include_thought), index=False)
            if args.include_test:
                df[is_test].reset_index(drop=True).to_parquet(view_path(view, include_thought, test=True), index=False)
            name = f"{view}{'_thought' if include_thought else ''}"
            by_split = prevalence(df, "origin_split").reindex([*SPLITS, "all"])
            entry = {
                "seconds": seconds,
                "rows": len(df),
                "history": history_lengths(df),
                "edge_cases": df.attrs["edge_cases"],
                "prevalence_split": json.loads(by_split.to_json(orient="index")),
            }
            if not include_thought:  # labels and rows do not depend on the thought variant
                by_domain = prevalence(df, "domain")
                cmp = compare_with_eda(view, by_split, n_traj)
                entry["prevalence_domain"] = json.loads(by_domain.to_json(orient="index"))
                entry["eda_comparison"] = json.loads(cmp.to_json(orient="index"))
                report += [f"\n=== {view}: {len(df):,} rows ({seconds}s)", "Prevalence by split:",
                           by_split.round(4).to_string(), "Prevalence by domain:", by_domain.round(4).to_string(),
                           "Comparison with the EDA step table:", cmp.to_string(),
                           examples(view, df, traj)]
            report.append(f"{name}: history {entry['history']}; edge cases {entry['edge_cases'] or 'none'}")
            summary["views"][name] = entry

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(report))
    written = "dev" + (" + test" if args.include_test else "")
    print(f"\nViews ({written}) written to {view_path(VIEWS[0]).parent.relative_to(ROOT)}; "
          f"summary in {(OUT / 'summary.json').relative_to(ROOT)}")
    gaps = [(v, s["eda_comparison"]) for v, s in summary["views"].items() if "eda_comparison" in s]
    bad = [v for v, c in gaps if any(r["gap_rows"] or r["gap_positives"] for r in c.values())]
    if bad:
        print(f"WARNING: rows / positives differ from the EDA for {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
