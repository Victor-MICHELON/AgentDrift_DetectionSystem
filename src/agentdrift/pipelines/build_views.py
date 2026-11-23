"""Build the scanner and gate views (B.1, KAN-16) on anonymised and masked text (KAN-60) and check them.

    python -m agentdrift.pipelines.build_views                       # dev views, text anon + masked
    python -m agentdrift.pipelines.build_views --raw                 # + raw text (H3 ablation ladder only, KAN-19)
    python -m agentdrift.pipelines.build_views --include-test        # + the fixed test set, in separate files
    python -m agentdrift.pipelines.build_views --atbench             # + ATBench views (raw text, see below)

Text variants (features/views.py): `anon` = world-anonymised tables (`read_*(anonymized=True)`, leak L9), the
default and the only model input; `masked` = anon + generator-artefact masking (`features/masking.mask_tables`,
H3 / L10), applied to the TABLES because amounts are numbered in the temporal order of a trajectory; `raw` =
original text, built only on request for the ablation ladder. Both `thought` variants are built for each (JOURNAL
19/11: without thought = reference).

Writes data/processed/views/<view>[_thought][_masked|_raw][_test].parquet (not committed), each with its variant in
the Parquet metadata, and results/runs/views/summary.json (versioned). Checks, all recorded in the summary (exit 1
if one fails):
    eda_comparison   rows and positives of every variant equal the EDA step table (results/runs/eda/metrics.json)
    invariance       keys, labels, groups and diagnostics are identical across text variants: only the text changes
    residual         trajectories whose view text still holds a world identifier (`data.anonymize.
                     residual_identifiers`, same definitions as the anonymisation itself). Must be 0 for full names,
                     world e-mails and company names on anon / masked; the raw variant is the positive control (the
                     check must find identifiers there, otherwise it measures nothing, AI_AUDIT A1).

ATBench (`--atbench`): views on RAW text, flagged in the metadata and the summary, until the team decides how to
anonymise a dataset that has no simulated world (PLAN_DONNEES, decision 5). Building views fits nothing and chooses
nothing, so it does not use ATBench for any decision (L8).

The summary is descriptive and covers every split, the fixed test set included, as in the EDA (`data/audit.py`):
views fit nothing, so building them never "uses" the test set; only writing it to disk is opt-in, and `read_view`
hides it by default.
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
from agentdrift.data.anonymize import residual_identifiers
from agentdrift.data.load import read_steps, read_trajectories
from agentdrift.features.masking import mask_tables
from agentdrift.features.views import (
    DEFAULT_TEXT, NON_FEATURE_COLUMNS, VIEWS, ViewConfig, build_view, feature_columns, view_path, write_view,
)
from agentdrift.utils.seed import set_seed

OUT = RUNS / "views"
EDA_METRICS = RUNS / "eda" / "metrics.json"
EDA_TARGET = {"scanner": "y_scan", "gate": "y_gate"}
SPLITS = ["train", "val", "test"]
# Identifiers that must be absent from anon / masked text. "domain label" is reported only: the anonymisation
# keeps 3 known substring false positives ("unitedhealthcare", a password, a class name; JOURNAL 20/11).
STRICT_IDENTIFIERS = ("full name", "world e-mail", "company name")
ATBENCH_STATUS = "raw text: anonymisation of ATBench pending team decision 5 (docs/PLAN_DONNEES.md)"


def _git_hash() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    return out.stdout.strip() + ("-dirty" if dirty.strip() else "")


def view_name(view: str, include_thought: bool, text: str) -> str:
    """Key of a view in the summary: scanner, gate_thought, scanner_masked, gate_thought_raw..."""
    return f"{view}{'_thought' if include_thought else ''}{'' if text == DEFAULT_TEXT else '_' + text}"


def load_tables(variants: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """(trajectories, steps) of each text variant, fixed test set included (written to separate files)."""
    anon = read_trajectories(include_test=True, anonymized=True), read_steps(include_test=True, anonymized=True)
    tables = {"anon": anon}
    if "masked" in variants:
        tables["masked"] = mask_tables(*anon)
    if "raw" in variants:
        tables["raw"] = read_trajectories(include_test=True), read_steps(include_test=True)
    return tables


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


def invariant_columns(df: pd.DataFrame) -> list[str]:
    """Columns that must not depend on the text variant (keys, label, groups, eval, diagnostics)."""
    return [c for c in df.columns if c in NON_FEATURE_COLUMNS]


def same_rows(df: pd.DataFrame, ref: pd.DataFrame) -> bool:
    cols = invariant_columns(ref)
    return list(df.columns) == list(ref.columns) and df[cols].reset_index(drop=True).equals(ref[cols].reset_index(drop=True))


def residual_in_view(df: pd.DataFrame, view: str, include_thought: bool, traj_raw: pd.DataFrame) -> dict[str, int]:
    """Trajectories whose view text (every feature column of every row) still holds a world identifier.

    Reuses `data.anonymize.residual_identifiers` (Lot A) unchanged: the whole text of a trajectory is passed as one
    observation with an empty task, so the definitions are exactly those used to validate the anonymisation."""
    cols = feature_columns(view, include_thought)  # type: ignore[arg-type]
    text = df[cols].astype(str).agg(" ".join, axis=1).groupby(df["trajectory_id"]).agg(" ".join)
    traj = traj_raw[traj_raw["traj_id"].isin(text.index)]
    steps_like = pd.DataFrame({"traj_id": text.index, "thought": "", "args_json": "", "obs": text.to_numpy()})
    res = residual_identifiers(traj, pd.DataFrame({"traj_id": traj["traj_id"], "task": ""}), steps_like)
    return {k: int(v) for k, v in res.items()} | {"trajectories_checked": int(len(traj))}


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


def build_agentdrift(variants: list[str], include_test: bool, cfg: ViewConfig, report: list[str]) -> dict:
    tables = load_tables(variants)
    traj_raw = read_trajectories(include_test=True)  # world identities, for the residual check only
    n_traj = traj_raw["origin_split"].value_counts()
    n_traj["all"] = len(traj_raw)
    entries: dict = {}
    for view in VIEWS:
        for include_thought in (False, True):
            ref = None  # anon view of the same (view, thought): reference of the invariance check
            for text in variants:
                traj, steps = tables[text]
                start = time.perf_counter()
                df = build_view(steps, traj, view, include_thought, cfg)
                seconds = round(time.perf_counter() - start, 1)
                is_test = df["origin_split"] == "test"
                write_view(df[~is_test], view, include_thought, text=text)
                if include_test:
                    write_view(df[is_test], view, include_thought, test=True, text=text)
                name = view_name(view, include_thought, text)
                by_split = prevalence(df, "origin_split").reindex([*SPLITS, "all"])
                cmp = compare_with_eda(view, by_split, n_traj)
                ref = df if text == "anon" else ref
                entry = {
                    "text": text, "include_thought": include_thought, "seconds": seconds, "rows": len(df),
                    "history": history_lengths(df), "edge_cases": df.attrs["edge_cases"],
                    "prevalence_split": json.loads(by_split.to_json(orient="index")),
                    "eda_comparison": json.loads(cmp.to_json(orient="index")),
                    "same_rows_as_anon": bool(same_rows(df, ref)),
                    "residual": residual_in_view(df, view, include_thought, traj_raw),
                }
                if text == "anon" and not include_thought:
                    by_domain = prevalence(df, "domain")
                    entry["prevalence_domain"] = json.loads(by_domain.to_json(orient="index"))
                    report += [f"\n=== {view}: {len(df):,} rows ({seconds}s)", "Prevalence by split:",
                               by_split.round(4).to_string(), "Prevalence by domain:", by_domain.round(4).to_string(),
                               "Comparison with the EDA step table:", cmp.to_string(), examples(view, df, traj)]
                report.append(f"{name}: {len(df):,} rows, {seconds}s, same rows as anon {entry['same_rows_as_anon']}, "
                              f"residual {entry['residual']}")
                entries[name] = entry
    return entries


def build_atbench(cfg: ViewConfig, report: list[str]) -> dict:
    from agentdrift.data.external import read_atbench

    traj, steps = read_atbench()
    out = {"status": ATBENCH_STATUS, "views": {}}
    for view in VIEWS:
        for include_thought in (False, True):
            df = build_view(steps, traj, view, include_thought, cfg)
            write_view(df, view, include_thought, text="raw", dataset="atbench")
            name = view_name(view, include_thought, "raw")
            out["views"][name] = {"rows": len(df), "trajectories": int(df["trajectory_id"].nunique()),
                                  "history": history_lengths(df), "edge_cases": df.attrs["edge_cases"]}
            report.append(f"atbench_{name}: {len(df):,} rows, {df['trajectory_id'].nunique()} trajectories")
    return out


def failures(summary: dict) -> list[str]:
    bad = []
    for name, e in summary["views"].items():
        if any(r["gap_rows"] or r["gap_positives"] for r in e["eda_comparison"].values()):
            bad.append(f"{name}: rows / positives differ from the EDA")
        if not e["same_rows_as_anon"]:
            bad.append(f"{name}: keys / labels / groups differ from the anon variant")
        found = {k: e["residual"][k] for k in STRICT_IDENTIFIERS}
        if e["text"] != "raw" and any(found.values()):
            bad.append(f"{name}: residual world identifiers {found}")
        if e["text"] == "raw" and not any(found.values()):
            bad.append(f"{name}: the residual check finds nothing on raw text (positive control failed)")
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the scanner / gate views into data/processed/views/.")
    parser.add_argument("--include-test", action="store_true", help="also write the fixed test set views")
    parser.add_argument("--raw", action="store_true", help="also build the raw-text views (H3 ablation only)")
    parser.add_argument("--atbench", action="store_true", help="also build the ATBench views (raw text, flagged)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    seed = set_seed()  # nothing is random here; seeded for uniformity with the other runs

    variants = ["anon", "masked"] + (["raw"] if args.raw else [])
    cfg = ViewConfig()
    start = time.perf_counter()
    report: list[str] = []
    summary: dict = {
        "run": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed, "git": _git_hash()},
        "config": asdict(cfg),
        "text_variants": variants,
        "default_text": DEFAULT_TEXT,
        "views": build_agentdrift(variants, args.include_test, cfg, report),
    }
    if args.atbench:
        summary["atbench"] = build_atbench(cfg, report)
    summary["run"]["seconds"] = round(time.perf_counter() - start, 1)
    bad = failures(summary)
    summary["checks_passed"] = not bad
    summary["failures"] = bad

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(report))
    written = "dev" + (" + test" if args.include_test else "")
    print(f"\nViews ({written}, text {variants}) written to {view_path(VIEWS[0]).parent.relative_to(ROOT)} in "
          f"{summary['run']['seconds']}s; summary in {(OUT / 'summary.json').relative_to(ROOT)}")
    for b in bad:
        print(f"FAILED: {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
