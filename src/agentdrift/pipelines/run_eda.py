"""Headless EDA: recompute every table and figure of the EDA notebooks.

    python -m agentdrift.pipelines.run_eda             # parts A, B and C
    python -m agentdrift.pipelines.run_eda --part A    # raw data (notebooks/01_eda.ipynb)
    python -m agentdrift.pipelines.run_eda --part B    # effect of the preprocessing (notebooks/02_eda_preprocessing.ipynb)
    python -m agentdrift.pipelines.run_eda --part C    # evaluation protocol + synthesis (notebooks/03_eda_protocol.ipynb)

Writes results/runs/eda/ (part A), results/runs/eda_b/ (part B) and results/runs/eda_c/ (part C): metrics.json +
the tables as CSV; figures to results/figures/ (PNG) and report/latex/figures/ (PDF). Requires
`python -m agentdrift.data.load` (and `python -m agentdrift.data.anonymize` for part B,
`python -m agentdrift.splits.grouped` for part C) first. The synthesis of part C reads the metrics.json of
parts A and B: run them first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agentdrift.config import ROOT, RUNS, load_config  # noqa: E402
from agentdrift.data import annotations as N  # noqa: E402
from agentdrift.data import audit as A  # noqa: E402
from agentdrift.utils.seed import set_seed  # noqa: E402
from agentdrift.viz import plots as P  # noqa: E402

OUT = RUNS / "eda"
OUT_B = RUNS / "eda_b"
OUT_C = RUNS / "eda_c"
INTERVAL_METRICS = ["roc_auc", "pr_auc", "recall@1%", "recall@0.2%"]


def compute(d: dict) -> dict:
    """All tables of part A (one place, shared by the notebook and this runner)."""
    traj, steps = d["trajectories"], d["steps"]
    prov = A.provenance_by_group(d)
    shortcuts = A.shortcut_baselines(d)
    vis = A.injection_visibility(steps)
    conf = A.confession_in_thought(steps)
    oox = A.out_of_expected_tools(d)
    pts, sil = A.text_projection(d)
    atb = A.load_atbench()
    return {
        "overview": A.overview(d),
        "category_by_domain": A.category_by_domain(traj),
        "split_composition": A.split_composition(traj),
        "examples": A.example_trajectories(d),
        "missing": A.missing_values(steps),
        "duplicates": A.duplicates(d),
        "addresses_with_space": A.addresses_with_space(traj) | {"rows": A.addresses_with_space_all_rows(traj)},
        "lengths": A.length_outliers(steps),
        "test_dev_similarity": A.test_dev_similarity(d),
        "prevalence_split": A.prevalence(d),
        "prevalence_domain": A.prevalence(d, "domain"),
        "hijacked_per_attack": A.hijacked_per_attack(traj, steps),
        "length_by_category": A.length_by_category(traj),
        "hijack_rate_by_position": A.hijack_rate_by_position(steps),
        "injection_position": A.injection_position(traj),
        "gap": A.injection_to_hijack_gap(traj),
        "after_first_h": A.after_first_hijack(steps, traj),
        "tools_summary": A.tools_summary(steps),
        "tool_table": A.tool_label_table(steps, top=25),
        "out_of_expected": oox,
        "text_lengths": A.text_lengths(steps),
        "prefixes": A.injection_prefixes(steps),
        "visibility": vis,
        "confession": conf,
        "provenance": prov,
        "world_purity": A.world_purity(traj),
        "world_purity_by_domain": A.world_purity_by_domain(traj),
        "associations": A.association_table(d),
        "shortcuts": shortcuts,
        "projection": pts,
        "silhouettes": sil,
        "atbench": atb,
        "external": A.compare_external(d, atb),
        "atbench_scope": A.atbench_scope(atb),
        "reproduction": A.reproduction_table(d, prov, shortcuts, vis, conf, oox),
    }


def figures(r: dict, d: dict) -> dict:
    conf = r["confession"].pivot(index="source_category", columns="label", values="share")
    conf = conf.reindex(columns=[c for c in P.LABEL_ORDER if c in conf.columns])
    oox = r["out_of_expected"]["share"].reindex(["B", "I", "F", "H full", "H partial", "H delayed"])
    return {
        "eda_a1_category_by_domain": P.plot_category_by_domain(r["category_by_domain"]),
        "eda_a1_examples": P.plot_example_trajectories(r["examples"]),
        "eda_a3_prevalence": P.plot_prevalence(r["prevalence_split"], r["prevalence_domain"]),
        "eda_a4_length": P.plot_length_by_category(r["length_by_category"]),
        "eda_a4_hijack_by_position": P.plot_hijack_rate_by_position(r["hijack_rate_by_position"]),
        "eda_a4_injection_gap": P.plot_injection_and_gap(r["injection_position"], r["gap"]),
        "eda_a5_tools": P.plot_tools(r["tool_table"]),
        "eda_a5_out_of_expected": P.plot_share_bars(
            oox, "Proposed tool outside the tools the task needs", "share of steps"),
        "eda_a6_prefixes": P.plot_table_heatmap(
            r["prefixes"], "Injection openers: share of poisoned responses containing each one", width=9.5),
        "eda_a6_visibility": P.plot_table_heatmap(
            r["visibility"].drop(columns="n"), "How readable is the injection in the poisoned response?", width=8.5),
        "eda_a6_confession": P.plot_table_heatmap(
            conf, "Confession words in the agent's reasoning (share of steps)", width=7, fmt="{:.0%}"),
        "eda_a7_provenance": P.plot_table_heatmap(
            r["provenance"].drop(columns="n"), "Where do the arguments of the proposed action come from? (share of steps)",
            width=9.5),
        "eda_a8_world_purity": P.plot_world_purity(r["world_purity"]),
        "eda_a8_shortcuts": P.plot_shortcuts(r["shortcuts"]),
        "eda_a8_associations": P.plot_associations(r["associations"]),
        "eda_a9_projection": P.plot_projection(r["projection"]),
        "eda_a10_external": P.plot_external(d["trajectories"], r["atbench"]),
    }


def compute_b(d: dict, da: dict) -> dict:
    """All tables of part B: what the anonymisation (and, once it exists, the H3 masking) changes."""
    tool_risk = N.read_tool_risk()
    sheet = N.read_injection_visible() if N.INJECTION_VISIBLE_PATH.exists() else None
    full_sheet = N.read_injection_visible(annotated_only=False) if sheet is not None else None
    dm = A.load_inputs_masked(da)
    return {
        "masking_available": dm is not None,
        "text_before_after": A.text_before_after(d, da, d_masked=dm),
        "residual": A.residual_before_after(d, da),
        "world_probe": A.world_probe_before_after(d, da, dm),
        "provenance": A.provenance_before_after(d, da, dm),
        "masking_rates": None if dm is None else A.masking_rates(da),
        "injection_signals": None if dm is None else A.injection_signals(da, dm),
        "attack_type_probe": None if dm is None else A.attack_type_probe_before_after(da, dm),
        "tool_risk_status": {"status": tool_risk.attrs["status"], "reviewer": tool_risk.attrs["reviewer"],
                             "n_ambiguous": int(tool_risk["ambiguous"].sum())},
        "tool_risk_counts": tool_risk["risk"].value_counts().reindex(list(N.RISK_CATEGORIES)),
        "tool_risk_by_group": A.tool_risk_by_group(d["steps"], tool_risk),
        "annotation_progress": None if full_sheet is None else N.annotation_progress(full_sheet, d["steps"]),
        "visible_vs_heuristic": A.visible_vs_heuristic(sheet, d["steps"]),
    }


def figures_b(rb: dict) -> dict:
    figs = {
        "eda_b1_residual_identifiers": P.plot_before_after_bars(
            rb["residual"][["raw", "anonymised"]], "World identifiers left in the text",
            "share of trajectories containing the identifier"),
        "eda_b2_world_probe": P.plot_world_probe(rb["world_probe"]),
        "eda_b3_provenance": P.plot_provenance_before_after(rb["provenance"]),
    }
    if rb["injection_signals"] is not None:
        figs["eda_b6_injection_signals"] = P.plot_injection_signals(rb["injection_signals"])
        figs["eda_b7_attack_type_probe"] = P.plot_before_after_bars(
            rb["attack_type_probe"][["accuracy", "balanced_accuracy"]].T.rename(
                index={"balanced_accuracy": "balanced accuracy"}),
            "Can the text of an injection tell its attack type?", "validation score (fitted on train)",
            reference=("chance, 4 balanced types", float(rb["attack_type_probe"]["chance_balanced"].iloc[0])))
    if rb["tool_risk_by_group"] is not None:
        figs["eda_b4_tool_risk"] = P.plot_table_heatmap(
            rb["tool_risk_by_group"].drop(columns="n"), "Tool risk category of the proposed action, by step group")
    if rb["visible_vs_heuristic"] is not None:
        figs["eda_b5_injection_visible"] = P.plot_table_heatmap(
            rb["visible_vs_heuristic"][["visible (human)", "visible (regex union)", "agreement"]],
            "Is the injection visible? Human annotation vs regex union")
    return figs


def compute_c(dp: dict, seed: int = A.SEED) -> dict:
    """All tables of part C (dp = A.load_inputs_protocol()). The synthesis reads the saved outputs of A and B."""
    boot = load_config("split")["bootstrap"]
    grouping = A.grouping_checks(dp)
    rc = {
        "folds": A.fold_table(dp),
        "balance": A.fold_balance(dp),
        "fold_prevalence": A.fold_prevalence(dp),
        "grouping": grouping,
        "lodo": A.lodo_table(dp),
        "resolution": A.resolution_table(dp),
        "intervals": A.compute_protocol_intervals(dp, grouping["test_templates"], boot["n_resamples"], seed),
        "bootstrap": boot,
    }
    missing = [p for p in (OUT / "metrics.json", OUT_B / "metrics.json") if not p.exists()]
    rc["synthesis"] = None if missing else A.synthesis_table(
        json.loads((OUT / "metrics.json").read_text()), json.loads((OUT_B / "metrics.json").read_text()), rc)
    return rc


def figures_c(rc: dict) -> dict:
    return {
        "eda_c1_fold_prevalence": P.plot_fold_prevalence(rc["fold_prevalence"]),
        "eda_c2_resolution": P.plot_resolution(rc["resolution"], "0.2%", A.MIN_FALSE_POSITIVES),
        "eda_c3_interval_widths": P.plot_interval_widths(
            {t: v["summary"] for t, v in rc["intervals"].items()}, INTERVAL_METRICS),
    }


def _git_hash() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    return out.stdout.strip() + ("-dirty" if dirty.strip() else "")


def _jsonable(v):
    if hasattr(v, "to_dict"):
        return json.loads(v.to_json(orient="split", default_handler=str))
    return v


def run_a(d: dict, seed: int) -> None:
    r = compute(d)
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["overview", "duplicates", "addresses_with_space", "tools_summary", "silhouettes"]
    tables = ["prevalence_split", "prevalence_domain", "gap", "injection_position", "out_of_expected", "visibility",
              "provenance", "world_purity_by_domain", "associations", "shortcuts", "external", "reproduction"]
    metrics = {
        "run": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed, "git": _git_hash()},
        **{k: r[k] for k in keep},
        "test_dev_max_similarity": {"max": float(r["test_dev_similarity"].max()),
                                    "n_above_0.9": int((r["test_dev_similarity"] >= 0.9).sum())},
        **{k: _jsonable(r[k]) for k in tables},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    for k in tables:
        r[k].to_csv(OUT / f"{k}.csv")
    for name, fig in figures(r, d).items():
        P.save(fig, name)
        plt.close(fig)
    print(f"EDA part A written to {OUT.relative_to(ROOT)} and {P.FIGURES.relative_to(ROOT)}")
    print(r["reproduction"].to_string(index=False))


def run_b(d: dict, seed: int) -> None:
    rb = compute_b(d, A.load_inputs_anon())
    OUT_B.mkdir(parents=True, exist_ok=True)
    tables = ["residual", "world_probe", "provenance", "masking_rates", "injection_signals", "attack_type_probe",
              "tool_risk_by_group", "annotation_progress", "visible_vs_heuristic"]
    metrics = {
        "run": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed, "git": _git_hash()},
        "masking_available": rb["masking_available"],
        "tool_risk_status": rb["tool_risk_status"],
        **{k: _jsonable(rb[k]) for k in tables + ["tool_risk_counts"]},
    }
    (OUT_B / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    for k in tables:
        if rb[k] is not None:
            rb[k].to_csv(OUT_B / f"{k}.csv")
    rb["text_before_after"].to_csv(OUT_B / "text_before_after.csv")
    for name, fig in figures_b(rb).items():
        P.save(fig, name)
        plt.close(fig)
    print(f"EDA part B written to {OUT_B.relative_to(ROOT)}")
    print(rb["world_probe"].to_string())
    pending = [k for k in ("tool_risk_by_group", "visible_vs_heuristic") if rb[k] is None]
    if not rb["masking_available"]:
        pending.append("masked text (features/masking.py, KAN-19)")
    if pending:
        print("Pending:", ", ".join(pending))


def run_c(seed: int) -> None:
    rc = compute_c(A.load_inputs_protocol(), seed)
    OUT_C.mkdir(parents=True, exist_ok=True)
    tables = {"folds": rc["folds"], "balance": rc["balance"], "lodo": rc["lodo"], "resolution": rc["resolution"]}
    for t, v in rc["intervals"].items():
        tables[f"intervals_{t}"] = v["summary"]
        tables[f"intervals_by_fold_{t}"] = v["widths"]
        tables[f"threshold_transfer_{t}"] = v["transfer"]
    if rc["synthesis"] is not None:
        tables["synthesis"] = rc["synthesis"]
    metrics = {
        "run": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed, "git": _git_hash()},
        "grouping": rc["grouping"], "bootstrap": rc["bootstrap"],
        **{k: _jsonable(v) for k, v in tables.items() if not k.startswith(("intervals_by_fold", "threshold"))},
    }
    (OUT_C / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    for k, v in tables.items():
        v.to_csv(OUT_C / f"{k}.csv")
    for name, fig in figures_c(rc).items():
        P.save(fig, name)
        plt.close(fig)
    print(f"EDA part C written to {OUT_C.relative_to(ROOT)}")
    print(rc["intervals"]["y_gate"]["summary"].round(3).to_string())
    if rc["synthesis"] is None:
        print("Synthesis skipped: run parts A and B first.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute the EDA tables and figures.")
    parser.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    args = parser.parse_args(argv)
    seed = set_seed()
    P.setup()
    if args.part in ("A", "B", "all"):
        d = A.load_inputs()
        if args.part in ("A", "all"):
            run_a(d, seed)
        if args.part in ("B", "all"):
            run_b(d, seed)
    if args.part in ("C", "all"):
        run_c(seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
