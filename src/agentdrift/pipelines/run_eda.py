"""Headless EDA (part A): recompute every table and figure of notebooks/01_eda.ipynb.

    python -m agentdrift.pipelines.run_eda

Writes results/runs/eda/metrics.json (+ the tables as CSV) and the figures to results/figures/
(PNG) and report/latex/figures/ (PDF). Requires `python -m agentdrift.data.load` first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agentdrift.config import ROOT, RUNS  # noqa: E402
from agentdrift.data import audit as A  # noqa: E402
from agentdrift.utils.seed import set_seed  # noqa: E402
from agentdrift.viz import plots as P  # noqa: E402

OUT = RUNS / "eda"


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
            oox, "Proposed tool outside the template's expected tools (oracle)", "share of steps"),
        "eda_a6_prefixes": P.plot_table_heatmap(
            r["prefixes"], "Injection openers: share of observations containing each prefix", width=9.5),
        "eda_a6_visibility": P.plot_table_heatmap(
            r["visibility"].drop(columns="n"), "How readable is the injection in the poisoned observation?", width=8.5),
        "eda_a6_confession": P.plot_table_heatmap(
            conf, "Confession vocabulary in the generated thought (share of steps)", width=7, fmt="{:.0%}"),
        "eda_a7_provenance": P.plot_table_heatmap(
            r["provenance"].drop(columns="n"), "Argument provenance of the proposed action a_t (share of steps)",
            width=9.5),
        "eda_a8_world_purity": P.plot_world_purity(r["world_purity"]),
        "eda_a8_shortcuts": P.plot_shortcuts(r["shortcuts"]),
        "eda_a8_associations": P.plot_associations(r["associations"]),
        "eda_a9_projection": P.plot_projection(r["projection"]),
        "eda_a10_external": P.plot_external(d["trajectories"], r["atbench"]),
    }


def _git_hash() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    return out.stdout.strip() + ("-dirty" if dirty.strip() else "")


def _jsonable(v):
    if hasattr(v, "to_dict"):
        return json.loads(v.to_json(orient="split", default_handler=str))
    return v


def main() -> int:
    seed = set_seed()
    P.setup()
    d = A.load_inputs()
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
    print(f"EDA written to {OUT.relative_to(ROOT)} and {P.FIGURES.relative_to(ROOT)}")
    print(r["reproduction"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
