"""Checks of the non-text blocks (B.3, KAN-18) on the development rows, before any model uses them.

    python -m agentdrift.pipelines.run_blocks_check

Writes results/runs/blocks_check/:
    provenance_vs_eda.csv   gate: rate of three provenance indicators by step group (B, I, F, H full / partial /
                            delayed), next to the EDA part B values on the anonymised text (results/runs/eda_b/
                            provenance.csv). Same definitions -> the rates must match; a gap is explained, not tuned
    position.csv            Spearman correlation of every feature with the step index t, and its rate on positives /
                            negatives: what a block can tell about t (the position shortcut, JOURNAL 19/11)
    summary.json            dimensions, seconds, share of non-zero values, tool_risk.yaml status, unknown tools
Nothing here is fitted; the fixed test set is not read (`read_view` hides it).
"""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from agentdrift.config import RUNS
from agentdrift.data.load import read_steps
from agentdrift.features.provenance import InjectionCuesBlock, ProvenanceBlock
from agentdrift.features.structural import CURRENT_TOOL, StructureBlock, _risk_table, tool_risk_status
from agentdrift.features.views import read_view
from agentdrift.splits.grouped import load_folds

log = logging.getLogger(__name__)
OUT = RUNS / "blocks_check"
EDA_PROVENANCE = RUNS / "eda_b" / "provenance.csv"
# block indicator -> EDA A7 / B3 indicator (same definition on the anonymised text)
EDA_MATCH = {"provenance__any_from_results_only": "arg from previous obs",
             "provenance__unknown_email": "unknown address",
             "provenance__ext_email_in_args": "external domain"}
LABEL_GROUP = {"benign": "B", "injection_point": "I", "failed_injection": "F"}


def step_group(view_df: pd.DataFrame) -> pd.Series:
    """B / I / F / H full / H partial / H delayed of each gate row, from the step label (analysis only)."""
    steps = read_steps(anonymized=True)[["traj_id", "t", "label"]].rename(columns={"traj_id": "trajectory_id",
                                                                                   "t": "step"})
    lab = view_df[["trajectory_id", "step"]].merge(steps, how="left", on=["trajectory_id", "step"])["label"]
    src = view_df["trajectory_id"].map(load_folds().set_index("traj_id")["source_category"])
    hij = "H " + src.str.replace("attacked_", "", regex=False)
    return pd.Series(np.where(lab.to_numpy() == "hijacked", hij.to_numpy(), lab.map(LABEL_GROUP).to_numpy()),
                     index=view_df.index)


def provenance_vs_eda(gate: pd.DataFrame, prov: pd.DataFrame) -> pd.DataFrame:
    g = step_group(gate)
    ours = prov[list(EDA_MATCH)].groupby(g.to_numpy()).mean().rename(columns=EDA_MATCH)
    eda = pd.read_csv(EDA_PROVENANCE, header=[0, 1], index_col=0)
    eda = eda.xs("anonymised", axis=1, level=1)[list(EDA_MATCH.values())]
    order = ["B", "I", "F", "H full", "H partial", "H delayed"]
    t = pd.concat({"blocks": ours, "eda": eda}, axis=1).reindex(order)
    for ind in EDA_MATCH.values():
        t[("gap", ind)] = t[("blocks", ind)] - t[("eda", ind)]
    t[("n", "rows")] = g.value_counts().reindex(order)
    return t.round(4)


def position_table(view: str, frame: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    y, t = df["label"].astype(int).to_numpy(), df["step"].to_numpy()
    rows = []
    for c in frame.columns:
        x = frame[c].to_numpy()
        if np.ptp(x) == 0:
            continue
        rows.append({"view": view, "feature": c, "spearman_t": spearmanr(x, t).statistic,
                     "mean_pos": x[y == 1].mean(), "mean_neg": x[y == 0].mean(), "nonzero": float((x != 0).mean())})
    out = pd.DataFrame(rows)
    out["abs_spearman_t"] = out["spearman_t"].abs()
    return out.sort_values("abs_spearman_t", ascending=False)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tool_risk_status": tool_risk_status(), "blocks": {}}
    positions, frames = [], {}
    for view in ("scanner", "gate"):
        df = read_view(view)
        blocks = [StructureBlock(view), InjectionCuesBlock(view)] + ([ProvenanceBlock("gate")] if view == "gate" else [])
        parts = []
        for b in blocks:
            start = time.perf_counter()
            f = b.frame(df)
            name = type(b).__name__
            summary["blocks"][f"{view}/{name}"] = {"columns": f.shape[1], "seconds": round(time.perf_counter() - start, 1),
                                                   "nonzero_share": round(float((f.to_numpy() != 0).mean()), 4)}
            parts.append(f)
        frame = pd.concat(parts, axis=1)
        frames[view] = frame
        positions.append(position_table(view, frame, df))
        known = set(_risk_table()[0])
        summary[f"{view}_unknown_tools"] = sorted(set(df[CURRENT_TOOL[view]].astype(str)) - known)
    pos = pd.concat(positions, ignore_index=True)
    pos.to_csv(OUT / "position.csv", index=False)
    comp = provenance_vs_eda(read_view("gate"), frames["gate"])
    comp.to_csv(OUT / "provenance_vs_eda.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("=== provenance (gate) vs EDA, anonymised text\n", comp.to_string())
        print("\n=== features most tied to the step index t\n",
              pos.head(15)[["view", "feature", "spearman_t", "mean_pos", "mean_neg", "nonzero"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
