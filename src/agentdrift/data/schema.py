"""Validated schema of the interim tables (trajectory, step, template) and cross-table invariants.

Column-level rules use pandera. Trajectory-level invariants (label sequences, injection -> hijack gap)
are checked in plain pandas by `trajectory_violations`, which returns every violating trajectory so
the audit can report them: nothing is ever corrected silently.

Step index `t` is 0-based (the dataset README's "step 2 to 4" is t = 1 to 3 here).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

LABELS = ("benign", "injection_point", "hijacked", "failed_injection")
DOMAINS = ("email", "banking", "web", "coding", "medical")
CATEGORIES = ("benign", "attacked", "hard_negative", "failed_attack")
SOURCE_CATEGORIES = (
    "benign", "hard_negative", "failed_attack", "attacked_full", "attacked_partial", "attacked_delayed",
)
COMPLIANCE = ("full_hijack", "partial_hijack", "delayed_execution", "resisted")
SPLITS = ("train", "val", "test")
COMPLEXITY = ("simple", "medium", "complex")

# Generator metadata: grouping, stratification and subgroup analysis only, NEVER model inputs (L12).
METADATA_COLUMNS = (
    "category", "source_category", "dataset_category", "attack_type", "compliance",
    "world_id", "world_json", "template_id", "complexity", "n_steps", "injection_step", "first_h_step",
    "y_traj", "origin_split", "label", "is_first_h", "y_scan", "y_gate",
)

# Expected injection -> first hijacked step gap (CONTEXT 6.2; 3 delayed trajectories have gap 4).
GAP_IMMEDIATE = {"attacked_full", "attacked_partial"}
GAP_DELAYED = {"attacked_delayed"}
INJECTION_STEPS = range(1, 4)  # README: poisoned observation at step 2 to 4 (1-based)


def _in(values) -> pa.Check:
    return pa.Check.isin(list(values))


TRAJECTORIES = pa.DataFrameSchema(
    {
        "traj_id": pa.Column(str, unique=True),
        "origin_split": pa.Column(str, _in(SPLITS)),
        "domain": pa.Column(str, _in(DOMAINS)),
        "category": pa.Column(str, _in(CATEGORIES)),
        "source_category": pa.Column(str, _in(SOURCE_CATEGORIES)),
        "dataset_category": pa.Column(str, _in(CATEGORIES)),
        "attack_type": pa.Column(str, nullable=True),
        "compliance": pa.Column(str, _in(COMPLIANCE), nullable=True),
        "task": pa.Column(str, pa.Check.str_length(min_value=1)),
        "template_id": pa.Column(str, pa.Check.str_matches(r"^[a-z]+_\d{3}$")),
        "complexity": pa.Column(str, _in(COMPLEXITY)),
        "world_id": pa.Column(str, pa.Check.str_length(12, 12)),
        "world_json": pa.Column(str),
        "n_steps": pa.Column(int, pa.Check.in_range(1, 30)),
        "injection_step": pa.Column("Int64", nullable=True),
        "first_h_step": pa.Column("Int64", nullable=True),
        "y_traj": pa.Column("int8", _in((0, 1))),
    },
    checks=[
        pa.Check(lambda d: (d["category"] == d["dataset_category"]).all(), error="category != dataset_category"),
        pa.Check(lambda d: d["template_id"].str.split("_").str[0].eq(d["domain"]).all(), error="template domain"),
        pa.Check(lambda d: (d["attack_type"].isna() == d["compliance"].isna()).all(), error="attack_type/compliance"),
        pa.Check(lambda d: (d["y_traj"] == (d["category"] == "attacked")).all(), error="y_traj"),
    ],
    strict=True,
    ordered=True,
)

STEPS = pa.DataFrameSchema(
    {
        "traj_id": pa.Column(str),
        "t": pa.Column(int, pa.Check.ge(0)),
        "origin_split": pa.Column(str, _in(SPLITS)),
        "domain": pa.Column(str, _in(DOMAINS)),
        "template_id": pa.Column(str),
        "world_id": pa.Column(str),
        "source_category": pa.Column(str, _in(SOURCE_CATEGORIES)),
        "thought": pa.Column(str),
        "tool": pa.Column(str, pa.Check.str_length(min_value=1)),
        "args_json": pa.Column(str, pa.Check.str_startswith("{")),
        "obs": pa.Column(str),
        "label": pa.Column(str, _in(LABELS)),
        "y_scan": pa.Column("int8", _in((0, 1))),
        "y_gate": pa.Column("int8", _in((0, 1))),
        "is_first_h": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda d: ~d.duplicated(["traj_id", "t"]).any(), error="duplicate (traj_id, t)"),
        pa.Check(lambda d: (d["y_gate"] == (d["label"] == "hijacked")).all(), error="y_gate != label"),
        pa.Check(lambda d: (d["is_first_h"] <= d["y_gate"].astype(bool)).all(), error="is_first_h outside H"),
    ],
    strict=True,
    ordered=True,
)

TEMPLATES = pa.DataFrameSchema(
    {
        "template_id": pa.Column(str, unique=True),
        "domain": pa.Column(str, _in(DOMAINS)),
        "instruction": pa.Column(str, unique=True),
        "complexity": pa.Column(str, _in(COMPLEXITY)),
        "expected_tools": pa.Column(object),
    },
    strict=True,
)


def trajectory_violations(traj: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """One row per (traj_id, rule) that is violated. Empty frame = all invariants hold."""
    s = steps.sort_values(["traj_id", "t"])
    g = s.groupby("traj_id", sort=False)
    per = pd.DataFrame(
        {
            "n_rows": g.size(),
            "t_ok": g["t"].agg(lambda x: list(x) == list(range(len(x)))),
            "n_I": g["label"].agg(lambda x: (x == "injection_point").sum()),
            "n_F": g["label"].agg(lambda x: (x == "failed_injection").sum()),
            "n_H": g["label"].agg(lambda x: (x == "hijacked").sum()),
            "n_first_h": g["is_first_h"].sum(),
            "y_scan_ok": g.apply(_y_scan_ok, include_groups=False),
        }
    )
    d = traj.set_index("traj_id").join(per, how="outer")
    sc = d["source_category"]
    gap = d["first_h_step"] - d["injection_step"]

    rules = {
        "trajectory without steps / steps without trajectory": d["n_rows"].isna() | sc.isna(),
        "t not contiguous from 0": ~d["t_ok"].fillna(False).astype(bool),
        "n_steps != number of step rows": d["n_steps"] != d["n_rows"],
        "benign / hard_negative contains a non-benign step": sc.isin(["benign", "hard_negative"])
        & ((d["n_I"] + d["n_F"] + d["n_H"]) > 0),
        "attacked: not exactly one injection_point": (d["category"] == "attacked") & (d["n_I"] != 1),
        "attacked: no hijacked step": (d["category"] == "attacked") & (d["n_H"] == 0),
        "failed_attack: not exactly one failed_injection, or hijacked present": (d["category"] == "failed_attack")
        & ((d["n_F"] != 1) | (d["n_H"] > 0) | (d["n_I"] > 0)),
        "injection outside t = 1..3": d["injection_step"].notna()
        & ~d["injection_step"].isin(list(INJECTION_STEPS)),
        "full/partial: gap injection -> first H != 1": sc.isin(GAP_IMMEDIATE) & (gap != 1),
        "delayed: gap injection -> first H < 2": sc.isin(GAP_DELAYED) & (gap < 2),
        "is_first_h count != 1 (attacked) or != 0 (others)": d["n_first_h"] != (d["category"] == "attacked"),
        "y_scan inconsistent with previous label": ~d["y_scan_ok"].fillna(False).astype(bool),
    }
    out = [
        pd.DataFrame({"traj_id": d.index[mask.fillna(True).to_numpy()], "rule": rule})
        for rule, mask in rules.items()
        if mask.fillna(True).any()
    ]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["traj_id", "rule"])


def _y_scan_ok(g: pd.DataFrame) -> bool:
    prev = g["label"].shift(1).isin(["injection_point", "failed_injection"]).astype("int8")
    return bool((g["y_scan"].to_numpy() == prev.to_numpy()).all())


def validate(traj: pd.DataFrame, steps: pd.DataFrame, templates: pd.DataFrame) -> pd.DataFrame:
    """Validate the three tables (raises pandera SchemaErrors, all errors at once) and return the
    trajectory-level violations for reporting."""
    TRAJECTORIES.validate(traj, lazy=True)
    STEPS.validate(steps, lazy=True)
    TEMPLATES.validate(templates, lazy=True)
    return trajectory_violations(traj, steps)
