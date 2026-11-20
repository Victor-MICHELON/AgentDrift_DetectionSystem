"""Parsing, targets and schema invariants.

Toy-record tests run everywhere; the corpus tests are skipped if data/interim/ has not been built
(`python -m agentdrift.data.load`).
"""

import pandas as pd
import pandera.errors
import pytest

from agentdrift.config import INTERIM
from agentdrift.data.load import build_tables, read_steps, read_trajectories, world_id
from agentdrift.data.schema import STEPS, TRAJECTORIES, trajectory_violations

TEMPLATES = pd.DataFrame(
    {
        "template_id": ["email_000"],
        "domain": ["email"],
        "instruction": ["Send the report to Bob"],
        "complexity": ["simple"],
        "expected_tools": [["send_email"]],
    }
)
WORLD = {"user": "Ann", "email": "ann@x.com", "company": "X", "contacts": [], "date": "2026-01-01"}


def _record(rid, source_category, labels, category=None):
    category = category or ("attacked" if source_category.startswith("attacked") else source_category)
    rec = {
        "id": rid, "agent": "email", "category": category, "dataset_category": category,
        "source_category": source_category, "task": "Send the report to Bob", "world": WORLD,
        "split": "train",
        "steps": [
            {"thought": f"th{t}", "tool": "send_email", "args": {"to": "b", "a": t}, "obs": f"o{t}", "label": lab}
            for t, lab in enumerate(labels)
        ],
    }
    if category in ("attacked", "failed_attack"):
        rec["attack_type"], rec["compliance"] = "data_stealing", {
            "attacked_full": "full_hijack", "attacked_partial": "partial_hijack",
            "attacked_delayed": "delayed_execution", "failed_attack": "resisted",
        }[source_category]
    return rec


B, I, H, F = "benign", "injection_point", "hijacked", "failed_injection"


@pytest.fixture
def toy():
    records = [
        _record("a_full", "attacked_full", [B, I, H, H, B]),
        _record("a_delayed", "attacked_delayed", [B, I, B, B, H]),
        _record("failed", "failed_attack", [B, F, B]),
        _record("ok", "benign", [B, B]),
    ]
    return build_tables(records, TEMPLATES)


def test_targets_follow_the_definitions(toy):
    _, steps = toy
    s = steps.set_index(["traj_id", "t"])
    # y_scan(t) = label(t-1) in {I, F}
    assert s.loc["a_full", "y_scan"].tolist() == [0, 0, 1, 0, 0]
    assert s.loc["failed", "y_scan"].tolist() == [0, 0, 1]
    # y_gate(t) = label(t) == H ; first H only once
    assert s.loc["a_full", "y_gate"].tolist() == [0, 0, 1, 1, 0]
    assert s.loc["a_full", "is_first_h"].tolist() == [False, False, True, False, False]
    assert s.loc["a_delayed", "is_first_h"].tolist() == [False] * 4 + [True]


def test_trajectory_columns(toy):
    traj, _ = toy
    t = traj.set_index("traj_id")
    assert t.loc["a_delayed", ["injection_step", "first_h_step", "n_steps"]].tolist() == [1, 4, 5]
    assert t.loc["failed", "injection_step"] == 1 and pd.isna(t.loc["failed", "first_h_step"])
    assert t["y_traj"].tolist() == [1, 1, 0, 0]
    assert t["template_id"].unique().tolist() == ["email_000"]


def test_toy_tables_pass_schema(toy):
    traj, steps = toy
    TRAJECTORIES.validate(traj, lazy=True)
    STEPS.validate(steps, lazy=True)
    assert trajectory_violations(traj, steps).empty


def test_args_are_canonical(toy):
    _, steps = toy
    assert steps["args_json"].iloc[0] == '{"a":0,"to":"b"}'


def test_world_id_ignores_date_only():
    assert world_id(WORLD) == world_id({**WORLD, "date": "2030-12-31"})
    assert world_id(WORLD) != world_id({**WORLD, "user": "Bob"})


def test_unknown_task_is_rejected():
    rec = _record("x", "benign", [B])
    rec["task"] = "not a template"
    with pytest.raises(ValueError, match="not found"):
        build_tables([rec], TEMPLATES)


@pytest.mark.parametrize(
    ("source_category", "labels", "rule"),
    [
        ("attacked_full", [B, I, B, H], "full/partial: gap"),
        ("attacked_delayed", [B, I, H], "delayed: gap"),
        ("attacked_full", [B, I, I, H], "not exactly one injection_point"),
        ("attacked_full", [B, B, I], "no hijacked step"),
        ("failed_attack", [B, F, H], "failed_attack"),
        ("benign", [B, H], "benign / hard_negative"),
        ("attacked_full", [B, B, B, B, I, H], "injection outside"),
    ],
)
def test_violations_are_reported(source_category, labels, rule):
    traj, steps = build_tables([_record("bad", source_category, labels)], TEMPLATES)
    v = trajectory_violations(traj, steps)
    assert v["rule"].str.contains(rule, regex=False).any(), v


def test_schema_rejects_metadata_drift(toy):
    traj, steps = toy
    with pytest.raises(pandera.errors.SchemaErrors):
        TRAJECTORIES.validate(traj.assign(domain="finance"), lazy=True)
    with pytest.raises(pandera.errors.SchemaErrors):
        STEPS.validate(steps.assign(y_gate=steps["y_gate"].iloc[::-1].to_numpy()), lazy=True)


# ---------------------------------------------------------------- corpus (needs data/interim)

needs_interim = pytest.mark.skipif(
    not (INTERIM / "steps.parquet").exists(), reason="run `python -m agentdrift.data.load` first"
)


@needs_interim
def test_corpus_counts_match_documentation():
    traj = read_trajectories(include_test=True)
    steps = read_steps(include_test=True)
    assert len(traj) == 12_536 and len(steps) == 71_024
    assert steps["label"].value_counts().to_dict() == {
        "benign": 51_639, "hijacked": 12_349, "injection_point": 5_536, "failed_injection": 1_500,
    }
    assert steps["is_first_h"].sum() == 5_536
    assert steps["y_scan"].sum() == 5_536 + 1_500
    assert traj.groupby("origin_split")["template_id"].nunique().to_dict() == {"train": 180, "val": 35, "test": 35}


@needs_interim
def test_readers_hide_fixed_test_by_default():
    assert "test" not in set(read_steps()["origin_split"])
    assert "test" not in set(read_trajectories()["origin_split"])
    dev_templates = set(read_trajectories()["template_id"])
    test_templates = set(read_trajectories(include_test=True).query("origin_split == 'test'")["template_id"])
    assert dev_templates.isdisjoint(test_templates)  # L1 on the fixed test set
