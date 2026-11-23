"""Manual annotations: tool risk table and blind injection_visible sheet.

Toy tests run everywhere; corpus tests are skipped until `python -m agentdrift.data.load` has run.
"""

import pandas as pd
import pytest

from agentdrift.config import INTERIM
from agentdrift.data.annotations import (
    INJECTION_KEY_PATH,
    INJECTION_VISIBLE_PATH,
    RISK_CATEGORIES,
    attach_injection_visible,
    attach_tool_risk,
    read_injection_visible,
    read_tool_risk,
    sample_injection_steps,
    tool_risk_problems,
)


def _toy(n_per_cat=5, split="train"):
    steps, traj = [], []
    for cat in ("attacked_full", "attacked_partial", "attacked_delayed", "benign"):
        for i in range(n_per_cat):
            tid = f"email_{cat}_{i:04d}"
            traj.append({"traj_id": tid, "task": f"task number {i}"})
            for t, label in enumerate(["benign", "injection_point" if cat != "benign" else "benign", "hijacked"]):
                steps.append({"traj_id": tid, "t": t, "origin_split": split, "domain": "email",
                              "source_category": cat, "tool": "read_email", "args_json": "{}",
                              "obs": f"obs {t}", "label": label if cat != "benign" or t < 2 else "benign"})
    return pd.DataFrame(steps), pd.DataFrame(traj)


def test_sheet_is_blind_seeded_and_balanced():
    steps, traj = _toy()
    sheet, key = sample_injection_steps(steps, traj, per_category=3, seed=1)
    assert len(sheet) == len(key) == 9
    assert "traj_id" not in sheet.columns and "source_category" not in sheet.columns
    assert not sheet.astype(str).apply(lambda c: c.str.contains("attacked")).any().any()
    joined = key.merge(steps, on=["traj_id", "t"])
    assert (joined["label"] == "injection_point").all()
    assert joined["source_category"].value_counts().eq(3).all()
    again, _ = sample_injection_steps(steps, traj, per_category=3, seed=1)
    pd.testing.assert_frame_equal(sheet, again)


def test_fixed_test_set_is_never_sampled():
    steps, traj = _toy(split="test")
    with pytest.raises(ValueError, match="L5"):
        sample_injection_steps(steps, traj)


def test_annotations_are_validated_and_joined(tmp_path):
    steps, traj = _toy()
    sheet, key = sample_injection_steps(steps, traj, per_category=2, seed=0)
    sheet.loc[0, "injection_visible"] = " OUI "
    sheet.loc[1, "injection_visible"] = "non"
    sheet.to_csv(tmp_path / "s.csv", index=False)
    key.to_csv(tmp_path / "k.csv", index=False)
    done = read_injection_visible(tmp_path / "s.csv", tmp_path / "k.csv")
    assert list(done["injection_visible"]) == ["oui", "non"]
    out = attach_injection_visible(steps, done)
    assert out["injection_visible"].notna().sum() == 2 and out["injection_visible"].sum() == 1

    sheet.loc[2, "injection_visible"] = "maybe"
    sheet.to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(ValueError, match="oui"):
        read_injection_visible(tmp_path / "s.csv", tmp_path / "k.csv")


def test_annotation_on_a_non_injection_step_is_rejected():
    steps, _ = _toy()
    wrong = pd.DataFrame({"traj_id": ["email_attacked_full_0000"], "t": [0], "injection_visible": ["oui"]})
    with pytest.raises(ValueError, match="not an injection_point"):
        attach_injection_visible(steps, wrong)


def test_unknown_tool_is_never_filled_silently():
    table = pd.DataFrame({"tool": ["read_email"], "domain": ["email"], "risk": ["read"]})
    steps = pd.DataFrame({"tool": ["read_email", "send_email"]})
    with pytest.raises(ValueError, match="send_email"):
        attach_tool_risk(steps, table)


def test_tool_risk_file_uses_only_known_categories():
    table = read_tool_risk()
    assert set(table["risk"]) <= set(RISK_CATEGORIES)
    assert table["tool"].is_unique


@pytest.mark.skipif(not (INTERIM / "steps.parquet").exists(), reason="run `python -m agentdrift.data.load` first")
def test_corpus_every_tool_has_exactly_one_category():
    steps = pd.read_parquet(INTERIM / "steps.parquet", columns=["tool", "domain"])
    assert tool_risk_problems(read_tool_risk(), steps) == []


@pytest.mark.skipif(not (INJECTION_VISIBLE_PATH.exists() and INJECTION_KEY_PATH.exists()
                         and (INTERIM / "steps.parquet").exists()), reason="annotation sheet not generated")
def test_corpus_sheet_points_at_dev_injection_steps():
    steps = pd.read_parquet(INTERIM / "steps.parquet", columns=["traj_id", "t", "label", "origin_split"])
    key = pd.read_csv(INJECTION_KEY_PATH, dtype={"traj_id": str}).merge(steps, on=["traj_id", "t"], how="left")
    assert (key["label"] == "injection_point").all()
    assert (key["origin_split"] != "test").all()
    read_injection_visible(annotated_only=False)  # values valid


def test_risk_by_label_is_hidden_while_the_table_is_a_draft():
    from agentdrift.data.audit import tool_risk_by_group, visible_vs_heuristic

    steps = pd.DataFrame({"tool": ["read_email", "send_email"], "label": ["benign", "hijacked"],
                          "source_category": ["benign", "attacked_full"]})
    table = pd.DataFrame({"tool": ["read_email", "send_email"], "risk": ["read", "external_send"]})
    table.attrs["status"] = "draft"
    assert tool_risk_by_group(steps, table) is None
    table.attrs["status"] = "validated"
    shown = tool_risk_by_group(steps, table)
    assert shown.loc["H full", "external_send"] == 1.0 and shown.loc["B", "read"] == 1.0
    assert visible_vs_heuristic(pd.DataFrame(), steps) is None
