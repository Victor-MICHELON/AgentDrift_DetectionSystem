"""Manual annotations (PLAN_DONNEES step 6): tool risk categories and injection visibility.

    python -m agentdrift.data.annotations --sample   # write the blank injection_visible.csv sheet (never overwrites)
    python -m agentdrift.data.annotations            # validate both files and print coverage

Files (versioned, edited by hand in configs/annotations/):
- tool_risk.yaml: each of the 103 tools -> one of 6 risk categories. Joined on the tool name.
- injection_visible.csv: a seeded sample of injection_point steps (full / partial / delayed attacks), annotated
  oui / non by a human. injection_visible_key.csv maps each sample_id to (traj_id, t), the join key.

The annotations are joined at read time (`attach_tool_risk`, `attach_injection_visible`) rather than written into
steps.parquet: they are edited by hand after the tables are built, and rebuilding the tables (and the anonymised
copies) for every edit would be error-prone.

Leakage rules: the sample comes from train + val only (the fixed test set is never looked at, L5). The sheet is
blind: shuffled, without the attack category and without traj_id (which spells the category, e.g.
`banking_attacked_delayed_0014`), so the annotator does not know whether the injection succeeded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from agentdrift.config import CONFIGS, ROOT
from agentdrift.data.load import read_steps, read_trajectories
from agentdrift.utils.seed import DEFAULT_SEED

ANNOTATIONS = CONFIGS / "annotations"
TOOL_RISK_PATH = ANNOTATIONS / "tool_risk.yaml"
INJECTION_VISIBLE_PATH = ANNOTATIONS / "injection_visible.csv"
INJECTION_KEY_PATH = ANNOTATIONS / "injection_visible_key.csv"

RISK_CATEGORIES = ("read", "internal_write", "external_send", "payment", "delete", "execute")
ATTACK_CATEGORIES = ("attacked_full", "attacked_partial", "attacked_delayed")
VISIBLE_VALUES = ("oui", "non")
SHEET_COLUMNS = ["sample_id", "domain", "task", "tool", "args_json", "obs", "injection_visible", "comment"]
KEY_COLUMNS = ["sample_id", "traj_id", "t"]


# --- tool risk -------------------------------------------------------------------------------------------------

def read_tool_risk(path: Path = TOOL_RISK_PATH) -> pd.DataFrame:
    """One row per tool: tool, domain, risk, ambiguous, note. Raises on an unknown category."""
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    table = pd.DataFrame.from_dict(doc["tools"], orient="index").rename_axis("tool").reset_index()
    table["ambiguous"] = table["ambiguous"].astype(bool)
    table["note"] = table.get("note", pd.Series(dtype=str)).fillna("")
    unknown = set(table["risk"]) - set(RISK_CATEGORIES)
    if unknown:
        raise ValueError(f"{path.name}: unknown risk categories {sorted(unknown)}")
    if set(doc["categories"]) != set(RISK_CATEGORIES):
        raise ValueError(f"{path.name}: categories must be exactly {RISK_CATEGORIES}")
    table.attrs.update(status=doc.get("status"), reviewer=doc.get("reviewer"))
    return table


def tool_risk_problems(table: pd.DataFrame, steps: pd.DataFrame) -> list[str]:
    """Coverage check against the corpus: every tool annotated once, with the right domain, no extra tool."""
    problems = []
    seen = steps.groupby("tool")["domain"].agg(lambda d: sorted(set(d)))
    missing = sorted(set(seen.index) - set(table["tool"]))
    extra = sorted(set(table["tool"]) - set(seen.index))
    if missing:
        problems.append(f"tools missing from tool_risk.yaml: {missing}")
    if extra:
        problems.append(f"tools in tool_risk.yaml but not in the corpus: {extra}")
    for tool, domain in table.set_index("tool")["domain"].items():
        if tool in seen.index and seen[tool] != [domain]:
            problems.append(f"{tool}: domain {domain!r} in yaml, {seen[tool]} in the corpus")
    return problems


def attach_tool_risk(steps: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add a `tool_risk` column. Raises if a tool has no category (never fills silently)."""
    table = read_tool_risk() if table is None else table
    mapping = table.set_index("tool")["risk"]
    out = steps.copy()
    out["tool_risk"] = out["tool"].map(mapping)
    if out["tool_risk"].isna().any():
        raise ValueError(f"tools without a risk category: {sorted(out.loc[out['tool_risk'].isna(), 'tool'].unique())}")
    return out


# --- injection visibility --------------------------------------------------------------------------------------

def sample_injection_steps(steps: pd.DataFrame, traj: pd.DataFrame, per_category: int = 100,
                           seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blank annotation sheet and its key: `per_category` injection_point steps per attack category, shuffled.

    Returns (sheet, key). The sheet is blind (no category, no traj_id); the key maps sample_id -> (traj_id, t).
    `steps` and `traj` must exclude the fixed test set; the text should be the anonymised one.
    """
    if (steps["origin_split"] == "test").any():
        raise ValueError("the fixed test set must not be sampled for annotation (L5)")
    inj = steps[steps["label"] == "injection_point"].merge(traj[["traj_id", "task"]], on="traj_id", how="left")
    rng = np.random.default_rng(seed)
    parts = []
    for cat in ATTACK_CATEGORIES:
        pool = inj[inj["source_category"] == cat].sort_values(["traj_id", "t"])
        idx = rng.choice(len(pool), size=min(per_category, len(pool)), replace=False)
        parts.append(pool.iloc[np.sort(idx)])
    sheet = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "sample_id", np.arange(1, len(sheet) + 1))
    sheet["injection_visible"] = ""
    sheet["comment"] = ""
    return sheet[SHEET_COLUMNS], sheet[KEY_COLUMNS]


def read_injection_visible(path: Path = INJECTION_VISIBLE_PATH, key_path: Path = INJECTION_KEY_PATH,
                           annotated_only: bool = True) -> pd.DataFrame:
    """The annotation sheet joined to its key (adds traj_id, t). Values are normalised to oui / non; anything
    else raises."""
    sheet = pd.read_csv(path, dtype={"injection_visible": str, "comment": str}, keep_default_na=False)
    key = pd.read_csv(key_path, dtype={"traj_id": str})
    if set(sheet["sample_id"]) != set(key["sample_id"]):
        raise ValueError(f"{path.name} and {key_path.name} do not have the same sample_id")
    sheet = sheet.merge(key, on="sample_id", how="left")
    sheet["injection_visible"] = sheet["injection_visible"].str.strip().str.lower()
    bad = sorted(set(sheet["injection_visible"]) - set(VISIBLE_VALUES) - {""})
    if bad:
        raise ValueError(f"{path.name}: injection_visible must be one of {VISIBLE_VALUES} or empty, got {bad}")
    if sheet.duplicated(["traj_id", "t"]).any():
        raise ValueError(f"{path.name}: duplicated (traj_id, t)")
    return sheet[sheet["injection_visible"] != ""].reset_index(drop=True) if annotated_only else sheet


def attach_injection_visible(steps: pd.DataFrame, sheet: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `injection_visible` (1 / 0, NA where not annotated) on the annotated injection_point steps."""
    sheet = read_injection_visible() if sheet is None else sheet
    values = sheet.assign(injection_visible=(sheet["injection_visible"] == "oui").astype("Int8"))
    out = steps.drop(columns="injection_visible", errors="ignore").merge(
        values[["traj_id", "t", "injection_visible"]], on=["traj_id", "t"], how="left")
    joined = out["injection_visible"].notna()
    if (out.loc[joined, "label"] != "injection_point").any():
        raise ValueError("injection_visible annotated on a step that is not an injection_point")
    if joined.sum() != len(sheet):
        raise ValueError(f"{len(sheet) - joined.sum()} annotated rows do not match any step")
    return out


def annotation_progress(sheet: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Annotated rows and share of `oui` per attack category (category recovered from the steps, not the sheet)."""
    cat = steps.set_index(["traj_id", "t"])["source_category"]
    s = sheet.assign(source_category=[cat.get((a, b)) for a, b in zip(sheet["traj_id"], sheet["t"])])
    done = s["injection_visible"] != ""
    return pd.DataFrame({
        "sampled": s.groupby("source_category").size(),
        "annotated": s[done].groupby("source_category").size(),
        "share_oui": s[done].groupby("source_category")["injection_visible"].apply(lambda v: (v == "oui").mean()),
    }).reindex(list(ATTACK_CATEGORIES)).fillna({"annotated": 0}).astype({"annotated": int})


# --- CLI -------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual annotations: sample the sheet, or validate both files.")
    parser.add_argument("--sample", action="store_true", help="write the blank injection_visible.csv sheet")
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="overwrite an existing sheet (destroys annotations)")
    args = parser.parse_args(argv)

    steps = read_steps(anonymized=True)
    if args.sample:
        if INJECTION_VISIBLE_PATH.exists() and not args.force:
            print(f"{INJECTION_VISIBLE_PATH.relative_to(ROOT)} exists: not overwritten (use --force)")
            return 1
        sheet, key = sample_injection_steps(steps, read_trajectories(anonymized=True), args.per_category)
        sheet.to_csv(INJECTION_VISIBLE_PATH, index=False)
        key.to_csv(INJECTION_KEY_PATH, index=False)
        print(f"Wrote {INJECTION_VISIBLE_PATH.relative_to(ROOT)}: {len(sheet)} rows to annotate "
              f"(key: {INJECTION_KEY_PATH.name})")
        return 0

    table = read_tool_risk()
    problems = tool_risk_problems(table, read_steps(include_test=True))
    print(f"tool_risk.yaml: {len(table)} tools, status={table.attrs['status']}, reviewer={table.attrs['reviewer']}")
    print(table["risk"].value_counts().reindex(list(RISK_CATEGORIES)).to_string())
    print(f"ambiguous: {int(table['ambiguous'].sum())} tools to discuss in review")
    for p in problems:
        print("PROBLEM:", p)
    if INJECTION_VISIBLE_PATH.exists():
        sheet = read_injection_visible(annotated_only=False)
        print(f"\ninjection_visible.csv: {len(sheet)} rows")
        print(annotation_progress(sheet, steps).to_string())
    else:
        print("\ninjection_visible.csv missing: run with --sample")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
