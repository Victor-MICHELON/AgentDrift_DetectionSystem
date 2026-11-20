"""Parse raw AgentDrift trajectories into interim tables.

    python -m agentdrift.data.load          # verify checksums, build, validate, write data/interim/*.parquet

Outputs (data/interim/):
    trajectories.parquet  one row per trajectory (metadata, template, world id, injection / first H step)
    steps.parquet         one row per step (thought, tool, args, obs, label) + targets y_scan, y_gate
    templates.parquet     the 250 task templates with expected_tools (oracle only, never a model input)

Targets (CONTEXT 3.3), t is 0-based:
    y_scan(t) = label(t-1) in {injection_point, failed_injection}, 0 at t = 0
    y_gate(t) = label(t) == hijacked
    is_first_h = first hijacked step of the trajectory (first-alert business metrics)
    y_traj     = category == attacked (provisional, JOURNAL: to confirm)

Readers `read_trajectories` / `read_steps` exclude the fixed test set unless asked explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from agentdrift.config import INTERIM, ROOT, load_config
from agentdrift.data.download import verify_checksums
from agentdrift.data.schema import DOMAINS, validate

INJECTION_LABELS = ("injection_point", "failed_injection")
TABLES = ("trajectories", "steps", "templates")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def world_id(world: dict) -> str:
    """Stable id of the simulated world (user, company, contacts). `date` is excluded: it is a property
    of the generation run, not of the identity, and is unique per world in the corpus anyway."""
    identity = {k: v for k, v in world.items() if k != "date"}
    return hashlib.sha1(canonical_json(identity).encode()).hexdigest()[:12]


def load_templates(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for t in json.loads((raw_dir / "pools" / f"{domain}_tasks.json").read_text(encoding="utf-8")):
            assert t["agent_type"] == domain, t["task_id"]
            rows.append(
                {
                    "template_id": t["task_id"],
                    "domain": domain,
                    "instruction": t["instruction"],
                    "complexity": t["complexity"],
                    "expected_tools": list(t["expected_tools"]),
                }
            )
    return pd.DataFrame(rows)


def build_tables(records: list[dict], templates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_instruction = templates.set_index("instruction")[["template_id", "complexity"]].to_dict("index")
    traj_rows, step_rows = [], []
    for r in records:
        tpl = by_instruction.get(r["task"])
        if tpl is None:
            raise ValueError(f"{r['id']}: task not found among the 250 templates")
        labels = [s["label"] for s in r["steps"]]
        inj = next((t for t, lab in enumerate(labels) if lab in INJECTION_LABELS), None)
        first_h = next((t for t, lab in enumerate(labels) if lab == "hijacked"), None)
        wid = world_id(r["world"])
        traj_rows.append(
            {
                "traj_id": r["id"],
                "origin_split": r["split"],
                "domain": r["agent"],
                "category": r["category"],
                "source_category": r["source_category"],
                "dataset_category": r["dataset_category"],
                "attack_type": r.get("attack_type"),
                "compliance": r.get("compliance"),
                "task": r["task"],
                "template_id": tpl["template_id"],
                "complexity": tpl["complexity"],
                "world_id": wid,
                "world_json": canonical_json(r["world"]),
                "n_steps": len(labels),
                "injection_step": inj,
                "first_h_step": first_h,
                "y_traj": int(r["category"] == "attacked"),
            }
        )
        for t, s in enumerate(r["steps"]):
            step_rows.append(
                {
                    "traj_id": r["id"],
                    "t": t,
                    "origin_split": r["split"],
                    "domain": r["agent"],
                    "template_id": tpl["template_id"],
                    "world_id": wid,
                    "source_category": r["source_category"],
                    "thought": s["thought"],
                    "tool": s["tool"],
                    "args_json": canonical_json(s["args"]),
                    "obs": s["obs"],
                    "label": s["label"],
                    "y_scan": int(t > 0 and labels[t - 1] in INJECTION_LABELS),
                    "y_gate": int(s["label"] == "hijacked"),
                    "is_first_h": t == first_h,
                }
            )

    traj = pd.DataFrame(traj_rows)
    for col in ("injection_step", "first_h_step"):
        traj[col] = traj[col].astype("Int64")
    traj["y_traj"] = traj["y_traj"].astype("int8")
    steps = pd.DataFrame(step_rows)
    steps[["y_scan", "y_gate"]] = steps[["y_scan", "y_gate"]].astype("int8")
    return traj, steps


def build(verify: bool = True) -> dict[str, pd.DataFrame]:
    cfg = load_config("data")["agentdrift"]
    raw = ROOT / cfg["raw_dir"]
    if verify:
        verify_checksums(raw, cfg["sha256"])
    templates = load_templates(raw)
    records = [r for part in ("train", "val", "test") for r in read_jsonl(raw / cfg["split_dir"] / f"{part}.jsonl")]
    traj, steps = build_tables(records, templates)
    violations = validate(traj, steps, templates)
    if not violations.empty:
        summary = violations.groupby("rule").size().to_string()
        raise ValueError(f"{violations['traj_id'].nunique()} trajectories violate invariants:\n{summary}")
    return {"trajectories": traj, "steps": steps, "templates": templates}


def write(tables: dict[str, pd.DataFrame], out_dir: Path = INTERIM) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def _read(name: str, include_test: bool) -> pd.DataFrame:
    path = INTERIM / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m agentdrift.data.load` first")
    df = pd.read_parquet(path)
    return df if include_test else df[df["origin_split"] != "test"].reset_index(drop=True)


def read_trajectories(include_test: bool = False) -> pd.DataFrame:
    """Trajectory table. The fixed test set (35 templates) is excluded unless include_test=True."""
    return _read("trajectories", include_test)


def read_steps(include_test: bool = False) -> pd.DataFrame:
    """Step table. The fixed test set (35 templates) is excluded unless include_test=True."""
    return _read("steps", include_test)


def read_templates() -> pd.DataFrame:
    """Task templates with expected_tools. Oracle / analysis only: never a model input (L12)."""
    return pd.read_parquet(INTERIM / "templates.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/interim/*.parquet from raw AgentDrift.")
    parser.add_argument("--no-verify", action="store_true", help="skip the checksum verification")
    args = parser.parse_args(argv)
    tables = build(verify=not args.no_verify)
    write(tables)
    traj, steps = tables["trajectories"], tables["steps"]
    print(f"Wrote {INTERIM.relative_to(ROOT)}/{{{','.join(TABLES)}}}.parquet")
    print(f"  {len(traj):,} trajectories, {len(steps):,} steps, {traj['template_id'].nunique()} templates, "
          f"{traj['world_id'].nunique():,} worlds")
    print(steps.groupby("origin_split")[["y_scan", "y_gate", "is_first_h"]].sum().to_string())
    gaps = (traj["first_h_step"] - traj["injection_step"]).groupby(traj["source_category"]).value_counts()
    print("Gap injection -> first H:\n" + gaps.sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
