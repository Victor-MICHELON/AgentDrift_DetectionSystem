"""ATBench adapter: convert the external test set to the common step schema (plan step 11, CONTEXT 8.1 item 5).

    python -m agentdrift.data.external      # verify checksum, convert, validate, write data/interim/atbench_*.parquet

ATBench (AI45Research, revision pinned in configs/data.yaml) stores each trajectory as a conversation:
    user         {content}                       the task, and sometimes follow-up requests (multi-turn)
    agent        {thought, action}               action = '{"name": tool, "arguments": {...}}' for a tool call,
                                                 or 'Complete{"response": "..."}' for a reply to the user
    environment  {content}                       the tool response, right after the call that produced it
plus trajectory-level fields: label (1 = unsafe), risk_source, failure_mode, real_world_harm, reason, tool_used.

Mapping (one step = one tool call, as in AgentDrift):
    step t       the t-th tool call (0-based): thought, tool, args_json (canonical JSON, same form as
                 `data.load.canonical_json`), obs = the environment message that follows it ("" if none)
    replies      agent replies to the user are not tool calls: no step (they are counted per trajectory)
    task         the FIRST user message only. Follow-up user messages arrive during the trajectory: putting
                 them in `task` would show the detector a request made after step t (future, L3). They are kept,
                 causally, in the step column `user_followup` (the messages received just before that call) and
                 in `n_user_turns`; the views of `features/views.py` do not read them yet (open point, Lot B)
    label        no step labels: `label` is missing (None). Evaluation is at trajectory level only (JOURNAL
                 19/11): y_traj = ATBench label, a trajectory alarm = any step alarm

Outputs (data/interim/):
    atbench_trajectories.parquet  one row per trajectory: traj_id, task, y_traj, risk_source, in_scope, ...
    atbench_steps.parquet         one row per tool call: traj_id, t, thought, tool, args_json, obs, label, ...
    atbench_tools.parquet         the tools offered to the agent in each trajectory, with their descriptions

Columns that reveal the answer and must NEVER be model inputs: y_traj, label_unsafe, risk_source, failure_mode,
real_world_harm, annotator_reason, in_scope (see `FORBIDDEN_TRAJECTORY_FIELDS`). ATBench is never used to choose
anything (L8): thresholds and models are fixed on AgentDrift and transferred as they are.

Known blind spot, kept visible on purpose: for `tool_description_injection` the injected instruction sits in the
tool description (`atbench_tools`), which the scanner (tool responses only) never reads.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import pandas as pd
import pandera.pandas as pa

from agentdrift.config import INTERIM, ROOT, load_config
from agentdrift.data.download import verify_checksums
from agentdrift.data.load import canonical_json

TRAJ_PATH = INTERIM / "atbench_trajectories.parquet"
STEPS_PATH = INTERIM / "atbench_steps.parquet"
TOOLS_PATH = INTERIM / "atbench_tools.parquet"
ORIGIN = "atbench"
REPLY_RE = re.compile(r"^\s*Complete\s*(\{.*\})?\s*$", re.DOTALL)

FORBIDDEN_TRAJECTORY_FIELDS = {
    "y_traj": "trajectory label",
    "label_unsafe": "copy of the ATBench label",
    "risk_source": "kind of risk: tells whether and how the trajectory is attacked",
    "failure_mode": "outcome of the attack",
    "real_world_harm": "outcome of the attack",
    "annotator_reason": "the annotator's explanation of the label",
    "in_scope": "derived from risk_source and the label",
}


def _cfg() -> dict:
    return load_config("data")["atbench"]


def read_raw() -> list[dict]:
    cfg = _cfg()
    path = ROOT / cfg["raw_dir"] / "ATBench" / "test.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_action(action: str | None) -> tuple[str, str | None, object]:
    """Classify an agent action: ("call", tool, arguments) | ("reply", None, text) | ("other", None, raw)."""
    if not action or not action.strip():
        return "other", None, action
    try:
        obj = json.loads(action)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and "name" in obj:
        return "call", str(obj["name"]), obj.get("arguments", {})
    m = REPLY_RE.match(action)
    if m:
        try:
            text = json.loads(m.group(1)).get("response", "") if m.group(1) else ""
        except ValueError:
            text = m.group(1)
        return "reply", None, text
    return "other", None, action


def args_to_json(arguments: object, stats: Counter) -> str:
    """Canonical JSON of the arguments. A JSON string is parsed first (6 cases in ATBench); anything else that is
    not a dict is wrapped as is and counted."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
            stats["arguments_given_as_json_string"] += 1
        except ValueError:
            stats["arguments_not_json"] += 1
            return arguments
    if arguments is None:
        arguments = {}
    return canonical_json(arguments)


def convert_record(rec: dict, scope_sources: set[str], stats: Counter) -> tuple[dict, list[dict], list[dict]]:
    """One ATBench record -> (trajectory row, step rows, tool rows)."""
    traj_id = f"{ORIGIN}_{int(rec['id']):04d}"
    conv = [m for block in rec["contents"] for m in block]
    users = [m.get("content", "") for m in conv if m["role"] == "user"]
    steps, pending_user, n_replies, turn = [], [], 0, -1
    for i, m in enumerate(conv):
        role = m["role"]
        if role == "user":
            turn += 1
            if turn > 0:
                pending_user.append(m.get("content", ""))
            continue
        if role != "agent":
            continue  # environment messages are attached to the call that precedes them
        kind, tool, payload = parse_action(m.get("action"))
        if kind == "reply":
            n_replies += 1
            continue
        if kind == "other":
            stats["agent_actions_not_understood"] += 1
            continue
        nxt = conv[i + 1] if i + 1 < len(conv) else None
        obs = nxt.get("content", "") if nxt is not None and nxt["role"] == "environment" else ""
        if not (nxt is not None and nxt["role"] == "environment"):
            stats["calls_without_response"] += 1
        steps.append({
            "traj_id": traj_id, "t": len(steps), "turn": max(turn, 0), "thought": m.get("thought") or "",
            "tool": tool, "args_json": args_to_json(payload, stats), "obs": obs,
            "user_followup": "\n\n".join(pending_user), "label": None, "origin_split": ORIGIN,
        })
        pending_user = []
    if pending_user:
        stats["followups_after_last_call"] += 1
    label = int(rec["label"])
    traj = {
        "traj_id": traj_id, "atbench_id": int(rec["id"]), "origin_split": ORIGIN,
        "task": users[0] if users else "", "n_user_turns": len(users), "n_steps": len(steps), "n_replies": n_replies,
        "n_tools_available": len(rec.get("tool_used", [])),
        "y_traj": label, "label_unsafe": label, "risk_source": rec["risk_source"],
        "failure_mode": rec.get("failure_mode"), "real_world_harm": rec.get("real_world_harm"),
        "annotator_reason": rec.get("reason"),
        "in_scope": label == 0 or rec["risk_source"] in scope_sources,
    }
    tools = [{"traj_id": traj_id, "tool": t.get("name", ""), "description": t.get("description", "")}
             for t in rec.get("tool_used", [])]
    return traj, steps, tools


def convert(raw: list[dict], scope_sources: set[str] | None = None) -> tuple[dict[str, pd.DataFrame], Counter]:
    scope_sources = set(_cfg()["scope_risk_sources"]) if scope_sources is None else set(scope_sources)
    stats: Counter = Counter()
    trajs, steps, tools = [], [], []
    for rec in raw:
        tr, st, tl = convert_record(rec, scope_sources, stats)
        trajs.append(tr)
        steps += st
        tools += tl
    traj = pd.DataFrame(trajs)
    step_cols = ["traj_id", "t", "turn", "thought", "tool", "args_json", "obs", "user_followup", "label", "origin_split"]
    st = pd.DataFrame(steps, columns=step_cols).astype({"t": "int64", "turn": "int64", "label": "object"})
    for c in ("y_traj", "label_unsafe"):
        traj[c] = traj[c].astype("int8")
    stats["trajectories_without_call"] = int((traj["n_steps"] == 0).sum())
    return {"trajectories": traj, "steps": st, "tools": pd.DataFrame(tools, columns=["traj_id", "tool", "description"])}, stats


# --------------------------------------------------------------------------- validation

TRAJECTORIES = pa.DataFrameSchema({
    "traj_id": pa.Column(str, unique=True, checks=pa.Check.str_startswith(f"{ORIGIN}_")),
    "task": pa.Column(str, pa.Check(lambda s: s.str.len() > 0, error="empty task")),
    "n_user_turns": pa.Column(int, pa.Check.ge(1)),
    "n_steps": pa.Column(int, pa.Check.ge(0)),
    "y_traj": pa.Column("int8", pa.Check.isin([0, 1])),
    "risk_source": pa.Column(str),
    "in_scope": pa.Column(bool),
}, strict=False)

STEPS = pa.DataFrameSchema({
    "traj_id": pa.Column(str),
    "t": pa.Column("int64", pa.Check.ge(0)),
    "tool": pa.Column(str, pa.Check(lambda s: s.str.len() > 0, error="empty tool name")),
    "args_json": pa.Column(str),
    "obs": pa.Column(str),
    "label": pa.Column(object, nullable=True),
}, strict=False, unique=["traj_id", "t"])


def validate(tables: dict[str, pd.DataFrame]) -> None:
    """Schemas + invariants: contiguous t from 0, step counts match, scope rule, nothing revealing in steps."""
    traj, steps = tables["trajectories"], tables["steps"]
    TRAJECTORIES.validate(traj, lazy=True)
    STEPS.validate(steps, lazy=True)
    g = steps.groupby("traj_id")["t"]
    if not ((g.min() == 0) & (g.max() == g.count() - 1)).all():
        raise ValueError("step index t must run 0..n-1 within each trajectory")
    n = steps.groupby("traj_id").size().reindex(traj["traj_id"], fill_value=0).to_numpy()
    if not (n == traj["n_steps"].to_numpy()).all():
        raise ValueError("n_steps does not match the step table")
    leaked = set(FORBIDDEN_TRAJECTORY_FIELDS) & set(steps.columns)
    if leaked:
        raise ValueError(f"label-revealing columns in the step table: {sorted(leaked)}")
    if steps["label"].notna().any():
        raise ValueError("ATBench has no step labels: `label` must stay missing")


def summary(tables: dict[str, pd.DataFrame], stats: Counter) -> pd.Series:
    traj, steps = tables["trajectories"], tables["steps"]
    sc = traj[traj["in_scope"]]
    return pd.Series({
        "trajectories": len(traj), "steps (tool calls)": len(steps),
        "in scope": len(sc), "in scope, unsafe": int(sc["y_traj"].sum()), "in scope, safe": int((sc["y_traj"] == 0).sum()),
        "multi-turn trajectories": int((traj["n_user_turns"] > 1).sum()),
        "steps after a follow-up request": int((steps["user_followup"] != "").sum()),
        "distinct tools": steps["tool"].nunique(),
        **{k: v for k, v in sorted(stats.items())},
    })


# --------------------------------------------------------------------------- I/O


def build(verify: bool = True) -> tuple[dict[str, pd.DataFrame], Counter]:
    cfg = _cfg()
    if verify:
        verify_checksums(ROOT / cfg["raw_dir"], cfg["sha256"])
    tables, stats = convert(read_raw())
    validate(tables)
    return tables, stats


def write(tables: dict[str, pd.DataFrame]) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    tables["trajectories"].to_parquet(TRAJ_PATH, index=False)
    tables["steps"].to_parquet(STEPS_PATH, index=False)
    tables["tools"].to_parquet(TOOLS_PATH, index=False)


def read_atbench(in_scope_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(trajectories, steps) of the external test set, restricted to the evaluation scope by default."""
    if not STEPS_PATH.exists():
        raise FileNotFoundError(f"{STEPS_PATH} missing: run `python -m agentdrift.data.external` first")
    traj, steps = pd.read_parquet(TRAJ_PATH), pd.read_parquet(STEPS_PATH)
    if in_scope_only:
        traj = traj[traj["in_scope"]].reset_index(drop=True)
        steps = steps[steps["traj_id"].isin(set(traj["traj_id"]))].reset_index(drop=True)
    return traj, steps


def main() -> int:
    tables, stats = build()
    write(tables)
    print(f"Wrote {', '.join(str(p.relative_to(ROOT)) for p in (TRAJ_PATH, STEPS_PATH, TOOLS_PATH))}")
    print(summary(tables, stats).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
