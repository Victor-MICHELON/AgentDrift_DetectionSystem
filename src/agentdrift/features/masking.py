"""Masking of generator artefacts (hypothesis H3, leaks L4 / L10; Jira KAN-19, part 1 = KAN-59).

The AgentDrift generator writes each attack type with its own tics: "CORRECTION:" opens partial injections,
"SECURITY ALERT:" failed ones, "Per policy NX-2026-SEC" full ones, and a few dollar amounts recur. A model that
learns these tics recognises the generator, not the attack. This module hides them with GENERIC rules, applied to
every text whatever its label, after the world anonymisation (`read_*(anonymized=True)`):

    header   an upper-case label that OPENS a sentence and is followed by ":" -> "<HEADER>:"
             ("SECURITY ALERT:", "CORRECTION:", "IMMEDIATE ACTION REQUIRED:"). Field names inside a list
             ("Name: x, DOB: 1990") are not at a sentence start and are kept; labels of fewer than 3 letters
             ("ID:") and identifiers with digits ("ERR-001:") are kept.
    code     the reference code that follows "per / under / according to ... policy | protocol | regulation |
             directive | mandate | guideline" -> "<CODE>" ("Per policy NX-2026-SEC" -> "Per policy <CODE>").
             The words stay: invoking a policy is a real persuasion technique; the code format is the tic.
    amount   "$3,000.00" -> "<AMOUNT_k>", k numbered by first appearance in the trajectory (task, then thought ->
             args -> obs of step 0, 1, ...), as the anonymisation does: the recurring values disappear, but
             "the same amount as in the request" stays visible.

Why rules and not the list of openers found by the EDA: that list was mined on the whole corpus (test included)
and missed variants of the train set ("COMPLIANCE NOTICE:", "ALERT:"...). The rules above were checked on the
train part only (JOURNAL 23/11). The variant without `thought` (the agent's confessions) is not done here: the
views drop it with `include_thought=False` (features/views.py).

Properties (tests/test_masking.py): deterministic, row-wise, nothing fitted (no preprocessing leak); causal
numbering; `args_json` stays valid JSON (only string values are touched); idempotent.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable

import pandas as pd

HEADER_TOKEN = "<HEADER>"
CODE_TOKEN = "<CODE>"
AMOUNT_TOKEN = "<AMOUNT_{k}>"
MASK_TOKEN_RE = re.compile(r"<(?:HEADER|CODE|AMOUNT_\d+)>")

# An upper-case label at a sentence start (text start, after . ! ? + space, a newline, an opening bracket or
# quote), made of 1 to 5 upper-case words, directly followed by ":".
HEADER_RE = re.compile(r"(?:^|(?<=[.!?]\s)|(?<=[\n\[(\"'*]))([A-Z]{2,}(?:[ _-][A-Z]{2,}){0,4})(?=\s?:)", re.MULTILINE)
MIN_HEADER_LETTERS = 3
POLICY_RE = re.compile(
    r"(?i)(\b(?:per|under|according to)\s+(?:[\w'.#<>-]+\s+){0,3}?"
    r"(?:policy|protocol|regulation|directive|mandate|guideline)\s+)"
    r"(#?[\w<>][\w<>.#/-]*)"
)
AMOUNT_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
FIELDS = ("thought", "args_json", "obs")


class AmountNumbering:
    """Causal numbering of amounts inside one trajectory: same amount -> same placeholder."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def __call__(self, raw: str) -> str:
        key = re.sub(r"[\s$,]", "", raw)
        key = key[:-3] if key.endswith(".00") else key
        if key not in self.ids:
            self.ids[key] = len(self.ids) + 1
        return AMOUNT_TOKEN.format(k=self.ids[key])


def _header(m: re.Match) -> str:
    label = m.group(1)
    return HEADER_TOKEN if sum(c.isalpha() for c in label) >= MIN_HEADER_LETTERS else label


def _code(m: re.Match) -> str:
    code = m.group(2)
    trail = code[len(code.rstrip(".")):]  # a sentence-final period is punctuation, not part of the code
    code = code.rstrip(".")
    if not code or MASK_TOKEN_RE.fullmatch(code) or not re.search(r"[\d-]", code):
        return m.group(0)  # "per policy and ..." : no code to hide
    return m.group(1) + CODE_TOKEN + trail


def mask_artifacts(text: str, amounts: AmountNumbering | None = None) -> str:
    """Mask headers, policy codes and amounts in one text. Pass the same `amounts` object for every text of a
    trajectory (in temporal order) to keep the numbering consistent; a fresh one is used otherwise."""
    if not isinstance(text, str) or not text:
        return text
    amounts = AmountNumbering() if amounts is None else amounts
    text = HEADER_RE.sub(_header, text)
    text = POLICY_RE.sub(_code, text)
    return AMOUNT_RE.sub(lambda m: amounts(m.group(0)), text)


def mask_json(args_json: str, amounts: AmountNumbering) -> str:
    """Mask the string values of a JSON document (keys and numbers untouched); non-JSON text is masked as text."""
    try:
        obj = json.loads(args_json)
    except (TypeError, ValueError):
        return mask_artifacts(args_json, amounts)

    def walk(v):
        if isinstance(v, str):
            return mask_artifacts(v, amounts)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return json.dumps(walk(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def mask_trajectory(task: str, steps: Iterable[dict]) -> tuple[str, list[dict]]:
    """Mask one trajectory in temporal order: task, then thought -> args_json -> obs of each step."""
    amounts = AmountNumbering()
    task = mask_artifacts(task, amounts)
    out = []
    for s in steps:
        s = dict(s)
        s["thought"] = mask_artifacts(s.get("thought", ""), amounts)
        s["args_json"] = mask_json(s.get("args_json", "{}"), amounts)
        s["obs"] = mask_artifacts(s.get("obs", ""), amounts)
        out.append(s)
    return task, out


def mask_tables(traj: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Masked copies of the (anonymised) trajectory and step tables; every other column is unchanged."""
    traj, steps = traj.copy(), steps.sort_values(["traj_id", "t"]).copy()
    tasks = traj.set_index("traj_id")["task"]
    new_task, new_cols = {}, {f: [] for f in FIELDS}
    index = []
    for tid, g in steps.groupby("traj_id", sort=False):
        t, rows = mask_trajectory(tasks.get(tid, ""), g[list(FIELDS)].to_dict("records"))
        new_task[tid] = t
        index.extend(g.index)
        for f in FIELDS:
            new_cols[f].extend(r[f] for r in rows)
    for f in FIELDS:
        steps.loc[index, f] = new_cols[f]
    traj["task"] = traj["traj_id"].map(new_task).fillna(traj["task"].map(lambda x: mask_artifacts(x)))
    return traj, steps.sort_index()


def mask_counts(text: pd.Series) -> pd.DataFrame:
    """Which rule fired in each text (bool per rule), computed on the unmasked text."""
    def header(x: str) -> bool:
        return any(sum(c.isalpha() for c in m.group(1)) >= MIN_HEADER_LETTERS for m in HEADER_RE.finditer(x))

    def code(x: str) -> bool:
        return any(not MASK_TOKEN_RE.fullmatch(m.group(2)) and re.search(r"[\d-]", m.group(2))
                   for m in POLICY_RE.finditer(x))

    return pd.DataFrame({
        "header": text.map(header), "policy code": text.map(code), "amount": text.str.contains(AMOUNT_RE),
    })


def main() -> int:
    """Smoke run on the anonymised dev tables: how much text each rule touches."""
    from agentdrift.data.load import read_steps

    steps = read_steps(anonymized=True)
    c = mask_counts(steps["obs"]).groupby(steps["label"]).mean().round(3)
    print(c.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
