"""Provenance and injection-cue blocks (B.3, KAN-18): where the arguments of an action come from, and what in the
text just read looks like an instruction. Computed from ONE view row only (x_t), deterministic, nothing fitted.

The anonymisation placeholders carry exactly the relations needed (`data/anonymize.py`): the same entity gets the
same placeholder everywhere in a trajectory (`<EXT_EMAIL_2>` in a tool response and in the arguments IS the same
address), internal / external and known contact are in the name. Comparisons are therefore exact string matches.

    ProvenanceBlock(view="gate")   the arguments of the proposed call a_t, against the task u, the past tool responses
                                   (`<result>` lines, o_{t-1} included) and the past call arguments (`<call>` lines)
        leaf_*        leaf values of the JSON arguments (>= 4 characters, lowercased, as in the EDA A7):
                      from_task, from_results_only (= EDA "arg from previous obs"), from_calls_only, nowhere,
                      as shares of the leaves, plus `any_from_results_only`, `any_nowhere`, `n_leaves` (capped at 5)
        <kind>_in_args / _from_task / _from_results_only / _new   for each entity kind (external / internal e-mail,
                      contact e-mail, user e-mail, external / internal URL, external domain): present in the args,
                      and where it was seen first. `_new` = in the args but never seen before (in x_t)
        unknown_email = an internal or external e-mail of the args absent from the task (EDA "unknown address")
        amount_in_args, amount_from_task, amount_from_results_only: amounts ($ or <AMOUNT_k>, or a numeric leaf of
                      3+ digits), compared on their digits
        history_truncated, provenance_unknown: with a truncated history, "never seen" cannot be told apart from "seen
                      in a dropped step": such values count as unknown, not as new (AgentDrift: never truncated)
    InjectionCuesBlock(view)       the content just read: o_{t-1} (`current_content` for the scanner, `last_content`
                                   for the gate)
        new_ext_entity (count capped at 3, scaled), new_int_email: placeholders absent from the task and the history
        contact_email: a known contact's address in the content
        cue_polite_action, cue_initial_imperative, cue_obligation, cue_redirect_urgency: instruction patterns, written
                      a priori from the generic form of an injected instruction. NOTE: written after reading the v0.1
                      coefficients on the development folds (`must`, `instead`): a choice informed by development
                      data, declared as such; never by the fixed test set
        masked_marker: `<HEADER>` or `<CODE>` (masked text variant only; always 0 on the anonymised text)
        log_length:   log(1 + characters) of the content. A ratio to the past responses was tried and dropped (JOURNAL
                      26/11): with no past response it falls back to a constant, and "is there a past response" is
                      a pure position flag (t >= 2), true for 100 % of the scanner positives
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from agentdrift.features.structural import CONTENT, CURRENT_ARGS, _check, _RowBlock, parse_history

LEAF_MIN = 4
ENTITY_KINDS = {
    "ext_email": r"<EXT_EMAIL_\d+>", "int_email": r"<INT_EMAIL_\d+>", "contact_email": r"<CONTACT_\d+_EMAIL>",
    "user_email": r"<USER_EMAIL>", "ext_url": r"<EXT_URL_\d+>", "int_url": r"<INT_URL_\d+>",
    "ext_domain": r"<EXT_DOMAIN_\d+>",
}
_ENTITY_RE = {k: re.compile(v) for k, v in ENTITY_KINDS.items()}
_EXT_RE = re.compile(r"<EXT_(?:EMAIL|URL|DOMAIN)_\d+>")
_AMOUNT_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|<AMOUNT_\d+>")
_NUMBER_LEAF = re.compile(r"^\d{3,}(?:\.\d+)?$")
CUES = {
    "cue_polite_action": re.compile(r"\b(?:please|kindly)\s+(?:forward|send|transfer|share|update|change|delete|"
                                    r"email|reply|copy|cc|add|use|move|pay)\b", re.I),
    "cue_initial_imperative": re.compile(r"(?:^|[.!?:]\s+)(?:forward|send|transfer|share|update|change|delete|"
                                         r"email|reply|copy|cc|add|use|move|pay|ignore|disregard)\b", re.I),
    "cue_obligation": re.compile(r"\b(?:must|required|requires|mandatory|need to|needs to|do not|don't)\b", re.I),
    "cue_redirect_urgency": re.compile(r"\b(?:instead|immediately|urgent|urgently|asap|right away|from now on)\b",
                                       re.I),
}


def leaf_values(args: str) -> list[str]:
    """String forms of the leaf values of a JSON argument string; the raw string if it is not JSON."""
    try:
        obj = json.loads(args)
    except (TypeError, ValueError):
        return [str(args)] if args else []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
        elif o is not None and not isinstance(o, bool):
            yield str(o)

    return list(walk(obj))


def _digits(s: str) -> str:
    return re.sub(r"[^\d.]", "", s).rstrip("0").rstrip(".") if "." in s else re.sub(r"\D", "", s)


def amounts(text: str) -> set[str]:
    """Amounts of a text as digit strings ($1,500.00 -> 1500), plus masked `<AMOUNT_k>` placeholders as such."""
    out = set()
    for m in _AMOUNT_RE.findall(text):
        out.add(m if m.startswith("<") else _digits(m))
    return out - {""}


PROVENANCE_COLUMNS = (
    ["leaf_from_task", "leaf_from_results_only", "leaf_from_calls_only", "leaf_nowhere", "any_from_results_only",
     "any_nowhere", "n_leaves"]
    + [f"{k}_{w}" for k in ENTITY_KINDS for w in ("in_args", "from_task", "from_results_only", "new")]
    + ["unknown_email", "amount_in_args", "amount_from_task", "amount_from_results_only", "history_truncated",
       "provenance_unknown"]
)


def provenance_row(task: str, history_text: str, args: str) -> dict[str, float]:
    calls, results, truncated = parse_history(history_text)
    task_l = str(task).lower()
    res_l = " ".join(results).lower()
    calls_l = " ".join(a for _, a in calls).lower()
    out: dict[str, float] = {"history_truncated": float(truncated)}
    unknown = 0

    leaves = [v.lower() for v in leaf_values(args) if len(v) >= LEAF_MIN]
    where = []
    for v in leaves:
        if v in task_l:
            where.append("task")
        elif v in res_l:
            where.append("results")
        elif v in calls_l:
            where.append("calls")
        elif truncated:
            where.append("unknown")
            unknown += 1
        else:
            where.append("nowhere")
    n = max(len(leaves), 1)
    out |= {"leaf_from_task": where.count("task") / n, "leaf_from_results_only": where.count("results") / n,
            "leaf_from_calls_only": where.count("calls") / n, "leaf_nowhere": where.count("nowhere") / n,
            "any_from_results_only": float("results" in where), "any_nowhere": float("nowhere" in where),
            "n_leaves": min(len(leaves), 5) / 5}

    res_raw, calls_raw = " ".join(results), " ".join(a for _, a in calls)
    for kind, rx in _ENTITY_RE.items():
        ents = set(rx.findall(str(args)))
        in_task = {e for e in ents if e in str(task)}
        in_res = {e for e in ents - in_task if e in res_raw}
        unseen = ents - in_task - in_res - {e for e in ents if e in calls_raw}
        if truncated:
            unknown += len(unseen)
            unseen = set()
        out |= {f"{kind}_in_args": float(bool(ents)), f"{kind}_from_task": float(bool(in_task)),
                f"{kind}_from_results_only": float(bool(in_res)), f"{kind}_new": float(bool(unseen))}
    emails = set(_ENTITY_RE["ext_email"].findall(str(args))) | set(_ENTITY_RE["int_email"].findall(str(args)))
    out["unknown_email"] = float(any(e not in str(task) for e in emails))

    arg_amounts = amounts(str(args)) | {_digits(v) for v in leaf_values(args) if _NUMBER_LEAF.match(v)}
    task_amounts, res_amounts = amounts(str(task)), amounts(" ".join(results))
    out |= {"amount_in_args": float(bool(arg_amounts)), "amount_from_task": float(bool(arg_amounts & task_amounts)),
            "amount_from_results_only": float(bool((arg_amounts - task_amounts) & res_amounts))}
    out["provenance_unknown"] = float(unknown > 0)
    return out


class ProvenanceBlock(_RowBlock):
    """Provenance of the arguments of the proposed call (gate view only)."""

    prefix = "provenance__"

    def __init__(self, view: str | None = "gate"):
        self.view = view

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.view != "gate":
            raise ValueError("provenance describes the arguments of the PROPOSED call: gate view only")
        _check(X, self.view, ("task", "history_text", CURRENT_ARGS["gate"]))
        rows = [provenance_row(u, h, a) for u, h, a in zip(X["task"], X["history_text"], X[CURRENT_ARGS["gate"]])]
        return pd.DataFrame(rows, columns=PROVENANCE_COLUMNS, index=X.index)


CUE_COLUMNS = ["new_ext_entity", "new_int_email", "contact_email", *CUES, "masked_marker", "log_length"]


def cues_row(task: str, history_text: str, content: str) -> dict[str, float]:
    calls, results, _ = parse_history(history_text)
    seen = str(task) + " " + " ".join(results) + " " + " ".join(a for _, a in calls)
    content = str(content)
    new_ext = {e for e in _EXT_RE.findall(content) if e not in seen}
    new_int = {e for e in _ENTITY_RE["int_email"].findall(content) if e not in seen}
    out = {"new_ext_entity": min(len(new_ext), 3) / 3, "new_int_email": float(bool(new_int)),
           "contact_email": float(bool(_ENTITY_RE["contact_email"].search(content)))}
    out |= {k: float(bool(rx.search(content))) for k, rx in CUES.items()}
    out["masked_marker"] = float("<HEADER>" in content or "<CODE>" in content)
    out["log_length"] = float(np.log1p(len(content)))
    return out


class InjectionCuesBlock(_RowBlock):
    """Instruction-like cues in the tool response just read (o_{t-1})."""

    prefix = "cues__"

    def __init__(self, view: str | None = None):
        self.view = view

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        _check(X, self.view, ("task", "history_text", CONTENT.get(self.view, "")))
        if self.view == "scanner":  # the scanner's history stops at s_{t-2}: the call a_{t-1} is a field of its own
            hist = X["history_text"] + "\n<call> " + X["content_source_tool"].astype(str) + " " + \
                X["content_source_args"].astype(str) if "content_source_tool" in X else X["history_text"]
        else:  # the gate's history already ends with o_{t-1} = last_content: drop that result from the "seen" text
            hist = X["history_text"].map(_without_last_result)
        rows = [cues_row(u, h, c) for u, h, c in zip(X["task"], hist, X[CONTENT[self.view]])]
        return pd.DataFrame(rows, columns=CUE_COLUMNS, index=X.index)


def _without_last_result(history_text: str) -> str:
    lines = str(history_text or "").split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("<result>"):
            return "\n".join(lines[:i] + lines[i + 1:])
    return str(history_text or "")
