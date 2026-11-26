"""Structure blocks (B.3, KAN-18): what the agent does, as risk categories of its tools, not as words.

Every value is computed from the columns of ONE view row only (x_t): the serialised history (`<call> tool {args}` /
`<result> ...` lines of `features/views.py`) and the fields of the current call. Causality is therefore the one of
the views, tested by perturbation in `tests/test_views.py`. The only outside input is the fixed table
`configs/annotations/tool_risk.yaml` (read, never fitted, never tuned on labels; its status is written in every run).

    StructureBlock(view)   risk categories (read, internal_write, external_send, payment, delete, execute, unknown)
        cur_<cat>              category of the CURRENT tool: the proposed call a_t (gate) or the call a_{t-1} that
                               produced the content being scanned (scanner)
        prev_<cat>             category of the call just before (all zeros if there is none)
        bi_<prev>><cur>        (previous, current) transition, fixed enumeration, only over REAL calls
        tri_<p2>><p1>><cur>    (2 before, 1 before, current), only over real calls. No "none" padding: a padded
                               n-gram such as none>read>read says "exactly one call before" (t = 2 for the scanner),
                               a position flag that was never on for a scanner positive (JOURNAL 26/11)
        past_<cat>             category present among the past calls (presence, not a count)
        done_external_send / done_payment   past calls of that category, capped at 2 and scaled to [0, 1]
        cat_change             current category differs from the previous one (0 if there is none)
        cur_risky              current category in {external_send, payment, delete, execute}
        first_risky            current call risky and no risky call before
    ToolNameBlock(view)    one indicator per tool name of the current call, vocabulary learned on the training rows
                           (unseen tool -> all zeros). A SEPARATE block: names do not exist in a left-out domain or in
                           ATBench, so it is the control of H1, never part of the "structure" variant.

Position (JOURNAL 19/11): no raw count of steps, no "none" token. What still depends on t is inherent to x_t
(presence features can only switch on as the history grows, an empty history is all zeros): measured by
`pipelines/run_blocks_check.py` (correlation of every feature with t) and by the structure + position control of H1. An unknown tool (absent from the table, e.g. ATBench) is `unknown`, never dropped.
"""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from agentdrift.data.annotations import RISK_CATEGORIES, read_tool_risk
from agentdrift.features.views import HISTORY_TRUNCATED, TAG_CALL, TAG_RESULT, VIEWS

CATEGORIES = (*RISK_CATEGORIES, "unknown")
RISKY = frozenset({"external_send", "payment", "delete", "execute"})
NONE = "none"
# Field holding the current call and the field holding the text just read, per view.
CURRENT_TOOL = {"scanner": "content_source_tool", "gate": "proposed_call_tool"}
CURRENT_ARGS = {"scanner": "content_source_args", "gate": "proposed_call_args"}
CONTENT = {"scanner": "current_content", "gate": "last_content"}
_CALL_RE = re.compile(rf"^{re.escape(TAG_CALL)} (\S+)(?: (.*))?$")


# ----------------------------------------------------------------------------- history parsing


def parse_history(history_text: str) -> tuple[list[tuple[str, str]], list[str], bool]:
    """(past calls as (tool, args), past results, truncated) of a serialised history. Thought lines are ignored."""
    calls, results = [], []
    lines = str(history_text or "").split("\n")
    truncated = bool(lines) and lines[0] == HISTORY_TRUNCATED
    for line in lines:
        m = _CALL_RE.match(line)
        if m:
            calls.append((m.group(1), m.group(2) or ""))
        elif line.startswith(TAG_RESULT):
            results.append(line[len(TAG_RESULT):].strip())
    return calls, results, truncated


@lru_cache(maxsize=1)
def _risk_table() -> tuple[dict[str, str], str]:
    table = read_tool_risk()
    return dict(zip(table["tool"], table["risk"])), str(table.attrs.get("status"))


def tool_risk_status() -> str:
    """`draft` or `validated` (configs/annotations/tool_risk.yaml), written in every run that uses the table."""
    return _risk_table()[1]


def risk_of(tool: str, mapping: dict[str, str] | None = None) -> str:
    mapping = _risk_table()[0] if mapping is None else mapping
    return mapping.get(str(tool), "unknown")


def _check(X: pd.DataFrame, view: str | None, needed: tuple[str, ...]) -> None:
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("structure blocks read named view fields: X must be a DataFrame")
    missing = [c for c in needed if c not in X.columns]
    if missing:
        raise ValueError(f"the {view} view fields {missing} are needed by this block and are not in X")


class _RowBlock(TransformerMixin, BaseEstimator):
    """A deterministic per-row block: `fit` only fixes the column names; output sparse float32."""

    prefix = ""

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_ = [f"{self.prefix}{c}" for c in self._frame(X.head(1)).columns]
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "feature_names_")
        f = self._frame(X)
        return sp.csr_matrix(f.to_numpy(dtype=np.float32))

    def frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """The block as a named DataFrame (analysis and tests)."""
        return self._frame(X).set_axis([f"{self.prefix}{c}" for c in self._frame(X.head(1)).columns], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)


# ----------------------------------------------------------------------------- structure


BIGRAMS = [f"bi_{p}>{c}" for p, c in product(CATEGORIES, CATEGORIES)]
TRIGRAMS = [f"tri_{a}>{b}>{c}" for a, b, c in product(CATEGORIES, CATEGORIES, CATEGORIES)]
STRUCTURE_COLUMNS = ([f"cur_{c}" for c in CATEGORIES] + [f"prev_{c}" for c in CATEGORIES] + BIGRAMS
                     + TRIGRAMS + [f"past_{c}" for c in CATEGORIES]
                     + ["done_external_send", "done_payment", "cat_change", "cur_risky", "first_risky"])


def structure_row(cur_tool: str, past_tools: list[str], mapping: dict[str, str] | None = None) -> dict[str, float]:
    cur = risk_of(cur_tool, mapping)
    past = [risk_of(t, mapping) for t in past_tools]
    p1 = past[-1] if past else NONE
    out = {f"cur_{cur}": 1.0}
    if len(past) >= 1:
        out |= {f"prev_{p1}": 1.0, f"bi_{p1}>{cur}": 1.0}
    if len(past) >= 2:
        out[f"tri_{past[-2]}>{p1}>{cur}"] = 1.0
    out |= {f"past_{c}": 1.0 for c in set(past)}
    out["done_external_send"] = min(past.count("external_send"), 2) / 2
    out["done_payment"] = min(past.count("payment"), 2) / 2
    out["cat_change"] = float(bool(past) and p1 != cur)
    out["cur_risky"] = float(cur in RISKY)
    out["first_risky"] = float(cur in RISKY and not RISKY & set(past))
    return out


class StructureBlock(_RowBlock):
    """Risk-category structure of the current call and of the past calls (see the module docstring)."""

    prefix = "structure__"

    def __init__(self, view: str | None = None):
        self.view = view

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        _check(X, self.view, ("history_text", CURRENT_TOOL.get(self.view, "")))
        mapping = _risk_table()[0]
        rows = []
        for hist, cur in zip(X["history_text"], X[CURRENT_TOOL[self.view]]):
            calls, _, _ = parse_history(hist)
            rows.append(structure_row(cur, [t for t, _ in calls], mapping))
        return pd.DataFrame(rows, columns=STRUCTURE_COLUMNS, index=X.index).fillna(0.0)


class ToolNameBlock(TransformerMixin, BaseEstimator):
    """Indicator of the current tool name; vocabulary learned on the training rows, unseen names -> zeros."""

    def __init__(self, view: str | None = None):
        self.view = view

    def fit(self, X: pd.DataFrame, y=None):
        _check(X, self.view, (CURRENT_TOOL.get(self.view, ""),))
        self.tools_ = sorted(X[CURRENT_TOOL[self.view]].astype(str).unique())
        self.index_ = {t: i for i, t in enumerate(self.tools_)}
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "tools_")
        col = [self.index_.get(t, -1) for t in X[CURRENT_TOOL[self.view]].astype(str)]
        rows = [i for i, c in enumerate(col) if c >= 0]
        cols = [c for c in col if c >= 0]
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(X), len(self.tools_)))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "tools_")
        return np.asarray([f"tool_name__{t}" for t in self.tools_], dtype=object)


class PositionBlock(_RowBlock):
    """CONTROL ONLY: the step index t as indicators (t >= 8 pooled). Needs the `step` column, which the harness gives
    only to a `kind: shortcut` config with `uses_position: true`. Used to measure how much of a block is position."""

    prefix = "position__"
    MAX_T = 8

    def __init__(self, view: str | None = None):
        self.view = view

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if "step" not in X.columns:
            raise ValueError("the position block needs the step index: `uses_position: true` and `kind: shortcut`")
        t = X["step"].astype(int).clip(upper=self.MAX_T).to_numpy()
        return pd.DataFrame((t[:, None] == np.arange(self.MAX_T + 1)).astype(float), index=X.index,
                            columns=[f"t{k}" for k in range(self.MAX_T + 1)])
