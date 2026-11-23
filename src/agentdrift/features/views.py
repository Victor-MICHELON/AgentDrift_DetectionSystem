"""Build the observation x_t for the scanner view and the gate view. Never reads steps after t.

Definitions (CONTEXT 3.3, JOURNAL 19/11/2026 "Vues strictes"), t is the 0-based step index of the
common step schema (`agentdrift.data.schema.STEPS`):

    scanner(t): reads o_{t-1}, the tool response that has just arrived, with the task u, the full steps
                s_0..s_{t-2} and the call a_{t-1} that produced o_{t-1}.
                Never reads thought_t, a_t, o_t or any step > t.
                label = y_scan(t) = label_{t-1} in {injection_point, failed_injection}
    gate(t):    reads the proposed call a_t (tool + arguments), before it is executed, with u and the
                full steps s_0..s_{t-1} (o_{t-1} included). Never reads o_t or any step > t.
                label = y_gate(t) = label_t == hijacked

Granularity: one row per decision point, key (trajectory_id, step).
    gate:    one row per step that carries a tool call (the common schema has exactly one call per step).
    scanner: one row per step t >= 1 whose previous step t-1 is present: at t = 0 there is nothing to
             read yet, so no row (an empty row would only add trivial negatives and a position shortcut).
             The observation of the last step is never scanned: no action follows it.

`thought` (the generated reasoning) exists in two variants (JOURNAL 19/11): `include_thought=False`
(default, the realistic reference) drops it everywhere; `include_thought=True` keeps the past thoughts
in the history and, for the gate only, adds thought_t (written before a_t) as `proposed_call_thought`.

The views are deterministic and fit nothing: anything learned (vocabulary, scaling) belongs to the
feature extractors of B.2+ and is fitted on the training fold only (L2). `build_view` copies the text of the
tables it receives; the text variant is chosen upstream (KAN-60, `pipelines/build_views.py`):

    anon    world-anonymised tables (`read_*(anonymized=True)`, leak L9): the DEFAULT, the only input of models
    masked  anonymised tables + generator-artefact masking (`features/masking.mask_tables`, H3 / L10), applied to
            the TABLES before the views because amounts are numbered in the temporal order of the trajectory
    raw     original text, world identity included: H3 ablation ladder only (KAN-19), never a default

Persisted views carry their variant in the Parquet metadata (`write_view`), and `read_view` refuses a file whose
variant is missing or different: a stale view built on raw text can never be read as the anonymised one.

Output columns (see `output_columns`):
    keys        trajectory_id (str), step (int64)
    target      label (Int8, <NA> when the source step label is missing)
    groups      domain, template_id, world_id, origin_split: grouping / strata / subgroups only, NEVER
                features (L1, L4, L12). Passed through when present in the input.
    eval        gate only: is_first_h (boolean), first hijacked step, for first-alert metrics (CONTEXT 8.2)
    diagnostics history_n_steps (int64), history_truncated (bool): never features (they encode position)
    features    `FEATURE_COLUMNS[view]` (+ `THOUGHT_FEATURE` for the gate with include_thought), all str
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from agentdrift.config import PROCESSED
from agentdrift.data.schema import LABELS

log = logging.getLogger(__name__)

View = Literal["scanner", "gate"]
VIEWS: tuple[View, ...] = ("scanner", "gate")
VIEWS_DIR = PROCESSED / "views"  # written by `python -m agentdrift.pipelines.build_views`

INJECTION_LABELS = frozenset({"injection_point", "failed_injection"})
HIJACK_LABEL = "hijacked"

# ----------------------------------------------------------------------------- input contract

# Common step schema (data/schema.py STEPS) and trajectory table (TRAJECTORIES). `label` may be missing
# (e.g. ATBench has no step labels); `thought` is only required with include_thought=True.
STEP_INPUT_COLUMNS = ("traj_id", "t", "tool", "args_json", "obs")
TRAJ_INPUT_COLUMNS = ("traj_id", "task")

# ----------------------------------------------------------------------------- output contract

KEY_COLUMNS = ("trajectory_id", "step")
TARGET_COLUMN = "label"
GROUP_COLUMNS = ("domain", "template_id", "world_id", "origin_split")
EVAL_COLUMNS: dict[str, tuple[str, ...]] = {"scanner": (), "gate": ("is_first_h",)}
DIAGNOSTIC_COLUMNS = ("history_n_steps", "history_truncated")
FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    # task u | serialised past | call a_{t-1} that produced the content | o_{t-1}
    "scanner": ("task", "history_text", "content_source_tool", "content_source_args", "current_content"),
    # task u | serialised past (o_{t-1} included) | o_{t-1} alone | proposed call a_t
    "gate": ("task", "history_text", "last_content", "proposed_call_tool", "proposed_call_args"),
}
THOUGHT_FEATURE = "proposed_call_thought"  # gate + include_thought only: thought_t, written before a_t

# Fields that must NEVER appear in a view, neither as a column nor as text (L3, L11, L12).
# A.5 (shortcuts) and B.4 (masking) extend this list; `tests/test_views.py` checks that every
# generator metadata column of the schema is either listed here or an explicit non-feature column.
FORBIDDEN_FIELDS: dict[str, str] = {
    # generator / trajectory metadata that reveals the label (L12, JOURNAL 19/11)
    "category": "trajectory class (attacked / failed_attack / ...): is the label",
    "dataset_category": "copy of category",
    "source_category": "attack type of the whole trajectory: label + future; join from trajectories for subgroups",
    "attack_type": "attack scenario name",
    "compliance": "outcome of the attack (full / partial / delayed / resisted): future",
    "complexity": "template metadata of the generator",
    "expected_tools": "template tool pool: oracle only (JOURNAL 19/11)",
    "world_json": "raw world identity (names, companies, addresses): L9, use world_id as a group key only",
    # whole-trajectory information (future, L3 / L7)
    "n_steps": "total trajectory length (attacked trajectories are longer, CONTEXT 6.2)",
    "injection_step": "position of the injection: label",
    "first_h_step": "position of the first hijack: label + future",
    "y_traj": "trajectory label",
    # per-step labels other than the view target
    "y_scan": "precomputed target, replaced by `label`",
    "y_gate": "precomputed target, replaced by `label`",
    "injection_visible": "manual audit of the injection (A, JOURNAL 19/11): subgroup analysis only",
    # step content after the decision point (L11)
    "obs_t": "o_t, the response to a_t: describes the already executed action (gate never sees it)",
    "steps_after_t": "any field of steps > t",
}
# Non-feature columns that ARE in the output, for keys / groups / evaluation (never model inputs).
# `trajectory_id` encodes the category in AgentDrift ("email_attacked_full_0012"): never vectorise it.
NON_FEATURE_COLUMNS = (*KEY_COLUMNS, TARGET_COLUMN, *GROUP_COLUMNS, "is_first_h", *DIAGNOSTIC_COLUMNS)

# ----------------------------------------------------------------------------- serialisation

HISTORY_TRUNCATED = "<history truncated>"
MISSING_STEPS = "<missing steps>"
FIELD_TRUNCATED = " <… truncated …> "
TAG_THOUGHT, TAG_CALL, TAG_RESULT = "<thought>", "<call>", "<result>"


@dataclass(frozen=True)
class ViewConfig:
    """History and length limits, chosen from the EDA / measured length distribution (AgentDrift, 71,024
    steps, <= 11 steps per trajectory): one serialised step is 267 chars on average (p99 559), and the
    longest `history_text` is 2,078 chars without thought / 2,738 with thought (p99 1,245 / 1,867). The
    default 4,000 therefore keeps every AgentDrift history intact (no truncation, no marker that would encode
    the position) while bounding external sets with long tool outputs (ATBench observations: median 685,
    max 51,719 chars).

    max_history_steps:  keep at most the N most recent past steps (None = all).
    max_history_chars:  then drop the oldest steps until `history_text` fits (left truncation, the most
                        recent step is cut on its left if it is alone and still too long); a
                        `HISTORY_TRUNCATED` marker is prepended. None = no limit.
    max_field_chars:    cap of each thought / args / result inside the history (middle cut with
                        `FIELD_TRUNCATED`); AgentDrift max field = 1,107 chars, so never hit there.
    max_content_chars:  same cap for the current content, last content and proposed call (None = keep all:
                        the decision input itself is never cut by default).
    """

    max_history_steps: int | None = None
    max_history_chars: int | None = 4_000
    max_field_chars: int | None = 2_000
    max_content_chars: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_history_steps", "max_history_chars", "max_field_chars", "max_content_chars"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be None or >= 1, got {value}")


def output_columns(view: View, include_thought: bool = False, groups: Iterable[str] = GROUP_COLUMNS) -> list[str]:
    """Exact column order of a view (groups restricted to those present in the input)."""
    _check_view(view)
    feats = list(FEATURE_COLUMNS[view]) + ([THOUGHT_FEATURE] if include_thought and view == "gate" else [])
    return [*KEY_COLUMNS, TARGET_COLUMN, *groups, *EVAL_COLUMNS[view], *DIAGNOSTIC_COLUMNS, *feats]


def feature_columns(view: View, include_thought: bool = False) -> list[str]:
    """The only columns B.2+ may turn into model inputs."""
    _check_view(view)
    return list(FEATURE_COLUMNS[view]) + ([THOUGHT_FEATURE] if include_thought and view == "gate" else [])


def _check_view(view: str) -> None:
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) if not isinstance(value, (list, dict, tuple)) else False
    except (TypeError, ValueError):
        return False


def to_text(value: Any, field: str, stats: Counter) -> str:
    """Deterministic text of one field. None / NaN -> "" ; dict / list -> canonical JSON ; other -> str."""
    if _is_missing(value):
        stats[f"null_{field}"] += 1
        return ""
    if isinstance(value, str):
        if not value.strip():
            stats[f"empty_{field}"] += 1
        return value
    if isinstance(value, (dict, list, tuple)):
        stats[f"non_text_{field}"] += 1
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    stats[f"non_text_{field}"] += 1
    return str(value)


def args_to_text(value: Any, stats: Counter) -> str:
    """Canonical JSON of the call arguments (same form as `data.load.canonical_json`); a non-JSON string is
    kept as is and counted."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            stats["args_not_json"] += 1
            return value
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return to_text(value, "args", stats)


def cut_middle(text: str, limit: int | None) -> tuple[str, bool]:
    """Keep the head and the tail of `text` within `limit` chars (marker included)."""
    if limit is None or len(text) <= limit:
        return text, False
    keep = max(limit - len(FIELD_TRUNCATED), 2)
    head = keep - keep // 2
    return text[:head] + FIELD_TRUNCATED + text[len(text) - keep // 2:], True


def _one_line(text: str) -> str:
    """History is line-tagged: escape newlines so that each tag stays on one line."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


# ----------------------------------------------------------------------------- per-step blocks


@dataclass(frozen=True)
class _Step:
    t: int
    thought: str
    tool: str
    args: str
    obs: str
    label: str | None


def _call_lines(s: _Step, include_thought: bool, cfg: ViewConfig, stats: Counter) -> list[str]:
    lines = []
    if include_thought:
        text, cut = cut_middle(s.thought, cfg.max_field_chars)
        stats["history_field_truncated"] += cut
        lines.append(f"{TAG_THOUGHT} {_one_line(text)}")
    if s.tool:
        args, cut = cut_middle(s.args, cfg.max_field_chars)
        stats["history_field_truncated"] += cut
        lines.append(f"{TAG_CALL} {_one_line(s.tool)} {_one_line(args)}")
    return lines


def _result_line(s: _Step, cfg: ViewConfig, stats: Counter) -> str:
    obs, cut = cut_middle(s.obs, cfg.max_field_chars)
    stats["history_field_truncated"] += cut
    return f"{TAG_RESULT} {_one_line(obs)}"


def serialize_history(blocks: list[str | None], cfg: ViewConfig) -> tuple[str, int, bool]:
    """Join past step blocks (oldest first; None = missing-steps marker) into `history_text`.

    Returns (text, number of steps kept, truncated). Truncation is from the left: the most recent steps
    are kept and `HISTORY_TRUNCATED` is prepended.
    """
    items = [b if b is not None else MISSING_STEPS for b in blocks]
    is_step = [b is not None for b in blocks]
    truncated = False
    if cfg.max_history_steps is not None and sum(is_step) > cfg.max_history_steps:
        kept, start = 0, len(items)
        while start > 0 and kept < cfg.max_history_steps:
            start -= 1
            kept += is_step[start]
        items, is_step, truncated = items[start:], is_step[start:], True
    if cfg.max_history_chars is not None:
        while len(items) > 1 and len("\n".join(items)) > cfg.max_history_chars:
            items, is_step, truncated = items[1:], is_step[1:], True
        if items and len(items[0]) > cfg.max_history_chars:  # one block alone is still too long
            items, truncated = [items[0][-cfg.max_history_chars:]], True
    text = "\n".join(items)
    if truncated:
        text = HISTORY_TRUNCATED + ("\n" + text if text else "")
    return text, sum(is_step), truncated


# ----------------------------------------------------------------------------- input preparation


def _prepare(steps: pd.DataFrame, trajectories: pd.DataFrame, include_thought: bool, stats: Counter) -> pd.DataFrame:
    """Validate the input and return the steps sorted by (traj_id, t) with the task attached."""
    required = list(STEP_INPUT_COLUMNS) + (["thought"] if include_thought else [])
    missing = [c for c in required if c not in steps.columns]
    if missing:
        raise ValueError(f"steps is missing columns {missing} (common step schema, data/schema.py)")
    missing = [c for c in TRAJ_INPUT_COLUMNS if c not in trajectories.columns]
    if missing:
        raise ValueError(f"trajectories is missing columns {missing}")
    if steps["t"].isna().any():
        raise ValueError(f"{int(steps['t'].isna().sum())} steps have no index t")
    if trajectories["traj_id"].duplicated().any():
        raise ValueError("trajectories: duplicated traj_id")

    dup = steps.duplicated(["traj_id", "t"])
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} duplicated (traj_id, t): the common schema has one tool call per step; an "
            "adapter must split parallel calls of one step into consecutive steps (views.py docstring)"
        )
    if "label" in steps.columns:
        bad = set(steps["label"].dropna().unique()) - set(LABELS)
        if bad:
            raise ValueError(f"unknown step labels {sorted(bad)}; expected {LABELS} or missing")

    s = steps.copy()
    s["t"] = s["t"].astype("int64")
    unsorted = s.groupby("traj_id", sort=False)["t"].apply(lambda x: not x.is_monotonic_increasing)
    if unsorted.any():
        stats["trajectories_reordered"] += int(unsorted.sum())
        log.info("%d trajectories were not sorted by t: sorted", int(unsorted.sum()))
    s = s.sort_values(["traj_id", "t"], kind="stable").reset_index(drop=True)

    tasks = trajectories.set_index("traj_id")["task"]
    unknown = ~s["traj_id"].isin(tasks.index)
    if unknown.any():
        n = s.loc[unknown, "traj_id"].nunique()
        raise ValueError(f"{n} trajectories of `steps` are absent from `trajectories`")
    s["task"] = s["traj_id"].map(tasks)
    return s


def _steps_of(g: pd.DataFrame, include_thought: bool, stats: Counter) -> list[_Step]:
    has_label = "label" in g.columns
    out = []
    for r in g.itertuples(index=False):
        label = getattr(r, "label") if has_label else None
        if _is_missing(label):
            stats["missing_label"] += 1
            label = None
        tool = to_text(r.tool, "tool", stats).strip()
        out.append(
            _Step(
                t=int(r.t),
                thought=to_text(r.thought, "thought", stats) if include_thought else "",
                tool=tool,
                args=args_to_text(r.args_json, stats) if tool else "",
                obs=to_text(r.obs, "obs", stats),
                label=label,
            )
        )
    return out


# ----------------------------------------------------------------------------- builders


def _rows_of_trajectory(
    view: View, steps: list[_Step], include_thought: bool, cfg: ViewConfig, stats: Counter
) -> list[dict]:
    """All rows of one trajectory. Row i only reads steps[:i] (gate also reads the call of steps[i])."""
    # blocks[i] = what step i contributes to the history once it is complete; gap markers (None) are
    # inserted before a step whose t does not follow the previous one.
    full, call_only, gap_before = [], [], []
    for i, s in enumerate(steps):
        expected = steps[i - 1].t + 1 if i else 0
        gap_before.append(s.t != expected)
        call = _call_lines(s, include_thought, cfg, stats)
        call_only.append("\n".join(call))
        full.append("\n".join([*call, _result_line(s, cfg, stats)]))
    if any(gap_before):
        stats["trajectories_with_gaps"] += 1

    def past(i: int, last_call_only: bool) -> list[str | None]:
        """Blocks of steps[:i] (gap markers included); the last one call-only if asked."""
        out: list[str | None] = []
        for k in range(i):
            if gap_before[k]:
                out.append(None)
            out.append(call_only[k] if (last_call_only and k == i - 1) else full[k])
        if gap_before[i]:  # steps missing between steps[i-1] and steps[i]: they are in the past
            out.append(None)
        return out

    rows, seen_h = [], False
    for i, s in enumerate(steps):
        if view == "gate":
            first_h = None if s.label is None else (s.label == HIJACK_LABEL and not seen_h)
            seen_h = seen_h or s.label == HIJACK_LABEL
            if not s.tool:
                stats["steps_without_call"] += 1
                continue
            text, n_kept, trunc = serialize_history(past(i, last_call_only=False), cfg)
            prev_obs = steps[i - 1].obs if i and not gap_before[i] else ""
            args, cut = cut_middle(s.args, cfg.max_content_chars)
            last, cut2 = cut_middle(prev_obs, cfg.max_content_chars)
            stats["content_truncated"] += cut + cut2
            row = {
                "label": None if s.label is None else int(s.label == HIJACK_LABEL),
                "is_first_h": first_h,
                "last_content": last,
                "proposed_call_tool": s.tool,
                "proposed_call_args": args,
            }
            if include_thought:
                thought, cut = cut_middle(s.thought, cfg.max_content_chars)
                stats["content_truncated"] += cut
                row[THOUGHT_FEATURE] = thought
        else:
            if i == 0 and s.t == 0:
                stats["scanner_first_step_skipped"] += 1
                continue
            if i == 0 or gap_before[i]:
                stats["scanner_previous_step_missing"] += 1
                continue
            prev = steps[i - 1]
            # the scanner at t stops before thought_t: the past ends with the call a_{t-1}
            text, n_kept, trunc = serialize_history(past(i, last_call_only=True), cfg)
            content, cut = cut_middle(prev.obs, cfg.max_content_chars)
            stats["content_truncated"] += cut
            row = {
                "label": None if prev.label is None else int(prev.label in INJECTION_LABELS),
                "content_source_tool": prev.tool,
                "content_source_args": prev.args,
                "current_content": content,
            }
        stats["history_truncated"] += trunc
        row.update({"step": s.t, "history_text": text, "history_n_steps": n_kept, "history_truncated": trunc})
        rows.append(row)
    if view == "scanner" and steps and steps[-1].label in INJECTION_LABELS:
        stats["unscanned_injection_in_last_obs"] += 1
    return rows


def build_view(
    steps: pd.DataFrame,
    trajectories: pd.DataFrame,
    view: View,
    include_thought: bool = False,
    config: ViewConfig | None = None,
) -> pd.DataFrame:
    """Build one view from tables in the common schema (`data/schema.py`: STEPS + TRAJECTORIES).

    `steps` needs traj_id, t, tool, args_json, obs (+ thought if include_thought; label and the group
    columns are optional); `trajectories` needs traj_id, task. Nothing is fitted. Edge cases are counted in
    `df.attrs["edge_cases"]` and logged (warnings for data anomalies).
    """
    _check_view(view)
    cfg = config or ViewConfig()
    stats: Counter = Counter()
    s = _prepare(steps, trajectories, include_thought, stats)
    groups = [c for c in GROUP_COLUMNS if c in s.columns]

    rows: list[dict] = []
    for traj_id, g in s.groupby("traj_id", sort=True):
        meta = {"trajectory_id": traj_id, "task": to_text(g["task"].iloc[0], "task", stats)}
        meta.update({c: g[c].iloc[0] for c in groups})
        for row in _rows_of_trajectory(view, _steps_of(g, include_thought, stats), include_thought, cfg, stats):
            rows.append({**meta, **row})

    cols = output_columns(view, include_thought, groups)
    df = pd.DataFrame(rows, columns=cols)
    df["trajectory_id"] = df["trajectory_id"].astype("str")
    df["step"] = df["step"].astype("int64")
    df["label"] = df["label"].astype("Int8")
    df["history_n_steps"] = df["history_n_steps"].astype("int64")
    df["history_truncated"] = df["history_truncated"].astype(bool)
    if view == "gate":
        df["is_first_h"] = df["is_first_h"].astype("boolean")
    for c in groups + feature_columns(view, include_thought):
        df[c] = df[c].astype("str")

    df.attrs["edge_cases"] = {k: int(v) for k, v in sorted(stats.items()) if v}
    _log_stats(view, len(df), df.attrs["edge_cases"])
    return df


ANOMALIES = (
    "trajectories_with_gaps", "scanner_previous_step_missing", "steps_without_call", "missing_label",
    "args_not_json", "unscanned_injection_in_last_obs", "trajectories_reordered",
)


def _log_stats(view: str, n_rows: int, edge: dict[str, int]) -> None:
    log.info("%s view: %d rows; edge cases %s", view, n_rows, edge or "none")
    for k in ANOMALIES:
        if edge.get(k):
            log.warning("%s view: %d × %s", view, edge[k], k)


def build_scanner_view(steps: pd.DataFrame, trajectories: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Scanner view: one row per received tool response o_{t-1}, label y_scan(t). See `build_view`."""
    return build_view(steps, trajectories, "scanner", **kwargs)


def build_gate_view(steps: pd.DataFrame, trajectories: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Gate view: one row per proposed call a_t (before execution), label y_gate(t). See `build_view`."""
    return build_view(steps, trajectories, "gate", **kwargs)


def build_views(
    steps: pd.DataFrame, trajectories: pd.DataFrame, include_thought: bool = False, config: ViewConfig | None = None
) -> dict[str, pd.DataFrame]:
    """Both views, same settings."""
    return {v: build_view(steps, trajectories, v, include_thought, config) for v in VIEWS}



# ----------------------------------------------------------------------------- persisted views


TextVariant = Literal["anon", "masked", "raw"]
TEXT_VARIANTS: tuple[TextVariant, ...] = ("anon", "masked", "raw")
DEFAULT_TEXT: TextVariant = "anon"
Dataset = Literal["agentdrift", "atbench"]
METADATA_KEY = b"agentdrift.view"  # Parquet schema metadata: {"view", "include_thought", "text", "dataset"}


class StaleViewError(ValueError):
    """A persisted view whose text variant is not recorded or not the one requested (rebuild it)."""


def _check_text(text: str) -> None:
    if text not in TEXT_VARIANTS:
        raise ValueError(f"text must be one of {TEXT_VARIANTS}, got {text!r}")


def view_path(view: View, include_thought: bool = False, test: bool = False, text: TextVariant = DEFAULT_TEXT,
              dataset: Dataset = "agentdrift") -> Path:
    """data/processed/views/[atbench_]<view>[_thought][_masked|_raw][_test].parquet.

    dev = train + val, test = fixed test set (AgentDrift only); the anonymised variant has no suffix."""
    _check_view(view)
    _check_text(text)
    if dataset not in ("agentdrift", "atbench"):
        raise ValueError(f"unknown dataset {dataset!r}")
    if dataset == "atbench" and test:
        raise ValueError("ATBench has no fixed test split: it is external as a whole")
    name = (f"{'atbench_' if dataset == 'atbench' else ''}{view}{'_thought' if include_thought else ''}"
            f"{'' if text == DEFAULT_TEXT else '_' + text}{'_test' if test else ''}")
    return VIEWS_DIR / f"{name}.parquet"


def _view_metadata(view: str, include_thought: bool, text: str, dataset: str) -> bytes:
    return json.dumps({"view": view, "include_thought": include_thought, "text": text, "dataset": dataset},
                      sort_keys=True).encode()


def write_view(df: pd.DataFrame, view: View, include_thought: bool = False, test: bool = False,
               text: TextVariant = DEFAULT_TEXT, dataset: Dataset = "agentdrift") -> Path:
    """Write a view with its variant recorded in the Parquet metadata (read back and checked by `read_view`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = view_path(view, include_thought, test, text, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    meta = {**(table.schema.metadata or {}), METADATA_KEY: _view_metadata(view, include_thought, text, dataset)}
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def _check_metadata(path: Path, expected: bytes) -> None:
    import pyarrow.parquet as pq

    found = (pq.read_schema(path).metadata or {}).get(METADATA_KEY)
    if found is None:
        raise StaleViewError(f"{path} has no text-variant metadata (built before KAN-60, possibly on raw text): "
                             "rebuild it with `python -m agentdrift.pipelines.build_views`")
    if json.loads(found) != json.loads(expected):
        raise StaleViewError(f"{path} holds {json.loads(found)}, expected {json.loads(expected)}: rebuild the views")


def read_view(view: View, include_thought: bool = False, include_test: bool = False,
              text: TextVariant = DEFAULT_TEXT, dataset: Dataset = "agentdrift") -> pd.DataFrame:
    """A view written by `build_views`; default = anonymised text without thought (the only model input).

    The fixed test set is excluded unless include_test=True (L5). The variant recorded in each file must match
    (`StaleViewError` otherwise)."""
    paths = [view_path(view, include_thought, text=text, dataset=dataset)]
    if include_test:
        paths.append(view_path(view, include_thought, test=True, text=text, dataset=dataset))
    for path in paths:
        if not path.exists():
            flags = (" --include-test" if path.name.endswith("_test.parquet") else "") + \
                    (" --raw" if text == "raw" and dataset == "agentdrift" else "") + \
                    (" --atbench" if dataset == "atbench" else "")
            raise FileNotFoundError(f"{path} missing: run `python -m agentdrift.pipelines.build_views{flags}` first")
        _check_metadata(path, _view_metadata(view, include_thought, text, dataset))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
