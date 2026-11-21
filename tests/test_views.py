"""Scanner / gate views (features/views.py): causality, labels, contract, edge cases.

Toy tests run everywhere; corpus tests are skipped if data/interim/ has not been built
(`python -m agentdrift.data.load`).
"""

import json
import time

import pandas as pd
import pytest

from agentdrift.config import INTERIM
from agentdrift.data.load import read_steps, read_trajectories
from agentdrift.data.schema import METADATA_COLUMNS
from agentdrift.features.views import (
    FEATURE_COLUMNS,
    FIELD_TRUNCATED,
    FORBIDDEN_FIELDS,
    HISTORY_TRUNCATED,
    MISSING_STEPS,
    NON_FEATURE_COLUMNS,
    THOUGHT_FEATURE,
    VIEWS,
    ViewConfig,
    build_gate_view,
    build_scanner_view,
    build_view,
    build_views,
    feature_columns,
    output_columns,
    serialize_history,
)

B, I, H, F = "benign", "injection_point", "hijacked", "failed_injection"
GROUPS = {"domain": "email", "template_id": "email_000", "world_id": "w" * 12, "origin_split": "train"}


def make(trajs: dict[str, list[dict]], task: str = "Send the report to Bob") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steps / trajectories tables in the common schema from {traj_id: [step dicts]}.
    Missing step fields get deterministic defaults derived from (traj_id, t)."""
    rows = []
    for tid, steps in trajs.items():
        for k, st in enumerate(steps):
            t = st.get("t", k)
            rows.append(
                {
                    "traj_id": tid, "t": t, **GROUPS, "source_category": "benign",
                    "thought": st.get("thought", f"thought {tid} {t}"),
                    "tool": st.get("tool", f"tool_{t}"),
                    "args_json": st.get("args_json", json.dumps({"x": t})),
                    "obs": st.get("obs", f"obs {tid} {t}"),
                    "label": st.get("label", B),
                }
            )
    steps = pd.DataFrame(rows)
    traj = pd.DataFrame({"traj_id": list(trajs), "task": task})
    return steps, traj


def labels(seq: list[str]) -> list[dict]:
    return [{"label": lab} for lab in seq]


# Hand-built fixture: one trajectory per AgentDrift category (labels as in the dataset README).
TOY = {
    "a_full": labels([B, I, H, H, B]),
    "a_partial": labels([B, B, I, H]),
    "a_delayed": labels([B, I, B, B, H]),
    "failed": labels([B, F, B]),
    "benign": labels([B, B, B]),
}


@pytest.fixture
def toy():
    return make(TOY)


def row(view: pd.DataFrame, tid: str, t: int) -> pd.Series:
    r = view[(view["trajectory_id"] == tid) & (view["step"] == t)]
    assert len(r) == 1, (tid, t, len(r))
    return r.iloc[0]


def text_of(r: pd.Series) -> str:
    return "\n".join(str(v) for v in r.to_numpy())


# ----------------------------------------------------------------------------- 1. causality (L3, L11)


def _perturb_after(steps: pd.DataFrame, tid: str, t: int, mode: str, include_current: tuple[str, ...]) -> pd.DataFrame:
    """Change (mode='modify') or delete (mode='delete') every step > t of `tid`, and also the listed fields of
    step t itself (fields the view at t must not read)."""
    s = steps.copy()
    mine = s["traj_id"] == tid
    after = mine & (s["t"] > t)
    if mode == "delete":
        s = s[~after]
        mine = s["traj_id"] == tid
    else:
        for c in ("thought", "tool", "args_json", "obs"):
            s.loc[after, c] = "FUTURE_" + s.loc[after, c].astype(str)
        s.loc[after, "label"] = s.loc[after, "label"].map({B: H, H: B, I: B, F: B})
    cur = mine & (s["t"] == t)
    for c in include_current:
        if c == "label":
            s.loc[cur, c] = s.loc[cur, c].map({B: H, H: B, I: H, F: H})
        elif c == "args_json":
            s.loc[cur, c] = '{"changed":"CURRENT"}'
        else:
            s.loc[cur, c] = "CURRENT_CHANGED"
    return s


# Fields of step t each view must ignore (CONTEXT 3.3: scanner never reads a_t / o_t, gate never reads o_t).
NOT_READ_AT_T = {"scanner": ("thought", "tool", "args_json", "obs", "label"), "gate": ("obs",)}


def _assert_causal(steps, traj, tid, view, include_thought):
    ref = build_view(steps, traj, view, include_thought)
    ts = sorted(steps.loc[steps["traj_id"] == tid, "t"])
    for t in ts:
        expected = ref[(ref["trajectory_id"] == tid) & (ref["step"] <= t)].reset_index(drop=True)
        for mode in ("modify", "delete"):
            pert = _perturb_after(steps, tid, t, mode, NOT_READ_AT_T[view])
            got = build_view(pert, traj, view, include_thought)
            got = got[(got["trajectory_id"] == tid) & (got["step"] <= t)].reset_index(drop=True)
            pd.testing.assert_frame_equal(got, expected, obj=f"{view} {tid} t={t} {mode}")


@pytest.mark.parametrize("include_thought", [False, True])
@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("tid", list(TOY))
def test_causality_toy(toy, tid, view, include_thought):
    steps, traj = toy
    _assert_causal(steps, traj, tid, view, include_thought)


def test_gate_never_contains_the_result_of_its_call(toy):
    """Sentinel in every o_k: the gate row at t contains no o_k with k >= t; the scanner row at t contains
    o_{t-1} as content and no o_k with k >= t."""
    steps, traj = toy
    steps["obs"] = "OBS_SENTINEL_" + steps["traj_id"] + "_" + steps["t"].astype(str) + "_END"
    for include_thought in (False, True):
        views = build_views(steps, traj, include_thought)
        for view, df in views.items():
            for _, r in df.iterrows():
                text = text_of(r)
                n = int(steps.loc[steps["traj_id"] == r["trajectory_id"], "t"].max())
                for k in range(r["step"], n + 1):
                    assert f"OBS_SENTINEL_{r['trajectory_id']}_{k}_END" not in text, (view, r["trajectory_id"], k)
                if view == "scanner":
                    assert r["current_content"] == f"OBS_SENTINEL_{r['trajectory_id']}_{r['step'] - 1}_END"


def test_scanner_never_reads_the_current_thought(toy):
    steps, traj = toy
    df = build_scanner_view(steps, traj, include_thought=True)
    for _, r in df.iterrows():
        assert f"thought {r['trajectory_id']} {r['step']}" not in text_of(r)
        assert f"thought {r['trajectory_id']} {r['step'] - 1}" in r["history_text"]


# ----------------------------------------------------------------------------- 3. forbidden fields


def test_forbidden_fields_never_in_output(toy):
    steps, traj = toy
    for view in VIEWS:
        for th in (False, True):
            cols = set(build_view(steps, traj, view, th).columns)
            assert not cols & set(FORBIDDEN_FIELDS), cols & set(FORBIDDEN_FIELDS)
            assert not set(feature_columns(view, th)) & set(NON_FEATURE_COLUMNS)


def test_every_schema_metadata_column_is_classified():
    """A new metadata column added to the schema must be explicitly forbidden or declared non-feature."""
    renamed = {"traj_id", "t", "label", "y_scan", "y_gate"}  # become keys / target of the views
    unclassified = set(METADATA_COLUMNS) - set(FORBIDDEN_FIELDS) - set(NON_FEATURE_COLUMNS) - renamed
    assert not unclassified, unclassified


def test_no_forbidden_text_reaches_the_features():
    """Forbidden values placed in the input (extra columns, label-revealing ids) never reach a feature."""
    content = [{"label": lab, "thought": f"th {k}", "obs": f"o {k}"} for k, lab in enumerate([B, I, H, B])]
    steps, traj = make({"email_attacked_full_SENTINEL": content})
    sentinels = {}
    for c in FORBIDDEN_FIELDS:
        sentinels[c] = f"FORBIDDEN_{c.upper()}_VALUE"
        steps[c] = sentinels[c]
        traj[c] = sentinels[c]
    for view in VIEWS:
        for th in (False, True):
            df = build_view(steps, traj, view, th)
            feats = "\n".join(df[feature_columns(view, th)].astype(str).to_numpy().ravel())
            for v in sentinels.values():
                assert v not in feats
            assert "SENTINEL" not in feats  # trajectory id encodes the category: never in a feature


def test_thought_only_in_the_thought_variant(toy):
    steps, traj = toy
    for view in VIEWS:
        df = build_view(steps, traj, view, include_thought=False)
        assert "thought " not in "\n".join(df[feature_columns(view)].to_numpy().ravel())
        assert THOUGHT_FEATURE not in df.columns
    gate = build_gate_view(steps, traj, include_thought=True)
    assert row(gate, "a_full", 2)[THOUGHT_FEATURE] == "thought a_full 2"


# ----------------------------------------------------------------------------- 4. labels


def test_labels_follow_context_definitions(toy):
    """y_scan(t) = label_{t-1} in {I, F} (rows t >= 1) ; y_gate(t) = label_t == H (CONTEXT 3.3)."""
    steps, traj = toy
    scan = build_scanner_view(steps, traj).set_index(["trajectory_id", "step"])["label"]
    gate = build_gate_view(steps, traj).set_index(["trajectory_id", "step"])["label"]
    expected_scan = {
        "a_full": [0, 1, 0, 0], "a_partial": [0, 0, 1], "a_delayed": [0, 1, 0, 0], "failed": [0, 1], "benign": [0, 0],
    }
    expected_gate = {
        "a_full": [0, 0, 1, 1, 0], "a_partial": [0, 0, 0, 1], "a_delayed": [0, 0, 0, 0, 1], "failed": [0, 0, 0],
        "benign": [0, 0, 0],
    }
    for tid, exp in expected_scan.items():
        assert scan.loc[tid].index.tolist() == list(range(1, len(exp) + 1))
        assert scan.loc[tid].tolist() == exp, tid
    for tid, exp in expected_gate.items():
        assert gate.loc[tid].tolist() == exp, tid
    first_h = build_gate_view(steps, traj).set_index(["trajectory_id", "step"])["is_first_h"]
    assert first_h.loc["a_full"].tolist() == [False, False, True, False, False]
    assert first_h.loc["a_delayed"].tolist() == [False] * 4 + [True]


# ----------------------------------------------------------------------------- 5-7. determinism, cardinality, contract


def test_deterministic_and_input_order_free(toy):
    steps, traj = toy
    shuffled = steps.sample(frac=1, random_state=0)
    for view in VIEWS:
        a = build_view(steps, traj, view)
        pd.testing.assert_frame_equal(a, build_view(steps, traj, view))
        pd.testing.assert_frame_equal(a, build_view(shuffled, traj.iloc[::-1], view))


def _expected_counts(steps: pd.DataFrame) -> dict[str, int]:
    """Independent count of eligible decision points."""
    has_call = steps["tool"].fillna("").astype(str).str.strip().ne("")
    prev = steps[["traj_id", "t"]].assign(t=steps["t"] + 1)
    has_prev = steps[["traj_id", "t"]].merge(prev, how="left", indicator=True)["_merge"].eq("both")
    return {"gate": int(has_call.sum()), "scanner": int(has_prev.sum())}


def test_cardinality(toy):
    steps, traj = toy
    exp = _expected_counts(steps)
    assert exp == {"gate": 20, "scanner": 15}
    for view in VIEWS:
        assert len(build_view(steps, traj, view)) == exp[view]


def test_contract(toy):
    steps, traj = toy
    for view in VIEWS:
        for th in (False, True):
            df = build_view(steps, traj, view, th)
            assert list(df.columns) == output_columns(view, th)
            assert not df.duplicated(["trajectory_id", "step"]).any()
            assert df["step"].dtype == "int64" and df["label"].dtype == "Int8"
            assert df["history_truncated"].dtype == bool and df["history_n_steps"].dtype == "int64"
            for c in ["trajectory_id", *feature_columns(view, th)]:
                assert pd.api.types.is_string_dtype(df[c]), c
            assert set(feature_columns(view, th)) >= set(FEATURE_COLUMNS[view])
    assert build_gate_view(steps, traj)["is_first_h"].dtype == "boolean"


def test_history_is_readable_and_tagged(toy):
    steps, traj = toy
    gate, scan = build_gate_view(steps, traj), build_scanner_view(steps, traj)
    assert row(gate, "a_full", 0)["history_text"] == ""
    assert row(gate, "a_full", 2)["history_text"] == (
        '<call> tool_0 {"x":0}\n<result> obs a_full 0\n<call> tool_1 {"x":1}\n<result> obs a_full 1'
    )
    assert row(gate, "a_full", 2)["last_content"] == "obs a_full 1"
    r = row(scan, "a_full", 2)  # the scanner stops after the call a_{t-1}: its result is the content
    assert r["history_text"] == '<call> tool_0 {"x":0}\n<result> obs a_full 0\n<call> tool_1 {"x":1}'
    assert (r["content_source_tool"], r["content_source_args"], r["current_content"]) == (
        "tool_1", '{"x":1}', "obs a_full 1")


# ----------------------------------------------------------------------------- 8. edge cases


def test_first_step(toy):
    steps, traj = toy
    gate, scan = build_gate_view(steps, traj), build_scanner_view(steps, traj)
    first = gate[gate["step"] == 0]
    assert len(first) == len(TOY) and first["history_text"].eq("").all() and first["last_content"].eq("").all()
    assert (scan["step"] >= 1).all()
    assert scan.attrs["edge_cases"]["scanner_first_step_skipped"] == len(TOY)


def test_step_without_tool_call_has_no_gate_row_but_stays_in_history():
    steps, traj = make({"x": [{}, {"tool": "", "obs": "user message"}, {"tool": None}, {}]})
    gate = build_gate_view(steps, traj)
    assert gate["step"].tolist() == [0, 3]
    assert gate.attrs["edge_cases"]["steps_without_call"] == 2
    assert "<result> user message" in row(gate, "x", 3)["history_text"]
    scan = build_scanner_view(steps, traj)
    assert row(scan, "x", 2)["content_source_tool"] == "" and row(scan, "x", 2)["current_content"] == "user message"


def test_several_calls_in_one_step_are_rejected():
    steps, traj = make({"x": [{"t": 0}, {"t": 1, "tool": "a"}, {"t": 1, "tool": "b"}]})
    for view in VIEWS:
        with pytest.raises(ValueError, match="one tool call per step"):
            build_view(steps, traj, view)


def test_null_empty_and_json_contents():
    steps, traj = make({"x": [{"obs": None}, {"obs": ""}, {}, {}]})
    steps["obs"] = steps["obs"].astype(object)
    steps.at[2, "obs"] = {"b": 2, "a": [1, "é"]}
    steps.at[3, "args_json"] = "not json"
    scan = build_scanner_view(steps, traj)
    assert row(scan, "x", 1)["current_content"] == ""
    assert row(scan, "x", 3)["current_content"] == '{"a":[1,"é"],"b":2}'
    edge = scan.attrs["edge_cases"]
    assert edge["null_obs"] == 1 and edge["empty_obs"] == 1 and edge["non_text_obs"] == 1 and edge["args_not_json"] == 1
    gate = build_gate_view(steps, traj)
    assert row(gate, "x", 3)["proposed_call_args"] == "not json"


def test_multiline_content_keeps_one_line_per_tag():
    steps, traj = make({"x": [{"obs": "line1\nline2\r\nline3"}, {}]})
    hist = row(build_gate_view(steps, traj), "x", 1)["history_text"]
    assert hist.splitlines() == ['<call> tool_0 {"x":0}', "<result> line1\\nline2\\nline3"]
    assert row(build_scanner_view(steps, traj), "x", 1)["current_content"] == "line1\nline2\r\nline3"


def test_very_long_contents_are_capped_with_markers():
    long_obs = "A" * 3000 + "MIDDLE" + "Z" * 3000
    steps, traj = make({"x": [{"obs": long_obs}, {}, {}]})
    gate = build_gate_view(steps, traj, config=ViewConfig(max_field_chars=200, max_content_chars=100))
    r = row(gate, "x", 1)
    assert FIELD_TRUNCATED in r["history_text"] and "MIDDLE" not in r["history_text"]
    assert len(r["last_content"]) == 100 and r["last_content"].startswith("A") and r["last_content"].endswith("Z")
    assert gate.attrs["edge_cases"]["content_truncated"] >= 1
    assert gate.attrs["edge_cases"]["history_field_truncated"] >= 1
    # default config: the decision input is never cut
    assert row(build_scanner_view(steps, traj), "x", 1)["current_content"] == long_obs


def test_history_left_truncation_keeps_the_most_recent_steps():
    steps, traj = make({"x": [{} for _ in range(8)]})
    gate = build_gate_view(steps, traj, config=ViewConfig(max_history_steps=2))
    r = row(gate, "x", 7)
    assert r["history_text"].startswith(HISTORY_TRUNCATED)
    assert "obs x 6" in r["history_text"] and "obs x 5" in r["history_text"] and "obs x 4" not in r["history_text"]
    assert r["history_n_steps"] == 2 and r["history_truncated"]
    assert not row(gate, "x", 2)["history_truncated"]

    by_chars = build_gate_view(steps, traj, config=ViewConfig(max_history_chars=60))
    r = row(by_chars, "x", 7)
    assert r["history_text"].startswith(HISTORY_TRUNCATED) and r["history_text"].endswith("<result> obs x 6")
    assert len(r["history_text"]) <= 60 + len(HISTORY_TRUNCATED) + 1


def test_serialize_history_single_block_too_long():
    text, n, trunc = serialize_history(["x" * 50 + "END"], ViewConfig(max_history_chars=10))
    assert trunc and n == 1 and text == HISTORY_TRUNCATED + "\n" + "xxxxxxxEND"


def test_unsorted_input_is_sorted_and_counted(toy):
    steps, traj = toy
    rev = steps.iloc[::-1]
    df = build_gate_view(rev, traj)
    pd.testing.assert_frame_equal(df, build_gate_view(steps, traj))
    assert df.attrs["edge_cases"]["trajectories_reordered"] == len(TOY)


def test_truncated_trajectory_with_missing_steps():
    """Steps 2-3 missing (and a trajectory starting at t=1): marker in the history, scanner rows that would
    need a missing o_{t-1} are skipped, counted, never silent."""
    steps, traj = make({"gap": [{"t": 0}, {"t": 1}, {"t": 4}, {"t": 5}], "late": [{"t": 1}, {"t": 2}]})
    gate, scan = build_gate_view(steps, traj), build_scanner_view(steps, traj)
    assert MISSING_STEPS in row(gate, "gap", 4)["history_text"]
    assert row(gate, "gap", 4)["last_content"] == ""
    assert row(gate, "late", 1)["history_text"] == MISSING_STEPS
    assert scan.loc[scan["trajectory_id"] == "gap", "step"].tolist() == [1, 5]
    assert scan.loc[scan["trajectory_id"] == "late", "step"].tolist() == [2]
    assert scan.attrs["edge_cases"]["scanner_previous_step_missing"] == 2
    assert gate.attrs["edge_cases"]["trajectories_with_gaps"] == 2


def test_missing_labels_give_na_targets():
    steps, traj = make({"x": labels([B, I, H])})
    steps.loc[1, "label"] = None
    scan, gate = build_scanner_view(steps, traj), build_gate_view(steps, traj)
    assert row(scan, "x", 2)["label"] is pd.NA and row(gate, "x", 1)["label"] is pd.NA
    assert row(gate, "x", 2)["label"] == 1
    assert scan.attrs["edge_cases"]["missing_label"] == 1
    no_label = build_gate_view(steps.drop(columns="label"), traj)  # e.g. ATBench: no step labels at all
    assert no_label["label"].isna().all() and no_label["is_first_h"].isna().all()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s, t: (s.assign(label="hijack"), t), "unknown step labels"),
        (lambda s, t: (s.drop(columns="obs"), t), "missing columns"),
        (lambda s, t: (s, t.iloc[1:]), "absent from `trajectories`"),
        (lambda s, t: (s.assign(t=s["t"].astype(float).where(s.index != 0)), t), "no index t"),
    ],
)
def test_invalid_inputs_raise(toy, mutate, match):
    steps, traj = mutate(*toy)
    with pytest.raises(ValueError, match=match):
        build_gate_view(steps, traj)


def test_invalid_view_and_config():
    with pytest.raises(ValueError, match="view"):
        output_columns("planner")
    with pytest.raises(ValueError, match="max_history_chars"):
        ViewConfig(max_history_chars=0)


def test_thought_required_only_for_thought_variant(toy):
    steps, traj = toy
    build_gate_view(steps.drop(columns="thought"), traj)
    with pytest.raises(ValueError, match="thought"):
        build_gate_view(steps.drop(columns="thought"), traj, include_thought=True)


# ----------------------------------------------------------------------------- corpus (needs data/interim)

needs_interim = pytest.mark.skipif(
    not (INTERIM / "steps.parquet").exists(), reason="run `python -m agentdrift.data.load` first"
)


@pytest.fixture(scope="module")
def corpus():
    return read_steps(include_test=True), read_trajectories(include_test=True)


def _sample_ids(traj: pd.DataFrame) -> list[str]:
    """One train trajectory per source category, the longest ones first (most steps to perturb)."""
    t = traj[traj["origin_split"] == "train"].sort_values(["n_steps", "traj_id"], ascending=[False, True])
    return t.groupby("source_category").head(1)["traj_id"].tolist()


@needs_interim
@pytest.mark.parametrize("view", VIEWS)
def test_causality_real_trajectories(corpus, view):
    steps, traj = corpus
    for tid in _sample_ids(traj):
        s = steps[steps["traj_id"] == tid]
        tr = traj.loc[traj["traj_id"] == tid, ["traj_id", "task"]]
        _assert_causal(s, tr, tid, view, include_thought=True)
        _assert_causal(s, tr, tid, view, include_thought=False)


@needs_interim
def test_full_corpus_smoke(corpus):
    """Both views on the 12,536 trajectories: counts, labels and first H equal the schema targets."""
    steps, traj = corpus
    start = time.perf_counter()
    views = build_views(steps, traj)
    elapsed = time.perf_counter() - start
    assert elapsed < 120, f"views took {elapsed:.0f}s"
    exp = _expected_counts(steps)
    scan, gate = views["scanner"], views["gate"]
    assert len(scan) == exp["scanner"] == 71_024 - 12_536 and len(gate) == exp["gate"] == 71_024
    assert int(scan["label"].sum()) == int(steps["y_scan"].sum()) == 7_036
    assert int(gate["label"].sum()) == int(steps["y_gate"].sum()) == 12_349
    ref = steps.set_index(["traj_id", "t"])
    s_idx = pd.MultiIndex.from_frame(scan[["trajectory_id", "step"]])
    g_idx = pd.MultiIndex.from_frame(gate[["trajectory_id", "step"]])
    assert (scan["label"].to_numpy() == ref.loc[s_idx, "y_scan"].to_numpy()).all()
    assert (gate["label"].to_numpy() == ref.loc[g_idx, "y_gate"].to_numpy()).all()
    assert (gate["is_first_h"].to_numpy() == ref.loc[g_idx, "is_first_h"].to_numpy()).all()
    assert not scan["history_truncated"].any() and not gate["history_truncated"].any()  # default limits
    for df in views.values():
        assert not set(df.columns) & set(FORBIDDEN_FIELDS)


def test_read_view_roundtrip_keeps_types_and_hides_test(toy, tmp_path, monkeypatch):
    import agentdrift.features.views as V

    monkeypatch.setattr(V, "VIEWS_DIR", tmp_path)
    steps, traj = toy
    gate = build_gate_view(steps, traj)
    gate.to_parquet(V.view_path("gate"), index=False)
    pd.testing.assert_frame_equal(V.read_view("gate"), gate, check_like=False)
    assert V.view_path("scanner", include_thought=True, test=True).name == "scanner_thought_test.parquet"
    with pytest.raises(FileNotFoundError, match="--include-test"):
        V.read_view("gate", include_test=True)
