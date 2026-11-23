"""Leak L12: no metadata column added by the data phase (Lot A) ever reaches the model inputs of the views.

tests/test_views.py (Lot B) already checks the generator metadata of the schema. This file covers the columns that
Lot A added since: fold assignments, manual annotations and the label-revealing fields of ATBench. Each one is
attached to the input tables with a recognisable value; neither the column nor the value may appear in a view.

Skipped while `features/views.py` is not on the branch (it comes from main, PR #2).
"""

import pandas as pd
import pytest

views = pytest.importorskip("agentdrift.features.views")
if not hasattr(views, "build_view"):
    pytest.skip("features/views.py not merged yet (main, PR #2)", allow_module_level=True)

from agentdrift.data.external import FORBIDDEN_TRAJECTORY_FIELDS, STEPS_PATH, read_atbench  # noqa: E402

LOT_A_STEP_METADATA = {
    "fold": "outer fold of the trajectory (splits/grouped.py)",
    "world_fold": "control fold by world",
    "injection_visible": "manual annotation of the injection (configs/annotations)",
    "turn": "index of the user turn (ATBench): a position",
}


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, labels in enumerate([["benign", "injection_point", "hijacked"], ["benign", "benign", "benign"]]):
        for t, lab in enumerate(labels):
            rows.append({"traj_id": f"tr{i}", "t": t, "thought": f"think {t}", "tool": "read_email",
                         "args_json": '{"id": 1}', "obs": f"response {t}", "label": lab})
    steps = pd.DataFrame(rows)
    for c in LOT_A_STEP_METADATA:
        steps[c] = f"SENTINEL_{c.upper()}"
    traj = pd.DataFrame({"traj_id": ["tr0", "tr1"], "task": ["Summarise my e-mails", "Read e-mail 1"]})
    for c in FORBIDDEN_TRAJECTORY_FIELDS:
        traj[c] = f"SENTINEL_{c.upper()}"
    return steps, traj


@pytest.mark.parametrize("view", views.VIEWS)
@pytest.mark.parametrize("include_thought", [False, True])
def test_l12_lot_a_metadata_never_reaches_a_view(view, include_thought):
    steps, traj = _toy()
    out = views.build_view(steps, traj, view, include_thought=include_thought)
    added = set(LOT_A_STEP_METADATA) | set(FORBIDDEN_TRAJECTORY_FIELDS)
    assert not added & set(out.columns)
    assert not added & set(views.feature_columns(view, include_thought))
    text = "\n".join(out[views.feature_columns(view, include_thought)].astype(str).to_numpy().ravel())
    assert "SENTINEL_" not in text


@pytest.mark.skipif(not STEPS_PATH.exists(), reason="run `python -m agentdrift.data.external` first")
def test_l12_atbench_labels_never_reach_a_view():
    traj, steps = read_atbench()
    for view in views.VIEWS:
        out = views.build_view(steps, traj, view)
        assert not set(FORBIDDEN_TRAJECTORY_FIELDS) & set(out.columns)
        assert out["label"].isna().all()   # no step label in ATBench: evaluation at trajectory level only
