"""H3 ablation ladder (KAN-19): every rung is the v0.1 config with only the text variant changed."""

import pandas as pd

from agentdrift.config import CONFIGS
from agentdrift.models import experiment as E
from agentdrift.pipelines import run_h3 as H

BASE = E.ExperimentConfig.from_yaml(CONFIGS / "models" / "v0_1_baseline.yaml")


def test_rungs_change_only_the_text_variant():
    for r in H.ALL:
        if r.reuse:
            continue
        c = H.config_of(BASE, r, "gate", "lodo")
        assert c.model == BASE.model and c.threshold_source == BASE.threshold_source and c.evaluation == "lodo"
        assert c.version == f"h3/{r.tag}" and c.kind == BASE.kind and not c.uses_position
        changed = {k for k in E.ExperimentConfig.__dataclass_fields__
                   if getattr(c, k) != getattr(BASE, k)} - {"version", "views", "description", "evaluation"}
        assert changed == set(r.changes)
    assert H.config_of(BASE, H.BY_TAG["raw_thought"], "scanner", "cv").include_thought
    assert H.config_of(BASE, H.BY_TAG["masked"], "scanner", "cv").text == "masked"
    assert BASE.text == "anon" and not BASE.include_thought  # the official config is not mutated


def test_ladder_order_and_reused_runs():
    assert [r.step for r in H.RUNGS] == ["2 bis", "3", "4", "5"] and H.RUNGS[-1].tag == H.REFERENCE
    assert H.BY_TAG["anon"].reuse == "v0_1_baseline"  # rung 4 IS v0.1, not a rerun
    assert all(r.views == ("gate",) for r in H.ACTION_RUNGS)
    assert H.config_of(BASE, H.BY_TAG["action_raw"], "gate", "cv").columns == H.ACTION


def test_pairs_are_successive_then_to_reference():
    pairs = H._pairs()
    succ = [(a.tag, b.tag) for a, b, k in pairs if k == "successive"]
    assert ("raw", "raw_thought") in succ and ("masked", "anon") in succ and ("action_masked", "action_anon") in succ
    to_ref = [(a.tag, b.tag) for a, b, k in pairs if k == "to_reference"]
    assert ("masked", "raw_thought") in to_ref and ("masked", "anon") not in to_ref  # already a successive pair


def test_plot_h3_draws():
    from agentdrift.viz import plots as P
    rows = [{"label": l, "view": v, "mode": m, "pr_auc": x, "pr_auc_low": x - .01, "pr_auc_high": x + .01,
             "recall_full": .9, "recall_partial": .6, "recall_delayed": .5, "recall_failed": .9}
            for l in ("a", "b") for v in ("scanner", "gate") for m, x in (("cv", .95), ("lodo", .9))]
    short = pd.DataFrame([{"version": "v0_0_style", "view": v, "pr_auc": .85} for v in ("scanner", "gate")])
    P.plot_h3(pd.DataFrame(rows), ["a", "b"], short)
    import matplotlib.pyplot as plt
    plt.close("all")
