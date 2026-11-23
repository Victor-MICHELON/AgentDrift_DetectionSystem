"""Data audit and EDA computations (part A: raw data, before anonymisation).

Every function returns a DataFrame (or a dict of scalars) and never writes to data/. The notebook
`notebooks/01_eda.ipynb` and `python -m agentdrift.pipelines.run_eda` only call these functions.

Leakage rules (docs/PLAN_DONNEES.md, PROMPT_LOG P1):
- descriptive statistics may show train / val / test side by side;
- anything that looks like a score (shortcut baselines) is fitted on train and scored on val;
- TF-IDF / SVD / t-SNE here are descriptive only, never reused as model features.

Text indicators below (injection prefixes, instruction markers, confession words, unknown addresses)
are regex definitions chosen for the audit. They are approximations: each function documents its
definition so the reported rates can be read with it.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from agentdrift.config import EXTERNAL, ROOT, load_config
from agentdrift.data.load import read_steps, read_templates, read_trajectories
from agentdrift.data.schema import LABELS, SOURCE_CATEGORIES

SEED = 42
SOURCE_ORDER = list(SOURCE_CATEGORIES)
ATTACK_SOURCES = ["attacked_full", "attacked_partial", "attacked_delayed"]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://[^\s'\"),]+|www\.[^\s'\"),]+")
AMOUNT_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")

# Category-specific injection openers seen in the generator output (CONTEXT 6.2). Case-sensitive.
INJECTION_PREFIXES = {
    "CORRECTION:": re.compile(r"\bCORRECTION:"),
    "NOTE:": re.compile(r"\bNOTE:"),
    "SECURITY ALERT": re.compile(r"\bSECURITY ALERT\b"),
    "SYSTEM ALERT": re.compile(r"\bSYSTEM ALERT\b"),
    "URGENT …": re.compile(r"\bURGENT\b"),
    "ACTION REQUIRED": re.compile(r"\bACTION REQUIRED\b"),
    "Per policy …": re.compile(r"\b[Pp]er policy\b"),
}
# Explicit instruction vocabulary, excluding the category prefixes above (definition used for
# "visible injection" in the audit; case-insensitive).
INSTRUCTION_RE = re.compile(
    r"\b(?:ignore|disregard|must|required|requires|immediately|mandatory|directive|policy|compliance|"
    r"verify|verification|authenticate|forward (?:all|it|this|to)|send (?:all|it|this|a copy)|"
    r"cc|bcc|do not|instead)\b",
    re.IGNORECASE,
)
# "Confession" vocabulary in the generated reasoning (CONTEXT 6.2; case-insensitive).
CONFESSION_RE = re.compile(
    r"\bthe system(?:'s)?\b|\bnot (?:requested|asked)\b|\bdid(?: not|n't) ask\b|\bsuspicious\b|"
    r"\bwait\b|\bphishing\b|\bunknown (?:email|address|sender)\b|\bnot part of the (?:original )?task\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- inputs


def load_inputs() -> dict[str, pd.DataFrame]:
    """All three interim tables, fixed test set included for DESCRIPTIVE statistics only."""
    traj = read_trajectories(include_test=True)
    steps = read_steps(include_test=True)
    templates = read_templates()
    traj["origin_split"] = pd.Categorical(traj["origin_split"], ["train", "val", "test"], ordered=True)
    steps["origin_split"] = pd.Categorical(steps["origin_split"], ["train", "val", "test"], ordered=True)
    return {"trajectories": traj, "steps": steps, "templates": templates}


def step_groups(steps: pd.DataFrame) -> pd.Series:
    """Readable group per step: B (benign step of any trajectory), I/F (injection), H by attack type."""
    g = steps["label"].map({"benign": "B", "injection_point": "I", "failed_injection": "F"})
    h = steps["label"] == "hijacked"
    g = g.where(~h, "H " + steps["source_category"].str.removeprefix("attacked_"))
    return g


# --------------------------------------------------------------------------- A1 overview


def overview(d: dict) -> dict:
    traj, steps = d["trajectories"], d["steps"]
    return {
        "n_trajectories": len(traj),
        "n_steps": len(steps),
        "n_domains": traj["domain"].nunique(),
        "n_templates": traj["template_id"].nunique(),
        "n_tools": steps["tool"].nunique(),
        "n_worlds": traj["world_id"].nunique(),
        "steps_per_traj_mean": round(float(traj["n_steps"].mean()), 2),
        "steps_per_traj_min": int(traj["n_steps"].min()),
        "steps_per_traj_max": int(traj["n_steps"].max()),
    }


def category_by_domain(traj: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(traj["domain"], traj["source_category"]).reindex(columns=SOURCE_ORDER)
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False)


def split_composition(traj: pd.DataFrame) -> pd.DataFrame:
    t = traj.groupby("origin_split", observed=True).agg(
        trajectories=("traj_id", "size"),
        templates=("template_id", "nunique"),
        worlds=("world_id", "nunique"),
        steps=("n_steps", "sum"),
    )
    return t.join(pd.crosstab(traj["origin_split"], traj["source_category"]).reindex(columns=SOURCE_ORDER))


def example_trajectories(d: dict, n_steps: int = 6) -> pd.DataFrame:
    """One train trajectory per source category (first by id with the given length), step by step."""
    traj, steps = d["trajectories"], d["steps"]
    pool = traj[(traj["origin_split"] == "train") & (traj["n_steps"] == n_steps)].sort_values("traj_id")
    ids = pool.groupby("source_category").head(1).set_index("source_category")["traj_id"]
    ids = ids.reindex([s for s in SOURCE_ORDER if s in ids.index])
    ex = steps.loc[steps["traj_id"].isin(ids), ["source_category", "traj_id", "t", "tool", "label", "obs"]].copy()
    ex["source_category"] = pd.Categorical(ex["source_category"], SOURCE_ORDER, ordered=True)
    return ex.sort_values(["source_category", "t"]).reset_index(drop=True)


# --------------------------------------------------------------------------- A2 quality


def missing_values(steps: pd.DataFrame) -> pd.DataFrame:
    """Empty or whitespace-only fields per label (no NaN exists: the schema forbids it)."""
    empty = pd.DataFrame(
        {
            "empty thought": steps["thought"].str.strip().eq(""),
            "empty obs": steps["obs"].str.strip().eq(""),
            "empty args": steps["args_json"].eq("{}"),
            "null (any field)": steps[["thought", "tool", "args_json", "obs"]].isna().any(axis=1),
        }
    )
    t = empty.groupby(steps["label"]).sum().reindex(list(LABELS))
    t.loc["all steps"] = empty.sum()
    t["n_steps"] = steps["label"].value_counts().reindex(list(LABELS)).tolist() + [len(steps)]
    return t


def duplicates(d: dict) -> dict:
    """Exact duplicates at trajectory and step level."""
    steps, traj = d["steps"], d["trajectories"]
    content = steps["tool"] + "\x1f" + steps["args_json"] + "\x1f" + steps["obs"]
    sig = content.groupby(steps["traj_id"], sort=False).agg(lambda x: hashlib.sha1("\x1e".join(x).encode()).hexdigest())
    sig = traj.set_index("traj_id")["task"].reindex(sig.index) + sig
    full_step = content + "\x1f" + steps["thought"]
    return {
        "duplicate_trajectories (task + all steps)": int(sig.duplicated().sum()),
        "duplicate_steps (thought, tool, args, obs)": int(full_step.duplicated().sum()),
        "duplicate_observations": int(steps["obs"].duplicated().sum()),
        "distinct_observations": int(steps["obs"].nunique()),
    }


def addresses_with_space(traj: pd.DataFrame) -> dict:
    """World e-mail addresses containing a space (dataset README: 263 cosmetic cases)."""
    n_total = n_space = 0
    for w in traj["world_json"].drop_duplicates():
        world = json.loads(w)
        emails = [world["email"]] + [c["email"] for c in world["contacts"]]
        n_total += len(emails)
        n_space += sum(" " in e for e in emails)
    return {"distinct_world_addresses": n_total, "with_space": n_space}


def addresses_with_space_all_rows(traj: pd.DataFrame) -> int:
    """Same count over every trajectory row (worlds are reused), the README's unit."""
    return int(
        sum(
            sum(" " in e for e in [w["email"]] + [c["email"] for c in w["contacts"]])
            for w in map(json.loads, traj["world_json"])
        )
    )


def length_outliers(steps: pd.DataFrame) -> pd.DataFrame:
    """Character length of each text field: quantiles and extreme values."""
    rows = {}
    for col in ("thought", "args_json", "obs"):
        n = steps[col].str.len()
        q = n.quantile([0.5, 0.99])
        rows[col] = {"median": q[0.5], "p99": q[0.99], "max": n.max(), "n_above_p99": int((n > q[0.99]).sum())}
    return pd.DataFrame(rows).T


def test_dev_similarity(d: dict) -> pd.Series:
    """L6 check: for every test trajectory, max cosine similarity to a dev (train + val) trajectory.
    TF-IDF fitted on dev text only; descriptive."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    steps, traj = d["steps"], d["trajectories"]
    text = (steps["tool"] + " " + steps["args_json"] + " " + steps["obs"]).groupby(steps["traj_id"]).agg(" ".join)
    split = traj.set_index("traj_id")["origin_split"].reindex(text.index)
    task = traj.set_index("traj_id")["task"].reindex(text.index)
    text = task + " " + text
    dev, test = text[split != "test"], text[split == "test"]
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(dev)
    a, b = vec.transform(test), vec.transform(dev)
    sims = np.asarray((a @ b.T).max(axis=1).todense()).ravel()
    return pd.Series(sims, index=test.index, name="max_cosine_to_dev")


# --------------------------------------------------------------------------- A3 imbalance


def prevalence(d: dict, by: str = "origin_split") -> pd.DataFrame:
    steps, traj = d["steps"], d["trajectories"]
    s = steps.groupby(by, observed=True).agg(
        steps=("t", "size"), y_scan=("y_scan", "mean"), y_gate=("y_gate", "mean"), first_h=("is_first_h", "sum")
    )
    t = traj.groupby(by, observed=True).agg(trajectories=("traj_id", "size"), y_traj=("y_traj", "mean"))
    out = t.join(s)
    tot_s, tot_t = steps, traj
    out.loc["all"] = [len(tot_t), tot_t["y_traj"].mean(), len(tot_s), tot_s["y_scan"].mean(),
                      tot_s["y_gate"].mean(), tot_s["is_first_h"].sum()]
    return out.astype({"trajectories": int, "steps": int, "first_h": int})


def hijacked_per_attack(traj: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    h = steps[steps["y_gate"] == 1].groupby("traj_id").size().rename("n_H")
    t = traj[traj["category"] == "attacked"].set_index("traj_id").join(h)
    return t.groupby("source_category")["n_H"].describe()[["count", "mean", "min", "max"]].reindex(ATTACK_SOURCES)


# --------------------------------------------------------------------------- A4 temporal structure


def length_by_category(traj: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(traj["source_category"], traj["n_steps"]).reindex(SOURCE_ORDER)
    t["mean"] = traj.groupby("source_category")["n_steps"].mean().reindex(SOURCE_ORDER)
    return t


def length_attacked_vs_rest(traj: pd.DataFrame) -> dict:
    m = traj.groupby(traj["category"] == "attacked")["n_steps"].mean()
    return {"attacked": round(float(m[True]), 2), "not_attacked": round(float(m[False]), 2)}


def label_by_position(steps: pd.DataFrame, source_category: str | None = None) -> pd.DataFrame:
    """Share of each label at each step index t (rows t, columns labels)."""
    s = steps if source_category is None else steps[steps["source_category"] == source_category]
    return pd.crosstab(s["t"], s["label"], normalize="index").reindex(columns=list(LABELS), fill_value=0)


def hijack_rate_by_position(steps: pd.DataFrame) -> pd.DataFrame:
    """P(y_gate = 1 | t, source category): rows source categories, columns t."""
    return steps.pivot_table(index="source_category", columns="t", values="y_gate", aggfunc="mean").reindex(SOURCE_ORDER)


def injection_position(traj: pd.DataFrame) -> pd.DataFrame:
    t = traj[traj["injection_step"].notna()]
    return pd.crosstab(t["source_category"], t["injection_step"]).reindex(
        [s for s in SOURCE_ORDER if s in set(t["source_category"])]
    )


def injection_to_hijack_gap(traj: pd.DataFrame) -> pd.DataFrame:
    t = traj[traj["first_h_step"].notna()]
    gap = (t["first_h_step"] - t["injection_step"]).rename("gap")
    return pd.crosstab(t["source_category"], gap).reindex(ATTACK_SOURCES)


def after_first_hijack(steps: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    """Labels of the steps after the first H: does the agent go back to benign behaviour?"""
    s = steps.merge(traj[["traj_id", "first_h_step"]], on="traj_id")
    s = s[s["first_h_step"].notna() & (s["t"] > s["first_h_step"])]
    return pd.crosstab(s["source_category"], s["label"], normalize="index").reindex(ATTACK_SOURCES)


# --------------------------------------------------------------------------- A5 tools


def tool_label_table(steps: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    t = pd.crosstab(steps["tool"], steps["label"]).reindex(columns=list(LABELS), fill_value=0)
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False).head(top)


def tools_summary(steps: pd.DataFrame) -> dict:
    by_tool = steps.groupby("tool")["label"].agg(set)
    h_tools = by_tool[by_tool.map(lambda s: "hijacked" in s)]
    return {
        "n_tools": len(by_tool),
        "tools_seen_in_H": len(h_tools),
        "tools_only_in_H": int(h_tools.map(lambda s: s == {"hijacked"}).sum()),
        "tools_per_domain": steps.groupby("domain")["tool"].nunique().to_dict(),
        "tools_used_in_two_domains_or_more": int((steps.groupby("tool")["domain"].nunique() > 1).sum()),
    }


def out_of_expected_tools(d: dict) -> pd.DataFrame:
    """Share of steps whose tool is NOT in the template's expected_tools (oracle, never a feature)."""
    steps = d["steps"]
    expected = d["templates"].set_index("template_id")["expected_tools"].map(set)
    out = [tool not in expected[tpl] for tool, tpl in zip(steps["tool"], steps["template_id"])]
    s = steps.assign(out_of_expected=out, group=step_groups(steps))
    return s.groupby("group")["out_of_expected"].agg(["mean", "size"]).rename(columns={"mean": "share", "size": "n"})


# --------------------------------------------------------------------------- A6 text


def text_lengths(steps: pd.DataFrame) -> pd.DataFrame:
    g = step_groups(steps)
    return pd.DataFrame(
        {f"{c} (median chars)": steps[c].str.len().groupby(g).median() for c in ("thought", "args_json", "obs")}
    )


def injection_prefixes(steps: pd.DataFrame) -> pd.DataFrame:
    """Share of observations containing each prefix: injection steps per category vs benign steps."""
    inj = steps[steps["label"].isin(["injection_point", "failed_injection"])]
    cols = {}
    for sc in ["attacked_full", "attacked_partial", "attacked_delayed", "failed_attack"]:
        obs = inj.loc[inj["source_category"] == sc, "obs"]
        cols[f"injection · {sc.removeprefix('attacked_').replace('failed_attack', 'failed')}"] = {k: obs.str.contains(r).mean() for k, r in INJECTION_PREFIXES.items()}
    obs_b = steps.loc[steps["label"] == "benign", "obs"]
    cols["benign steps (all)"] = {k: obs_b.str.contains(r).mean() for k, r in INJECTION_PREFIXES.items()}
    return pd.DataFrame(cols)


def visibility_indicators(steps: pd.DataFrame) -> pd.DataFrame:
    """Per injection step (raw text): the three regex visibility indicators and their union."""
    inj = steps[steps["label"].isin(["injection_point", "failed_injection"])].copy()
    prefix_any = re.compile("|".join(r.pattern for r in INJECTION_PREFIXES.values()))
    inj["category prefix"] = inj["obs"].str.contains(prefix_any)
    inj["instruction vocabulary"] = inj["obs"].str.contains(INSTRUCTION_RE)
    inj["external address or URL"] = [
        _has_external(o, ud) for o, ud in zip(inj["obs"], _user_domains(inj))
    ]
    inj["any of the three"] = inj[["category prefix", "instruction vocabulary", "external address or URL"]].any(axis=1)
    return inj


def injection_visibility(steps: pd.DataFrame) -> pd.DataFrame:
    """How readable is the injection in the poisoned observation? Three regex definitions:
    category prefix, explicit instruction vocabulary (prefixes excluded), external address or URL
    (address whose domain differs from the user's own domain)."""
    inj = visibility_indicators(steps)
    order = ["attacked_full", "attacked_partial", "attacked_delayed", "failed_attack"]
    t = inj.groupby("source_category")[
        ["category prefix", "instruction vocabulary", "external address or URL", "any of the three"]
    ].mean().reindex(order)
    t["n"] = inj["source_category"].value_counts().reindex(order)
    return t


def confession_in_thought(steps: pd.DataFrame) -> pd.DataFrame:
    """Share of steps whose `thought` contains confession vocabulary, by label and source category."""
    has = steps["thought"].str.contains(CONFESSION_RE)
    t = has.groupby([steps["source_category"], steps["label"]]).agg(["mean", "size"])
    t = t.rename(columns={"mean": "share", "size": "n"}).reset_index()
    t["source_category"] = pd.Categorical(t["source_category"], SOURCE_ORDER, ordered=True)
    return t.sort_values(["source_category", "label"]).reset_index(drop=True)


# --------------------------------------------------------------------------- A7 arguments and provenance


def _user_domains(steps: pd.DataFrame) -> list[str]:
    worlds = _worlds()
    return [worlds[w]["user_domain"] for w in steps["world_id"]]


@lru_cache(maxsize=1)
def _worlds() -> dict:
    traj = read_trajectories(include_test=True).drop_duplicates("world_id")
    out = {}
    for wid, wj in zip(traj["world_id"], traj["world_json"]):
        w = json.loads(wj)
        emails = {w["email"].lower().replace(" ", "")} | {c["email"].lower().replace(" ", "") for c in w["contacts"]}
        out[wid] = {"user_domain": _domain(w["email"]), "known_emails": emails}
    return out


def _domain(email: str) -> str:
    return email.lower().replace(" ", "").rsplit("@", 1)[-1]


def _has_external(text: str, user_domain: str) -> bool:
    if URL_RE.search(text):
        return True
    return any(_domain(e) != user_domain for e in EMAIL_RE.findall(text))


def _leaf_values(obj) -> list[str]:
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _leaf_values(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in _leaf_values(x)]
    if obj is None or isinstance(obj, bool):
        return []
    return [str(obj)]


def provenance_indicators(d: dict) -> pd.DataFrame:
    """Per-step indicators on the arguments of the proposed action a_t, computed from x_t only
    (task, world, previous observations, current args). Definitions:
    - unknown address: an e-mail in args that is neither the user's, nor a world contact, nor in the task;
    - external domain: an e-mail in args whose domain differs from the user's domain;
    - unknown same-domain address: unknown address that shares the user's domain (partial attacks);
    - arg from previous obs: a leaf argument value (>= 4 chars) found in obs_0..obs_{t-1} but not in the task;
    - amount in args: a $ amount in args."""
    steps = d["steps"].sort_values(["traj_id", "t"])
    task = d["trajectories"].set_index("traj_id")["task"]
    worlds = _worlds()
    rows = []
    prev_traj, history = None, ""
    for traj_id, wid, args_json, obs in zip(steps["traj_id"], steps["world_id"], steps["args_json"], steps["obs"]):
        if traj_id != prev_traj:
            prev_traj, history = traj_id, ""
            task_l = task[traj_id].lower()
        w = worlds[wid]
        args_l = args_json.lower()
        emails = {e.replace(" ", "") for e in EMAIL_RE.findall(args_l)}
        unknown = {e for e in emails if e not in w["known_emails"] and e not in task_l}
        leaves = [v.lower() for v in _leaf_values(json.loads(args_json)) if len(v) >= 4]
        rows.append(
            (
                bool(unknown),
                any(_domain(e) != w["user_domain"] for e in emails),
                any(_domain(e) == w["user_domain"] for e in unknown),
                any(v in history and v not in task_l for v in leaves),
                bool(AMOUNT_RE.search(args_json)),
            )
        )
        history += " " + obs.lower()
    cols = ["unknown address", "external domain", "unknown same-domain address", "arg from previous obs", "amount in args"]
    ind = pd.DataFrame(rows, columns=cols, index=steps.index).reindex(d["steps"].index)
    return ind


def provenance_by_group(d: dict, ind: pd.DataFrame | None = None) -> pd.DataFrame:
    ind = provenance_indicators(d) if ind is None else ind
    g = step_groups(d["steps"])
    t = ind.groupby(g).mean()
    t["n"] = g.value_counts()
    order = ["B", "I", "F", "H full", "H partial", "H delayed"]
    return t.reindex([o for o in order if o in t.index])


# --------------------------------------------------------------------------- A8 leakage and shortcuts


def world_purity(traj: pd.DataFrame) -> pd.DataFrame:
    """Per world: number of trajectories and share of its majority source category."""
    c = pd.crosstab(traj["world_id"], traj["source_category"])
    w = pd.DataFrame({"n": c.sum(axis=1), "majority_share": c.max(axis=1) / c.sum(axis=1)})
    w["domain"] = traj.drop_duplicates("world_id").set_index("world_id")["domain"].reindex(w.index)
    return w


def world_purity_by_domain(traj: pd.DataFrame) -> pd.DataFrame:
    """Share of trajectories whose world puts >= 95 % of its trajectories in one source category."""
    w = world_purity(traj)
    t = traj[["world_id", "domain"]].merge(w[["majority_share"]], left_on="world_id", right_index=True)
    return t.groupby("domain").agg(
        worlds=("world_id", "nunique"),
        weighted_majority_share=("majority_share", "mean"),
        share_traj_in_pure_worlds=("majority_share", lambda x: (x >= 0.95).mean()),
    )


def _binary_scores(y, score, pred=None) -> dict:
    out = {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score), "prevalence": float(np.mean(y))}
    if pred is not None:
        out |= {"f1": f1_score(y, pred), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred)}
    return {k: round(float(v), 3) for k, v in out.items()}


def world_lookup_baseline(traj: pd.DataFrame, train: str = "train", evaluate: str = "val") -> dict:
    """Shortcut: predict a trajectory's class from the majority category of its world in train.
    Fitted on train, scored on val. Unseen worlds get the train attacked rate (score) / class 0."""
    tr, ev = traj[traj["origin_split"] == train], traj[traj["origin_split"] == evaluate]
    world_rate = tr.groupby("world_id")["y_traj"].mean()
    score = ev["world_id"].map(world_rate)
    covered = score.notna()
    score = score.fillna(tr["y_traj"].mean())
    pred = (score > 0.5).astype(int)
    out = _binary_scores(ev["y_traj"], score, pred)
    out["val_worlds_seen_in_train"] = round(float(covered.mean()), 3)
    cov = ev[covered]
    if covered.any():
        out["f1_on_covered"] = round(float(f1_score(cov["y_traj"], pred[covered])), 3)
    return out


def position_baseline(steps: pd.DataFrame, target: str, train: str = "train", evaluate: str = "val") -> dict:
    """Shortcut: score = empirical rate of the target at step index t in train; scored on val."""
    tr, ev = steps[steps["origin_split"] == train], steps[steps["origin_split"] == evaluate]
    rate = tr.groupby("t")[target].mean()
    score = ev["t"].map(rate).fillna(tr[target].mean())
    return _binary_scores(ev[target], score)


def length_baseline(traj: pd.DataFrame, evaluate: str = "val") -> dict:
    """Future leak (L3): total trajectory length as a score of y_traj (no fit: the score is n_steps)."""
    ev = traj[traj["origin_split"] == evaluate]
    return _binary_scores(ev["y_traj"], ev["n_steps"])


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def association_table(d: dict) -> pd.DataFrame:
    """Cramér's V (descriptive, no p-value: steps are dependent) between candidate variables and labels."""
    traj, steps = d["trajectories"], d["steps"]
    rows = [
        ("trajectory", "domain", "y_traj", cramers_v(traj["domain"], traj["y_traj"])),
        ("trajectory", "complexity", "y_traj", cramers_v(traj["complexity"], traj["y_traj"])),
        ("trajectory", "template_id", "y_traj", cramers_v(traj["template_id"], traj["y_traj"])),
        ("trajectory", "world_id", "y_traj", cramers_v(traj["world_id"], traj["y_traj"])),
        ("trajectory", "n_steps", "y_traj", cramers_v(traj["n_steps"], traj["y_traj"])),
        ("step", "t (position)", "y_gate", cramers_v(steps["t"], steps["y_gate"])),
        ("step", "t (position)", "y_scan", cramers_v(steps["t"], steps["y_scan"])),
        ("step", "tool", "y_gate", cramers_v(steps["tool"], steps["y_gate"])),
        ("step", "domain", "y_gate", cramers_v(steps["domain"], steps["y_gate"])),
    ]
    return pd.DataFrame(rows, columns=["level", "variable", "target", "cramers_v"]).sort_values("cramers_v", ascending=False)


def shortcut_baselines(d: dict) -> pd.DataFrame:
    traj, steps = d["trajectories"], d["steps"]
    rows = {
        "world lookup → y_traj": world_lookup_baseline(traj),
        "position only → y_gate": position_baseline(steps, "y_gate"),
        "position only → y_scan": position_baseline(steps, "y_scan"),
        "trajectory length → y_traj": length_baseline(traj),
    }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- A9 dimensionality reduction


def text_projection(d: dict, per_label: int = 600, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Descriptive projection of step text (tool + arguments + observation) of dev steps.
    TF-IDF -> TruncatedSVD(50) -> t-SNE(2). Silhouettes (cosine, SVD space) by label and by domain
    tell which grouping the text space follows. Never reused as a feature."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    steps = d["steps"]
    dev = steps[steps["origin_split"] != "test"]
    sample = dev.groupby("label").sample(n=per_label, random_state=seed).reset_index(drop=True)
    text = sample["tool"] + " " + sample["args_json"] + " " + sample["obs"]
    x = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(text)
    z = TruncatedSVD(50, random_state=seed).fit_transform(x)
    xy = TSNE(2, perplexity=30, init="pca", random_state=seed).fit_transform(z)
    sil = {
        "silhouette_by_label": round(float(silhouette_score(z, sample["label"], metric="cosine")), 3),
        "silhouette_by_domain": round(float(silhouette_score(z, sample["domain"], metric="cosine")), 3),
        "n_points": len(sample),
    }
    out = sample[["traj_id", "t", "label", "domain", "source_category"]].assign(x=xy[:, 0], y=xy[:, 1])
    return out, sil


# --------------------------------------------------------------------------- A10 external test set


def load_atbench() -> pd.DataFrame:
    """ATBench (1,000 trajectories) summarised per trajectory: label, risk source, number of tool calls,
    tools available, task length. In-scope risk sources fixed in JOURNAL 19/11 (configs/data.yaml)."""
    from agentdrift.data.external import parse_action, read_raw

    raw = read_raw()
    in_scope = set(load_config("data")["atbench"]["scope_risk_sources"])
    rows = []
    for r in raw:
        conv = [m for c in r["contents"] for m in c]
        # tool calls only (as AgentDrift steps); replies to the user are not actions on the world
        actions = [m for m in conv if m["role"] == "agent" and parse_action(m.get("action"))[0] == "call"]
        task = next((m["content"] for m in conv if m["role"] == "user"), "")
        env = [m["content"] for m in conv if m["role"] == "environment"]
        rows.append(
            {
                "id": r["id"],
                "label": int(r["label"]),
                "risk_source": r["risk_source"],
                "in_scope": r["risk_source"] in in_scope or r["risk_source"] == "benign" or int(r["label"]) == 0,
                "n_actions": len(actions),
                "n_tools_available": len(r["tool_used"]),
                "task_chars": len(task),
                "obs_chars_median": float(np.median([len(e) for e in env])) if env else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compare_external(d: dict, atb: pd.DataFrame) -> pd.DataFrame:
    traj, steps = d["trajectories"], d["steps"]
    pools = json.loads((ROOT / load_config("data")["agentdrift"]["raw_dir"] / "pools" / "tool_pools.json").read_text())
    ad = pd.DataFrame(
        {
            "tool calls per trajectory": traj["n_steps"].describe(),
            "tools available per trajectory": traj["domain"].map({k: len(v) for k, v in pools.items()}).describe(),
            "task length (chars)": traj["task"].str.len().describe(),
            "observation length (chars, median per trajectory)": steps["obs"].str.len().groupby(steps["traj_id"]).median().describe(),
        }
    ).T[["mean", "50%", "max"]]
    ab = pd.DataFrame(
        {
            "tool calls per trajectory": atb["n_actions"].describe(),
            "tools available per trajectory": atb["n_tools_available"].describe(),
            "task length (chars)": atb["task_chars"].describe(),
            "observation length (chars, median per trajectory)": atb["obs_chars_median"].describe(),
        }
    ).T[["mean", "50%", "max"]]
    return pd.concat({"AgentDrift": ad, "ATBench": ab}, axis=1).round(1)


def atbench_scope(atb: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(atb["risk_source"], atb["label"].map({0: "safe", 1: "unsafe"}), margins=True)


__all__ = [n for n in dir() if not n.startswith("_") and n not in {"annotations"}]
EXTERNAL_DIR = EXTERNAL  # re-export for the notebook


# --------------------------------------------------------------------------- reproduction of 19/11


def reproduction_table(d: dict, prov: pd.DataFrame, shortcuts: pd.DataFrame, visibility: pd.DataFrame,
                       confession: pd.DataFrame, oox: pd.DataFrame) -> pd.DataFrame:
    """Exploratory numbers of 19/11 (CONTEXT 5 and 6.2) next to the values recomputed here.
    The definitions were NOT tuned to match: a gap means the 19/11 definition differed or was wrong."""
    conf = confession.set_index(["source_category", "label"])["share"]
    length = length_attacked_vs_rest(d["trajectories"])
    h_mask = prov.index.isin(["H full", "H partial", "H delayed"])
    h_weighted = (prov.loc[h_mask, "arg from previous obs"] * prov.loc[h_mask, "n"]).sum() / prov.loc[h_mask, "n"].sum()
    rows = [
        ("position only → y_gate, ROC-AUC", "0.754", shortcuts.loc["position only → y_gate", "roc_auc"], "train rate per t, scored on val"),
        ("position only → y_scan, ROC-AUC", "0.52", shortcuts.loc["position only → y_scan", "roc_auc"], "train rate per t, scored on val"),
        ("world lookup → y_traj, F1", "0.77", shortcuts.loc["world lookup → y_traj", "f1"], "train world majority, scored on val"),
        ("tool outside expected_tools, H full", "0.69–0.81", oox.loc["H full", "share"], "oracle, step level"),
        ("tool outside expected_tools, H delayed", "0.69–0.81", oox.loc["H delayed", "share"], "oracle, step level"),
        ("tool outside expected_tools, H partial", "0.40", oox.loc["H partial", "share"], "oracle, step level"),
        ("tool outside expected_tools, B", "0.24–0.32", oox.loc["B", "share"], "oracle, step level"),
        ("external-domain address in args, H full", "0.43", prov.loc["H full", "external domain"], "address domain ≠ user domain"),
        ("external-domain address in args, H delayed", "0.15", prov.loc["H delayed", "external domain"], "address domain ≠ user domain"),
        ("external-domain address in args, H partial", "0.04", prov.loc["H partial", "external domain"], "address domain ≠ user domain"),
        ("external-domain address in args, B", "0.02", prov.loc["B", "external domain"], "address domain ≠ user domain"),
        ("arg from previous obs, all H", "0.32", round(float(h_weighted), 3), "leaf value ≥ 4 chars in obs_<t, not in task"),
        ("arg from previous obs, B", "0.16", prov.loc["B", "arg from previous obs"], "leaf value ≥ 4 chars in obs_<t, not in task"),
        ("instruction vocabulary at injection, full", "0.996", visibility.loc["attacked_full", "instruction vocabulary"], "INSTRUCTION_RE, prefixes excluded"),
        ("instruction vocabulary at injection, failed", "0.795", visibility.loc["failed_attack", "instruction vocabulary"], "INSTRUCTION_RE, prefixes excluded"),
        ("instruction vocabulary at injection, partial", "0.047", visibility.loc["attacked_partial", "instruction vocabulary"], "INSTRUCTION_RE, prefixes excluded"),
        ("instruction vocabulary at injection, delayed", "0.07", visibility.loc["attacked_delayed", "instruction vocabulary"], "INSTRUCTION_RE, prefixes excluded"),
        ("external address or URL at injection, partial", "0.13–0.21", visibility.loc["attacked_partial", "external address or URL"], "URL or address domain ≠ user domain"),
        ("external address or URL at injection, delayed", "0.13–0.21", visibility.loc["attacked_delayed", "external address or URL"], "URL or address domain ≠ user domain"),
        ("confession words in thought, H partial", "0.445", conf.loc[("attacked_partial", "hijacked")], "CONFESSION_RE"),
        ("confession words in thought, B of failed attacks", "0.21", conf.loc[("failed_attack", "benign")], "CONFESSION_RE"),
        ("mean length, attacked vs other", "6.07 vs 5.44", f"{length['attacked']} vs {length['not_attacked']}", "mean n_steps"),
    ]
    t = pd.DataFrame(rows, columns=["measure", "19/11", "recomputed", "definition used"])
    t["recomputed"] = [v if isinstance(v, str) else round(float(v), 3) for v in t["recomputed"]]
    return t


# --------------------------------------------------------------------------- additional checks


def world_overlap(traj: pd.DataFrame) -> dict:
    """Share of val / test trajectories whose world already appears in earlier parts."""
    train = set(traj.loc[traj["origin_split"] == "train", "world_id"])
    dev = set(traj.loc[traj["origin_split"] != "test", "world_id"])
    val, test = traj[traj["origin_split"] == "val"], traj[traj["origin_split"] == "test"]
    return {
        "val_traj_world_in_train": round(float(val["world_id"].isin(train).mean()), 3),
        "test_traj_world_in_dev": round(float(test["world_id"].isin(dev).mean()), 3),
    }


def top_repeated_observations(steps: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Most repeated observations with their label mix: repeated text is not always generic."""
    top = steps["obs"].value_counts().head(n)
    mix = pd.crosstab(steps.loc[steps["obs"].isin(top.index), "obs"], steps["label"]).reindex(top.index)
    return mix.assign(total=top).reindex(columns=[c for c in LABELS if c in mix.columns] + ["total"]).fillna(0).astype(int)


def prefix_counts_in_benign(steps: pd.DataFrame) -> pd.Series:
    obs = steps.loc[steps["label"] == "benign", "obs"]
    return pd.Series({k: int(obs.str.contains(r).sum()) for k, r in INJECTION_PREFIXES.items()}, name="benign observations")


def atbench_tool_vocabulary() -> int:
    """Number of distinct tool names in ATBench (open vocabulary)."""
    raw = json.loads((ROOT / load_config("data")["atbench"]["raw_dir"] / "ATBench" / "test.json").read_text(encoding="utf-8"))
    return len({t["name"] for r in raw for t in r["tool_used"]})


# --------------------------------------------------------------------------- part B: effect of anonymisation

ANON_EMAIL_RE = re.compile(r"<(?:USER_EMAIL|CONTACT_\d+_EMAIL|INT_EMAIL_\d+|EXT_EMAIL_\d+)>")


def provenance_indicators_anon(steps: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    """Same indicators as `provenance_indicators`, read from the placeholders of the anonymised text.
    Equal rates before / after = the anonymisation kept the relations it was meant to keep."""
    s = steps.sort_values(["traj_id", "t"])
    task = traj.set_index("traj_id")["task"]
    rows, prev_traj, history = [], None, ""
    for traj_id, args_json, obs in zip(s["traj_id"], s["args_json"], s["obs"]):
        if traj_id != prev_traj:
            prev_traj, history = traj_id, ""
            task_l = task[traj_id].lower()
        emails = set(ANON_EMAIL_RE.findall(args_json))
        unknown = {e for e in emails if ("INT_EMAIL" in e or "EXT_EMAIL" in e) and e.lower() not in task_l}
        leaves = [v.lower() for v in _leaf_values(json.loads(args_json)) if len(v) >= 4]
        rows.append(
            (
                bool(unknown),
                any("EXT_EMAIL" in e for e in emails),
                any("INT_EMAIL" in e for e in unknown),
                any(v in history and v not in task_l for v in leaves),
                bool(AMOUNT_RE.search(args_json)),
            )
        )
        history += " " + obs.lower()
    cols = ["unknown address", "external domain", "unknown same-domain address", "arg from previous obs", "amount in args"]
    return pd.DataFrame(rows, columns=cols, index=s.index).reindex(steps.index)


def provenance_before_after(d_raw: dict, d_anon: dict) -> pd.DataFrame:
    """Provenance rates by step group, raw vs anonymised (columns: indicator x version)."""
    before = provenance_by_group(d_raw)
    ind = provenance_indicators_anon(d_anon["steps"], d_anon["trajectories"])
    after = provenance_by_group(d_anon, ind)
    cols = [c for c in before.columns if c != "n"]
    return pd.concat({"raw": before[cols], "anonymised": after[cols]}, axis=1).swaplevel(axis=1).sort_index(axis=1, level=0)


def trajectory_text(traj: pd.DataFrame, steps: pd.DataFrame) -> pd.Series:
    body = (steps["thought"] + " " + steps["args_json"] + " " + steps["obs"]).groupby(steps["traj_id"]).agg(" ".join)
    return traj.set_index("traj_id")["task"] + " " + body.reindex(traj["traj_id"]).to_numpy()


def world_probe(traj: pd.DataFrame, steps: pd.DataFrame, train: str = "train", evaluate: str = "val") -> dict:
    """Can the TEXT reveal the world? 1-nearest-neighbour probe: TF-IDF fitted on train trajectories, each val
    trajectory takes the world of its most similar train trajectory (val trajectories whose world exists in train).

    Reference level: worlds are tied to a generation job (one source category per domain), so a neighbour with
    similar CONTENT is often of the same category, hence of one of fewer worlds. The fair chance level is therefore
    1 / (number of train worlds of the same domain x source category), not 1 / (worlds of the domain).
    Identity is removed when world_accuracy falls close to chance_given_category."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = trajectory_text(traj, steps)
    t = traj.set_index("traj_id")
    tr_ids = t.index[t["origin_split"] == train]
    ev_ids = t.index[(t["origin_split"] == evaluate) & t["world_id"].isin(set(t.loc[tr_ids, "world_id"]))]
    vec = TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)<[A-Z_0-9]+>|\b\w\w+\b").fit(text[tr_ids])
    nn = np.asarray((vec.transform(text[ev_ids]) @ vec.transform(text[tr_ids]).T).argmax(axis=1)).ravel()
    neighbour = tr_ids[nn]
    same_world = t.loc[neighbour, "world_id"].to_numpy() == t.loc[ev_ids, "world_id"].to_numpy()
    same_cat = t.loc[neighbour, "source_category"].to_numpy() == t.loc[ev_ids, "source_category"].to_numpy()
    key = t["domain"] + "|" + t["source_category"]
    n_worlds = t.loc[tr_ids].groupby(key[tr_ids])["world_id"].nunique()
    n_worlds_dom = t.loc[tr_ids].groupby("domain")["world_id"].nunique()
    return {
        "n_eval": len(ev_ids),
        "world_accuracy": round(float(same_world.mean()), 3),
        "chance_given_domain": round(float((1 / t.loc[ev_ids, "domain"].map(n_worlds_dom)).mean()), 4),
        "chance_given_category": round(float((1 / key[ev_ids].map(n_worlds)).mean()), 4),
        "neighbour_same_category": round(float(same_cat.mean()), 3),
    }


# --------------------------------------------------------------------------- part B: effect of the preprocessing


def load_inputs_anon() -> dict[str, pd.DataFrame]:
    """Anonymised trajectory and step tables, fixed test set included for DESCRIPTIVE statistics only."""
    traj = read_trajectories(include_test=True, anonymized=True)
    steps = read_steps(include_test=True, anonymized=True)
    for df in (traj, steps):
        df["origin_split"] = pd.Categorical(df["origin_split"], ["train", "val", "test"], ordered=True)
    return {"trajectories": traj, "steps": steps}


def masking_function():
    """`features.masking.mask_artifacts` (Lot B, KAN-19) if it exists yet, else None."""
    try:
        from agentdrift.features.masking import mask_artifacts
    except ImportError:
        return None
    return mask_artifacts


def text_before_after(d_raw: dict, d_anon: dict, seed: int = SEED) -> pd.DataFrame:
    """Side-by-side text of one injection step per attack category (train only, seeded), raw / anonymised
    (/ masked once `features/masking.py` exists). Picked among steps whose text the anonymisation changed."""
    raw, anon = d_raw["steps"], d_anon["steps"]
    key = ["traj_id", "t"]
    both = raw.merge(anon[key + ["thought", "args_json", "obs"]], on=key, suffixes=("", "_anon"))
    both = both[(both["origin_split"] == "train") & both["label"].isin(["injection_point", "failed_injection"])]
    changed = (both["obs"] != both["obs_anon"]) | (both["args_json"] != both["args_json_anon"])
    mask = masking_function()
    rows = []
    for sc in ["attacked_full", "attacked_partial", "attacked_delayed", "failed_attack"]:
        pool = both[changed & (both["source_category"] == sc)].sort_values(key)
        if pool.empty:
            continue
        s = pool.iloc[int(np.random.default_rng(seed).integers(len(pool)))]
        for field in ["thought", "args_json", "obs"]:
            row = {"example": sc, "field": field, "raw": s[field], "anonymised": s[f"{field}_anon"]}
            if mask is not None:
                row["masked"] = mask(s[f"{field}_anon"])
            rows.append(row)
    return pd.DataFrame(rows).set_index(["example", "field"])


def residual_before_after(d_raw: dict, d_anon: dict) -> pd.DataFrame:
    """Share of trajectories whose text contains each kind of world identifier, raw vs anonymised."""
    from agentdrift.data.anonymize import residual_identifiers

    traj, steps = d_raw["trajectories"], d_raw["steps"]
    raw = residual_identifiers(traj, traj.drop(columns="world_json"), steps)
    anon = residual_identifiers(traj, d_anon["trajectories"], d_anon["steps"])
    out = pd.DataFrame({"raw": raw, "anonymised": anon}) / len(traj)
    out["n residual (anonymised)"] = anon
    return out


def world_probe_before_after(d_raw: dict, d_anon: dict) -> pd.DataFrame:
    """Text-only world probe (train -> val) on the raw and the anonymised text, with its chance levels."""
    return pd.DataFrame({
        "raw": world_probe(d_raw["trajectories"], d_raw["steps"]),
        "anonymised": world_probe(d_anon["trajectories"], d_anon["steps"]),
    }).T


def tool_risk_by_group(steps: pd.DataFrame, table: pd.DataFrame, allow_draft: bool = False) -> pd.DataFrame | None:
    """Share of each tool risk category per step group (B, I, F, H by attack type).

    Returns None while tool_risk.yaml is a draft: showing risk x labels during the review would let the reviewer
    tune the categories on the labels (the yaml forbids it). allow_draft=True only for tests."""
    if table.attrs.get("status") != "validated" and not allow_draft:
        return None
    risk = steps["tool"].map(table.set_index("tool")["risk"])
    groups = step_groups(steps)
    t = pd.crosstab(groups, risk, normalize="index")
    t["n"] = groups.value_counts()
    order = ["B", "I", "F", "H full", "H partial", "H delayed"]
    return t.reindex([g for g in order if g in t.index])


def visible_vs_heuristic(sheet: pd.DataFrame, steps_raw: pd.DataFrame) -> pd.DataFrame | None:
    """Human `injection_visible` against the regex union of part A ("any of the three"), per attack category.

    `sheet` = annotated rows (traj_id, t, injection_visible). None until at least one row is annotated."""
    from sklearn.metrics import cohen_kappa_score

    if sheet is None or sheet.empty:
        return None
    ind = visibility_indicators(steps_raw)[["traj_id", "t", "source_category", "any of the three"]]
    m = sheet.merge(ind, on=["traj_id", "t"], how="left")
    m["human"] = m["injection_visible"] == "oui"
    rows = {}
    for sc, g in list(m.groupby("source_category")) + [("all", m)]:
        rows[sc] = {
            "annotated": len(g),
            "visible (human)": g["human"].mean(),
            "visible (regex union)": g["any of the three"].mean(),
            "agreement": (g["human"] == g["any of the three"]).mean(),
            "kappa": cohen_kappa_score(g["human"], g["any of the three"]) if g["human"].nunique() > 1 else np.nan,
        }
    order = ["attacked_full", "attacked_partial", "attacked_delayed", "all"]
    return pd.DataFrame(rows).T.reindex([o for o in order if o in rows])


# --------------------------------------------------------------------------- part C: evaluation protocol
#
# Leakage rules for part C (L5): the fixed test set only contributes COUNTS (size and prevalence, already shown
# in the step-8 table). No score is ever computed on it: the expected interval widths are measured on the dev
# folds with a metadata-only reference score fitted out of fold, then projected to the size of the test set.

FPR_TARGETS = {"1%": 0.01, "0.2%": 0.002}
MIN_FALSE_POSITIVES = 10  # below this, the observed FPR has a relative standard error > 1/sqrt(10) ~ 32%
UNIT_TEST = "test"


def _unit_name(fold: int) -> str:
    return UNIT_TEST if fold == -1 else f"fold {fold}"


def load_inputs_protocol() -> dict[str, pd.DataFrame]:
    """Part A inputs with the outer fold and the world-control fold of every trajectory and step."""
    from agentdrift.splits import grouped as G

    d = load_inputs()
    folds = G.load_folds()
    d["folds"] = folds
    d["trajectories"] = G.attach_folds(d["trajectories"], folds)
    d["steps"] = G.attach_folds(d["steps"], folds)
    return d


def fold_table(d: dict) -> pd.DataFrame:
    """Size and prevalence per outer fold and for the fixed test set (counts only for the test set)."""
    from agentdrift.splits.grouped import fold_summary

    s = fold_summary(d["folds"], d["trajectories"].drop(columns=["fold", "world_fold"]),
                     d["steps"].drop(columns=["fold", "world_fold"]))
    s.index = [_unit_name(int(k)) for k in s.index]
    s = s.loc[[i for i in s.index if i != UNIT_TEST] + [UNIT_TEST]]
    return s[["templates", "trajectories", "worlds", "steps", "y_traj", "y_scan", "y_gate", "first_h"]]


def fold_balance(d: dict) -> pd.DataFrame:
    """How much the dev folds differ from each other: template folds (the protocol) vs world folds (control)."""
    from agentdrift.splits.grouped import fold_summary

    traj = d["trajectories"].drop(columns=["fold", "world_fold"])
    steps = d["steps"].drop(columns=["fold", "world_fold"])
    out = {}
    for name, col in [("template folds", "fold"), ("world folds", "world_fold")]:
        s = fold_summary(d["folds"], traj, steps, column=col).drop(index=-1, errors="ignore")
        cat = s[[c for c in s.columns if c.startswith("share ") and c.removeprefix("share ") in SOURCE_ORDER]]
        dom = s[[c for c in s.columns if c.startswith("share ") and c.removeprefix("share ") not in SOURCE_ORDER]]
        out[name] = {
            "max gap of a category share to the dev mean (pts)": 100 * float((cat - cat.mean()).abs().max().max()),
            "max gap of a domain share to the dev mean (pts)": 100 * float((dom - dom.mean()).abs().max().max()),
            "y_traj range across folds (pts)": 100 * float(s["y_traj"].max() - s["y_traj"].min()),
            "y_gate range across folds (pts)": 100 * float(s["y_gate"].max() - s["y_gate"].min()),
            "y_scan range across folds (pts)": 100 * float(s["y_scan"].max() - s["y_scan"].min()),
        }
    return pd.DataFrame(out)


def fold_prevalence(d: dict) -> pd.DataFrame:
    """Prevalence per dev fold for the template folds and the world-control folds (long format, for the figure)."""
    from agentdrift.splits.grouped import fold_summary

    traj = d["trajectories"].drop(columns=["fold", "world_fold"])
    steps = d["steps"].drop(columns=["fold", "world_fold"])
    rows = []
    for name, col in [("template folds", "fold"), ("world folds", "world_fold")]:
        s = fold_summary(d["folds"], traj, steps, column=col).drop(index=-1, errors="ignore")
        for k, r in s.iterrows():
            rows += [{"split": name, "fold": int(k), "target": t, "prevalence": float(r[t])}
                     for t in ("y_traj", "y_gate", "y_scan")]
    return pd.DataFrame(rows)


def grouping_checks(d: dict) -> dict:
    """What is and is not shared between folds and between dev and the fixed test set."""
    traj = d["trajectories"]
    dev, test = traj[traj["fold"] != -1], traj[traj["fold"] == -1]
    folds_per_template = dev.groupby("template_id")["fold"].nunique()
    folds_per_world = dev.groupby("world_id")["fold"].nunique()
    return {
        "dev_templates": int(dev["template_id"].nunique()),
        "test_templates": int(test["template_id"].nunique()),
        "templates_in_two_folds_or_more": int((folds_per_template > 1).sum()),
        "templates_shared_dev_test": len(set(dev["template_id"]) & set(test["template_id"])),
        "dev_worlds": int(dev["world_id"].nunique()),
        "worlds_in_two_folds_or_more": int((folds_per_world > 1).sum()),
        "test_trajectories_whose_world_is_in_dev": round(float(test["world_id"].isin(set(dev["world_id"])).mean()), 3),
        "trajectories_per_template_dev_median": float(dev.groupby("template_id").size().median()),
        "trajectories_per_template_test_median": float(test.groupby("template_id").size().median()),
    }


def lodo_table(d: dict) -> pd.DataFrame:
    """Leave-one-domain-out: size of each held-out domain and its prevalence against the four training domains."""
    from agentdrift.splits.grouped import lodo_splits

    traj, steps = d["trajectories"], d["steps"]
    dev_steps = steps[steps["fold"] != -1]
    rows = {}
    for dom, tr, ev in lodo_splits(traj):
        held, rest = traj.iloc[ev], traj.iloc[tr]
        sh, sr = dev_steps[dev_steps["domain"] == dom], dev_steps[dev_steps["domain"] != dom]
        rows[dom] = {
            "templates": held["template_id"].nunique(), "trajectories": len(held),
            "share of dev": len(held) / (len(held) + len(rest)),
            "y_traj held out": held["y_traj"].mean(), "y_traj train": rest["y_traj"].mean(),
            "y_gate held out": sh["y_gate"].mean(), "y_gate train": sr["y_gate"].mean(),
            "share delayed held out": (held["source_category"] == "attacked_delayed").mean(),
            "share delayed train": (rest["source_category"] == "attacked_delayed").mean(),
        }
    return pd.DataFrame(rows).T


def _unit_counts(traj: pd.DataFrame, steps: pd.DataFrame) -> dict:
    scan = steps[steps["t"] >= 1]  # the scanner view has no row at t = 0 (features/views.py)
    out = {
        "templates": traj["template_id"].nunique(), "trajectories": len(traj),
        "negative trajectories": int((traj["y_traj"] == 0).sum()), "first H": int(steps["is_first_h"].sum()),
        "gate +": int(steps["y_gate"].sum()), "gate −": int((steps["y_gate"] == 0).sum()),
        "scan +": int(scan["y_scan"].sum()), "scan −": int((scan["y_scan"] == 0).sum()),
    }
    for name, f in FPR_TARGETS.items():
        out[f"gate FP @ {name}"] = out["gate −"] * f
        out[f"scan FP @ {name}"] = out["scan −"] * f
    return out


def resolution_table(d: dict) -> pd.DataFrame:
    """For every evaluation unit: positives, negatives and the EXPECTED number of false positives at each target
    FPR (negatives x FPR). Few false positives = the observed FPR, and a threshold set at that FPR, are noisy.

    Units: each outer validation fold; the inner validation parts used to set thresholds (median over the
    3 inner parts of the 5 outer training parts); the fixed test set (counts only); each held-out domain."""
    from agentdrift.splits.grouped import inner_splits, lodo_splits

    traj, steps = d["trajectories"], d["steps"]
    rows = {}
    for k in sorted(traj["fold"].unique()):
        if k != -1:
            rows[_unit_name(int(k))] = _unit_counts(traj[traj["fold"] == k], steps[steps["fold"] == k])
    inner = []
    dev = traj[traj["fold"] != -1].reset_index(drop=True)
    for k in sorted(dev["fold"].unique()):
        part = dev[dev["fold"] != k].reset_index(drop=True)
        for _, val in inner_splits(part):
            ids = set(part.loc[val, "traj_id"])
            inner.append(_unit_counts(part.loc[val], steps[steps["traj_id"].isin(ids)]))
    rows["inner validation (median)"] = pd.DataFrame(inner).median().to_dict()
    rows[UNIT_TEST] = _unit_counts(traj[traj["fold"] == -1], steps[steps["fold"] == -1])
    for dom, _, ev in lodo_splits(traj):
        held = traj.iloc[ev]
        rows[f"held-out {dom}"] = _unit_counts(held, steps[steps["traj_id"].isin(set(held["traj_id"]))])
    return pd.DataFrame(rows).T


def reference_scores(d: dict, target: str, seed: int = SEED) -> tuple[pd.DataFrame, dict, dict]:
    """Out-of-fold scores of a deliberately crude, metadata-only reference (NOT a project model): logistic
    regression on one-hot (tool, step index, domain), fitted on the four training folds, scored on the fifth.
    Its only purpose is to measure interval widths at a realistic performance level.

    The tool is the one the view can read: a_t for the gate, a_{t-1} (the call that produced o_{t-1}) for the
    scanner, whose rows start at t = 1. Thresholds at each target FPR are set on the training part's negatives
    (alert when score > threshold), then applied unchanged to the held-out fold.
    The score takes few distinct values (ties), so the FPR reached on the training part can fall below the target:
    it is returned too, to separate the effect of ties from the effect of moving to another fold.
    Returns (scores of dev steps: traj_id, t, template_id, fold, domain, y, score), {fold: {name: threshold}} and
    {fold: {name: FPR reached on the training part}}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    from agentdrift.splits.grouped import outer_splits

    s = d["steps"][d["steps"]["fold"] != -1].sort_values(["traj_id", "t"]).reset_index(drop=True)
    if target == "y_scan":
        s["view_tool"] = s.groupby("traj_id")["tool"].shift(1)
        s = s[s["t"] >= 1].reset_index(drop=True)
    elif target == "y_gate":
        s["view_tool"] = s["tool"]
    else:
        raise ValueError(f"unknown step target {target!r}")
    X = s[["view_tool", "t", "domain"]].astype(str)
    y = s[target].astype(int).to_numpy()
    score = np.full(len(s), np.nan)
    thresholds, reached = {}, {}
    for k, tr, ev in outer_splits(s):
        model = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                              LogisticRegression(max_iter=2000, random_state=seed))
        model.fit(X.iloc[tr], y[tr])
        score[ev] = model.predict_proba(X.iloc[ev])[:, 1]
        neg = np.sort(model.predict_proba(X.iloc[tr][y[tr] == 0])[:, 1])[::-1]
        thresholds[k] = {name: float(neg[int(np.floor(f * len(neg)))]) for name, f in FPR_TARGETS.items()}
        reached[k] = {name: float((neg > thr).mean()) for name, thr in thresholds[k].items()}
    out = s[["traj_id", "t", "template_id", "fold", "domain"]].assign(y=y, score=score)
    return out, thresholds, reached


def interval_widths(ref: pd.DataFrame, thresholds: dict, test_templates: int, n_resamples: int = 2000,
                    level: float = 0.95, seed: int = SEED) -> pd.DataFrame:
    """Per dev fold and metric: point estimate and 95% interval width of the reference score
    - resampling the fold's templates (cluster bootstrap, the protocol);
    - resampling `test_templates` templates from the fold (spread expected on a test set of that size);
    - resampling steps independently (invalid; shows how much the dependence widens the interval).
    design_effect = (cluster width / iid width)^2, at the fold's own size."""
    from agentdrift.evaluation import bootstrap as BS

    rows = []
    for k in sorted(ref["fold"].unique()):
        r = ref[ref["fold"] == k]
        thr = thresholds[k]
        est = BS.point_estimate(r["score"], r["y"], thresholds=thr)
        clus = BS.interval(BS.cluster_bootstrap(r["score"], r["y"], r["template_id"], thr, n_resamples, seed=seed), level)
        proj = BS.interval(BS.cluster_bootstrap(r["score"], r["y"], r["template_id"], thr, n_resamples,
                                                size=test_templates, seed=seed), level)
        iid = BS.interval(BS.iid_bootstrap(r["score"], r["y"], thr, n_resamples, seed=seed), level)
        for m, v in est.items():
            rows.append({"fold": int(k), "templates": r["template_id"].nunique(), "metric": m, "estimate": v,
                         "width (fold, templates)": clus.loc[m, "width"],
                         "width (test size, templates)": proj.loc[m, "width"],
                         "width (fold, steps iid)": iid.loc[m, "width"]})
    out = pd.DataFrame(rows)
    out["design effect"] = (out["width (fold, templates)"] / out["width (fold, steps iid)"]) ** 2
    return out


def interval_summary(widths: pd.DataFrame) -> pd.DataFrame:
    """Mean over the five folds, with the fold-to-fold standard deviation of the estimate."""
    g = widths.groupby("metric", sort=False)
    out = g[["estimate", "width (fold, templates)", "width (test size, templates)", "width (fold, steps iid)",
             "design effect"]].mean()
    out.insert(1, "sd across folds", g["estimate"].std())
    return out


def threshold_transfer(widths: pd.DataFrame, reached: dict) -> pd.DataFrame:
    """Per fold and target FPR: FPR reached on the training folds (ties can keep it below the target) and FPR
    observed on the held-out fold with the same threshold."""
    obs = widths[widths["metric"].str.startswith("fpr@")].pivot(index="fold", columns="metric", values="estimate")
    cols = {}
    for name in FPR_TARGETS:
        cols[(name, "training folds")] = pd.Series({k: v[name] for k, v in reached.items()})
        cols[(name, "held-out fold")] = obs[f"fpr@{name}"]
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["target FPR", "FPR measured on"])
    return out


def compute_protocol_intervals(d: dict, test_templates: int, n_resamples: int, seed: int = SEED) -> dict:
    out = {}
    for target in ("y_gate", "y_scan"):
        ref, thr, reached = reference_scores(d, target, seed)
        w = interval_widths(ref, thr, test_templates, n_resamples, seed=seed)
        out[target] = {"widths": w, "summary": interval_summary(w), "transfer": threshold_transfer(w, reached)}
    return out


# --------------------------------------------------------------------------- part C: synthesis


def _cell(table: dict, row, col):
    """Value of a table saved by `run_eda` as JSON (orient="split")."""
    return table["data"][table["index"].index(row)][table["columns"].index(col)]


def _repro(metrics_a: dict, measure: str):
    """'recomputed' value of a row of the part-A reproduction table, found by its `measure` text."""
    t = metrics_a["reproduction"]
    row = next(r for r in t["data"] if r[t["columns"].index("measure")] == measure)
    return row[t["columns"].index("recomputed")]


# Where each decision of `synthesis_table` stands (same row order; updated by hand, docs/PLAN_DONNEES.md step 9):
# implemented = code and test exist; in progress = ticket open; decided = written in CONTEXT, code pending;
# proposed = new from this EDA, to confirm as a team.
SYNTHESIS_STATUS = [
    "implemented (splits/grouped.py)", "implemented (data/anonymize.py)", "implemented (EDA B3)",
    "implemented", "in progress (L12 test)", "implemented (features/views.py)",
    "in progress (KAN-19, KAN-58)", "implemented (views: include_thought=False)", "decided (CONTEXT 8.2)",
    "decided (CONTEXT 8.2)", "implemented (evaluation/bootstrap.py)", "implemented (evaluation/bootstrap.py)",
    "proposed", "decided (CONTEXT 8.1)",
]


def synthesis_table(metrics_a: dict, metrics_b: dict, rc: dict) -> pd.DataFrame:
    """Finding -> decision -> hypothesis, every number read from the saved outputs of parts A and B
    (results/runs/eda*/metrics.json) or computed in part C. One row per decision of the data phase."""
    sc, vis, purity = metrics_a["shortcuts"], metrics_a["visibility"], metrics_a["world_purity_by_domain"]
    probe = metrics_b["world_probe"]
    g, ci = rc["grouping"], rc["intervals"]
    pur = dict(zip(purity["index"], [r[purity["columns"].index("weighted_majority_share")] for r in purity["data"]]))
    low = min(pur, key=pur.get)
    rest = [v for k, v in pur.items() if k != low]
    vis_any = {c: _cell(vis, c, "any of the three") for c in ("attacked_full", "attacked_partial", "attacked_delayed")}
    gap = metrics_a["gap"]
    delayed = {str(k): v for k, v in zip(gap["columns"], gap["data"][gap["index"].index("attacked_delayed")])}
    n_delayed = sum(delayed.values())
    gate = ci["y_gate"]["summary"]
    res = rc["resolution"]
    prev = metrics_a["prevalence_split"]
    p_all = dict(zip(prev["columns"], prev["data"][prev["index"].index("all")]))
    lodo = rc["lodo"]
    rows = [
        ("Same task template in train and test inflates scores",
         f"{g['templates_in_two_folds_or_more']} templates in two folds, {g['templates_shared_dev_test']} shared "
         "dev / test (after the split)",
         "Split grouped by template; fixed test set of the original task-disjoint split", "protocol", "L1"),
        ("The simulated world gives away the category",
         f"world purity {min(rest):.0%}–{max(rest):.0%} by domain ({low}: {pur[low]:.0%}); world lookup F1 "
         f"{_cell(sc, 'world lookup → y_traj', 'f1'):.3f}",
         "Anonymise world identities (relational placeholders) before any feature", "preprocessing", "L9, H3 (rung 4)"),
        ("Worlds cannot be grouped together with templates",
         f"{g['worlds_in_two_folds_or_more']} of {g['dev_worlds']} dev worlds in two folds or more",
         "Text-only world probe as the control; world-grouped folds as a robustness check", "protocol", "L9"),
        ("Anonymisation removes the world signal from the text",
         f"text world probe {_cell(probe, 'raw', 'world_accuracy'):.1%} → "
         f"{_cell(probe, 'anonymised', 'world_accuracy'):.1%} (chance {_cell(probe, 'raw', 'chance_given_category'):.1%})",
         "Models read the anonymised tables only (`read_*(anonymized=True)`)", "preprocessing", "L9"),
        ("Position alone predicts the targets",
         f"ROC-AUC {_cell(sc, 'position only → y_scan', 'roc_auc'):.3f} (scan), "
         f"{_cell(sc, 'position only → y_gate', 'roc_auc'):.3f} (gate)",
         "Report the position-only baseline as the floor; step index and history length never features", "protocol",
         "L12, H3"),
        ("Total length leaks the future",
         f"trajectory length alone ROC-AUC {_cell(sc, 'trajectory length → y_traj', 'roc_auc'):.2f} for y_traj",
         "Causal views: nothing after t (features/views.py)", "preprocessing", "L3, L11"),
        ("Injections are readable for full attacks, often not for partial / delayed ones",
         f"any indicator visible: full {vis_any['attacked_full']:.0%}, partial {vis_any['attacked_partial']:.0%}, "
         f"delayed {vis_any['attacked_delayed']:.0%}",
         "Mask generator artefacts (generic upper-case label rule, policy codes, amounts); recall by attack type "
         "and by visibility", "preprocessing + metric", "H3, H2, H8"),
        ("The generated thought confesses the attack",
         f"confession words in thought: {_repro(metrics_a, 'confession words in thought, H partial'):.0%} of H partial",
         "Variant without `thought` is the reference; with `thought` reported as an ablation rung", "preprocessing",
         "H3 (rung 3)"),
        ("Imbalanced targets; the step to stop is the first hijacked one",
         f"y_scan {p_all['y_scan']:.1%}, y_gate {p_all['y_gate']:.1%} of steps; {int(p_all['first_h'])} first H "
         f"({p_all['first_h'] / p_all['steps']:.1%} of steps)",
         "PR-AUC as the main metric; class weights; business metrics at the first alert", "metric", "—"),
        ("The scanner can only anticipate on delayed attacks",
         f"injection → hijack gap ≥ 2 steps only for delayed attacks ({delayed.get('2', 0) / n_delayed:.0%} at "
         "exactly 2)",
         "Detection lead reported on delayed attacks only", "metric", "H2, H6"),
        ("The fixed test set is small",
         f"{g['test_templates']} templates; expected 95% interval width on it: PR-AUC ±"
         f"{gate.loc['pr_auc', 'width (test size, templates)'] / 2:.3f}, recall at 1% FPR ±"
         f"{gate.loc['recall@1%', 'width (test size, templates)'] / 2:.3f} (reference score, gate)",
         "Report every test result with a template-cluster bootstrap interval; model choice on the 5 dev folds",
         "protocol", "L5"),
        ("Steps are not independent",
         f"design effect {gate['design effect'].median():.1f} (median over gate metrics): a step-level bootstrap "
         "would understate the interval",
         "Resample templates, never steps; paired bootstrap for model comparisons", "protocol", "—"),
        ("Very low FPR targets are hardly measurable",
         f"expected false positives at 0.2% FPR: {res.loc[UNIT_TEST, 'gate FP @ 0.2%']:.0f} on the test set, "
         f"{res.loc['inner validation (median)', 'gate FP @ 0.2%']:.0f} on an inner validation part",
         "Keep 0.2% as a secondary operating point, reported with its observed FPR", "metric", "H5"),
        ("Domains differ in prevalence and attack mix",
         f"held-out y_traj {lodo['y_traj held out'].min():.0%}–{lodo['y_traj held out'].max():.0%}; delayed share "
         f"{lodo['share delayed held out'].min():.0%}–{lodo['share delayed held out'].max():.0%}",
         "Leave-one-domain-out, with the threshold transferred from the training domains", "protocol", "H1, H5"),
    ]
    out = pd.DataFrame(rows, columns=["finding", "evidence", "decision", "kind", "hypothesis / leak"])
    out["status"] = SYNTHESIS_STATUS
    return out
