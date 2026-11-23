"""World anonymisation (leak L9, JOURNAL 19/11): remove the identity of the simulated world, keep its relations.

    python -m agentdrift.data.anonymize     # data/interim/{trajectories,steps}_anon.parquet + residual check

What is replaced, in `task`, `thought`, `args_json` and `obs`:
- people of the world: user -> <USER>, contact k -> <CONTACT_k> (full name, and first or last name alone when
  that token identifies a single person of the world);
- the company name and the label of the user's e-mail domain (e.g. "bankofafrica") -> <COMPANY>;
- e-mail addresses: user -> <USER_EMAIL>, contact k -> <CONTACT_k_EMAIL>, any other address -> <INT_EMAIL_k> if it
  shares the user's domain, else <EXT_EMAIL_k>;
- URLs -> <INT_URL_k> / <EXT_URL_k>; bare domains ("financeco.com") -> <INT_DOMAIN_k> / <EXT_DOMAIN_k>.

Relations are kept (known contact vs unknown address, internal vs external, same address = same placeholder),
identities are removed. The 263 world addresses with a space are repaired first.

Properties (tested in tests/test_anonymize.py):
- deterministic and row-wise: nothing is fitted on the corpus, so applying it before the split is not a
  preprocessing leak (INSTRUCTIONS rule 2);
- causal: counters k are assigned by first appearance in temporal order (task, then thought -> args -> obs of
  step 0, 1, ...), so the text of step t never depends on steps after t (L3);
- generic: rules use the world object and regular expressions only, never a list of strings mined from the corpus.
"""

from __future__ import annotations

import json
import re
import sys

import pandas as pd

from agentdrift.config import INTERIM, ROOT
from agentdrift.data.load import read_steps, read_trajectories
from agentdrift.data.schema import STEPS, TRAJECTORIES

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s'\"\\<>),\]}]+", re.IGNORECASE)
TLDS = (
    "com|net|org|io|co|ai|edu|gov|info|biz|inc|jp|uk|us|de|fr|ca|au|in|cn|eu|health|bank|app|dev|tech|me|ly|tv|cloud"
)
# "@" is allowed before a bare domain: e-mails are replaced first, so "@domain.com" left in the text has no
# local part (e.g. "addresses typically end with @greencorp.com").
DOMAIN_RE = re.compile(rf"(?<![\w.<-])((?:[a-z0-9-]+\.)+(?:{TLDS}))(?![\w-])", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"<[A-Z_0-9]+>")
TEXT_COLUMNS = ("thought", "args_json", "obs")


def _clean_email(e: str) -> str:
    return e.replace(" ", "").lower()


def _domain(e: str) -> str:
    return _clean_email(e).rsplit("@", 1)[-1]


def _host(url: str) -> str:
    h = re.sub(r"^(?:https?://)?(?:www\.)?", "", url, flags=re.IGNORECASE)
    return re.split(r"[/?#:]", h, maxsplit=1)[0].lower()


def _word(s: str, flags: int = 0) -> re.Pattern:
    return re.compile(rf"(?<![\w<]){re.escape(s)}(?![\w>])", flags)


class WorldAnonymizer:
    """Stateful anonymiser for ONE trajectory: call `apply` on texts in temporal order."""

    def __init__(self, world: dict):
        self.user_domain = _domain(world["email"])
        self.raw_spaced = [e for e in [world["email"]] + [c["email"] for c in world["contacts"]] if " " in e]
        self.known_emails = {_clean_email(world["email"]): "<USER_EMAIL>"}
        people = [("<USER>", world["user"])]
        for k, c in enumerate(world["contacts"], start=1):
            self.known_emails.setdefault(_clean_email(c["email"]), f"<CONTACT_{k}_EMAIL>")
            people.append((f"<CONTACT_{k}>", c["name"]))

        # Names: full names first, then first / last tokens that identify a single person of this world.
        firsts = [n.split()[0] for _, n in people]
        lasts = [n.split()[-1] for _, n in people]
        name_rules = [(n, ph) for ph, n in people]
        for (ph, n), f, last in zip(people, firsts, lasts):
            if firsts.count(f) == 1 and len(f) >= 3:
                name_rules.append((f, ph))
            if lasts.count(last) == 1 and len(last) >= 3 and last != f:
                name_rules.append((last, ph))
        name_rules.sort(key=lambda r: -len(r[0]))
        self.name_rules = [(_word(n), ph) for n, ph in name_rules]

        self.company_rules = [(_word(world["company"], re.IGNORECASE), "<COMPANY>")]
        label = self.user_domain.split(".")[0]
        if len(label) >= 4:
            self.company_rules.append((re.compile(rf"(?<![A-Za-z<]){re.escape(label)}(?![A-Za-z>])", re.IGNORECASE),
                                       "<COMPANY>"))
        self.maps: dict[str, dict[str, str]] = {"EMAIL": {}, "URL": {}, "DOMAIN": {}}
        self.counters = {f"{side}_{kind}": 0 for side in ("INT", "EXT") for kind in ("EMAIL", "URL", "DOMAIN")}

    def _internal(self, host: str) -> bool:
        return host == self.user_domain or host.endswith("." + self.user_domain)

    def _placeholder(self, kind: str, key: str, host: str) -> str:
        m = self.maps[kind]
        if key not in m:
            side = "INT" if self._internal(host) else "EXT"
            self.counters[f"{side}_{kind}"] += 1
            m[key] = f"<{side}_{kind}_{self.counters[f'{side}_{kind}']}>"
        return m[key]

    def _email(self, match: re.Match) -> str:
        e = _clean_email(match.group(0))
        return self.known_emails.get(e) or self._placeholder("EMAIL", e, _domain(e))

    def _url(self, match: re.Match) -> str:
        url = match.group(0).rstrip(".;:!?")
        tail = match.group(0)[len(url):]
        return self._placeholder("URL", url.lower(), _host(url)) + tail

    def _bare_domain(self, match: re.Match) -> str:
        d = match.group(1).lower()
        return self._placeholder("DOMAIN", d, d)

    def apply(self, text: str) -> str:
        for raw in self.raw_spaced:
            text = text.replace(raw, raw.replace(" ", ""))
        text = URL_RE.sub(self._url, text)
        text = EMAIL_RE.sub(self._email, text)
        text = DOMAIN_RE.sub(self._bare_domain, text)
        for pattern, ph in self.name_rules:
            text = pattern.sub(ph, text)
        for pattern, ph in self.company_rules:
            text = pattern.sub(ph, text)
        return text


def anonymize_trajectory(world: dict, task: str, steps: list[dict]) -> tuple[str, list[dict]]:
    """Anonymise one trajectory. `steps` items need `thought`, `args_json`, `obs`; other keys are kept."""
    a = WorldAnonymizer(world)
    task_anon = a.apply(task)
    out = []
    for s in steps:
        out.append({**s, **{c: a.apply(s[c]) for c in TEXT_COLUMNS}})
    return task_anon, out


def anonymize_tables(traj: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anonymise the interim tables (trajectory order preserved; `world_json` dropped from the output)."""
    worlds = traj.set_index("traj_id")["world_json"].map(json.loads)
    tasks = traj.set_index("traj_id")["task"]
    s = steps.sort_values(["traj_id", "t"])
    new_task, new_text = {}, {c: [] for c in TEXT_COLUMNS}
    index = []
    for traj_id, g in s.groupby("traj_id", sort=False):
        records = g[list(TEXT_COLUMNS)].to_dict("records")
        task_anon, recs = anonymize_trajectory(worlds[traj_id], tasks[traj_id], records)
        new_task[traj_id] = task_anon
        for c in TEXT_COLUMNS:
            new_text[c].extend(r[c] for r in recs)
        index.extend(g.index)
    steps_anon = steps.copy()
    for c in TEXT_COLUMNS:
        steps_anon.loc[index, c] = new_text[c]
    traj_anon = traj.drop(columns="world_json").assign(task=traj["traj_id"].map(new_task))
    return traj_anon, steps_anon


def residual_identifiers(traj_raw: pd.DataFrame, traj_anon: pd.DataFrame, steps_anon: pd.DataFrame) -> pd.Series:
    """Number of trajectories whose anonymised text still contains each kind of world identifier."""
    text = (steps_anon["thought"] + " " + steps_anon["args_json"] + " " + steps_anon["obs"]).groupby(steps_anon["traj_id"]).agg(" ".join)
    text = traj_anon.set_index("traj_id")["task"] + " " + text.reindex(traj_anon["traj_id"]).to_numpy()
    counts = dict.fromkeys(["full name", "world e-mail", "company name", "domain label"], 0)
    for traj_id, wj in zip(traj_raw["traj_id"], traj_raw["world_json"]):
        w, x = json.loads(wj), text[traj_id]
        names = [w["user"]] + [c["name"] for c in w["contacts"]]
        counts["full name"] += any(_word(n).search(x) for n in names)
        xl = x.lower()
        counts["world e-mail"] += any(_clean_email(e) in xl for e in [w["email"]] + [c["email"] for c in w["contacts"]])
        counts["company name"] += bool(_word(w["company"], re.IGNORECASE).search(x))
        label = _domain(w["email"]).split(".")[0]
        counts["domain label"] += len(label) >= 4 and label in xl
    return pd.Series(counts, name="trajectories with residual")


def placeholder_counts(steps_anon: pd.DataFrame) -> pd.Series:
    text = steps_anon["thought"] + " " + steps_anon["args_json"] + " " + steps_anon["obs"]
    found = text.str.findall(PLACEHOLDER_RE).explode().dropna()
    kind = found.str.replace(r"_\d+", "_k", regex=True)
    return kind.value_counts()


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    traj = read_trajectories(include_test=True)
    steps = read_steps(include_test=True)
    traj_anon, steps_anon = anonymize_tables(traj, steps)
    TRAJECTORIES.remove_columns(["world_json"]).validate(traj_anon, lazy=True)
    STEPS.validate(steps_anon, lazy=True)
    bad_json = [a for a in steps_anon["args_json"] if not _is_json(a)]
    if bad_json:
        raise ValueError(f"{len(bad_json)} args_json no longer valid JSON after anonymisation, e.g. {bad_json[0][:120]}")
    return traj_anon, steps_anon


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
    except ValueError:
        return False
    return True


def main() -> int:
    traj_anon, steps_anon = build()
    traj_anon.to_parquet(INTERIM / "trajectories_anon.parquet", index=False)
    steps_anon.to_parquet(INTERIM / "steps_anon.parquet", index=False)
    print(f"Wrote {INTERIM.relative_to(ROOT)}/{{trajectories,steps}}_anon.parquet")
    print(residual_identifiers(read_trajectories(include_test=True), traj_anon, steps_anon).to_string())
    print(placeholder_counts(steps_anon).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
