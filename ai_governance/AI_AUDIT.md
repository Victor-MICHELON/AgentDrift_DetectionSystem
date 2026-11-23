# AI_AUDIT.md — Catch & Correct Report

> Concrete flaws in AI-generated Machine Learning logic that we caught, with root cause, fix and **verified** evidence that the fix works.
> Target: 6–8 substantive flaws (minimum 4), at least one per student. Only real flaws, recorded when they happen.
> Entries are written independently of [PROMPT_LOG.md](PROMPT_LOG.md): add one whenever a significant flaw or hallucination is documentable (exact output, reason, fix, evidence), whether or not it came from a logged prompt. "Linked prompt" is optional.

## Summary

Only the flaws of **major** severity are kept (they biased or would have biased results); minor ones were dropped.

| # | Date | Caught by | Category | Severity | File / step | Found via | Fixed in |
|---|---|---|---|---|---|---|---|
| A1 | 2026-11-20 | Claude Code (own check). To confirm: Victor MICHELON | wrong metric (leak control) | major | `docs/PLAN_DONNEES.md` steps 5 and 7, CONTEXT 8.1 | plan review, then run on both versions | `c2cb185` (code), CONTEXT v1.2 (21/11) |
| A2 | 2026-11-20 | Claude Code (own check). To confirm: Victor MICHELON | wrong baseline / statistical reference | major | `data/audit.py::world_probe` | suspicious ratio to chance | `c2cb185` |
| A3 | 2026-11-20 | Claude Code (own check). To confirm: Victor MICHELON | other (regex corrupts data, leaves identity) | major | `data/anonymize.py` | JSON-validity and residual checks | `c2cb185` |
| A4 | 2026-11-20 | Claude Code (own check). To confirm: Victor MICHELON | hallucinated number (origin to confirm) | major if AI-made | CONTEXT 5 and 6.2 | recomputation in EDA | CONTEXT v1.2 (21/11) |

**Categories:** preprocessing leakage · split / group leakage · temporal leakage (future steps) · shortcut / artifact feature · wrong metric under imbalance · threshold or calibration chosen on test · violated statistical assumption · invented API or wrong library behaviour · hallucinated reference or number · other.
**Severity:** critical (invalidates results) · major (biases results) · minor (no effect on reported numbers).

---

## A1 — Success criterion of the anonymisation that cannot measure anything

| Field | Value |
|---|---|
| **Caught by** | Claude Code (own check), to confirm: Victor MICHELON |
| **Date** | 2026-11-20 |
| **Category / severity** | wrong metric (leak control) / major |
| **Linked prompt** | P1 (context) |
| **Files** | `docs/PLAN_DONNEES.md` (steps 5 and 7), `docs/CONTEXT.md` 8.1, `src/agentdrift/data/audit.py` |
| **Version before → after** | plan written 2026-11-20 (uncommitted wording) → `c2cb185` (probe + tests) |

### What the AI produced

```text
# PLAN_DONNEES.md, step 7 (and CONTEXT.md 8.1, "contrôle complémentaire")
- Baseline « monde seul » avant / après : doit retomber au hasard.
```

### Why it is wrong

The "world lookup" baseline predicts a trajectory's class from `world_id`, a hash of the raw world object stored as a metadata column. Anonymising the **text** never changes `world_id`, so the baseline gives the same score before and after. A control that cannot move cannot show that the leak (L9) is gone, and it would have given false assurance (or a permanent false alarm).

### Root cause

The criterion was copied from CONTEXT.md without asking what input the baseline reads. "World identity in the text" and "world id as a metadata column" were conflated.

### How we caught it

While preparing step 5 ("what does this baseline read, and what does the anonymisation change?"), then by running it on both versions of the data.

### Fix

Three criteria instead of one (JOURNAL 2026-11-20): no residual world identifier in the text (`residual_identifiers`), a text probe that tries to recover the world (`audit.world_probe`: 1-nearest-neighbour on TF-IDF fitted on train, scored on val), and unchanged provenance rates (`audit.provenance_before_after`).

### Evidence that the fix works

| Metric / test | Before | After |
|---|---|---|
| World lookup on `world_id` (F1 / ROC-AUC, fitted on train, scored on val) | 0.838 / 0.961 | 0.838 / 0.961 (identical: cannot measure the effect) |
| Text probe: world recovered from the text (val) | 12.2 % | 3.1 % (chance 2.1 %, see A2) |
| Trajectories with a residual full name, world e-mail or company name | not applicable | 0 / 12,536 |

Guard: `tests/test_anonymize.py::test_corpus_has_no_world_identity_left`.

### Prevention

Every leakage control must state the exact input it reads and be shown to change when the leak is removed. Check a control on a known positive and a known negative before trusting it.

---

## A2 — Wrong chance level for the world probe

| Field | Value |
|---|---|
| **Caught by** | Claude Code (own check), to confirm: Victor MICHELON |
| **Date** | 2026-11-20 |
| **Category / severity** | wrong baseline / statistical reference / major (it would have reversed the conclusion) |
| **Linked prompt** | none |
| **Files** | `src/agentdrift/data/audit.py` (`world_probe`) |
| **Version before → after** | never committed → `c2cb185` |

### What the AI produced

```python
n_worlds_dom = world[tr_ids].groupby(domain[tr_ids]).nunique()
chance = float((1 / domain[ev_ids].map(n_worlds_dom)).mean())     # 0.0074
```

### Why it is wrong

Worlds are tied to a generation job (in four domains a world holds one source category), so a neighbour with similar content is usually of the same category and therefore one of far fewer worlds. The fair reference is one over the number of train worlds of the same domain **and** category. Against 0.74 %, an anonymised accuracy of 3.1 % reads as four times chance ("anonymisation failed"), although the probe fell from 12.2 % to 3.1 %.

### Root cause

The structure documented in the dataset README, and measured in EDA A8 itself (world purity), was not used when choosing the reference.

### How we caught it

The probe also printed how often the nearest neighbour has the same source category (0.768 raw, 0.791 anonymised), which exposed the coupling.

### Fix

```python
key = t["domain"] + "|" + t["source_category"]
n_worlds = t.loc[tr_ids].groupby(key[tr_ids])["world_id"].nunique()
chance_given_category = (1 / key[ev_ids].map(n_worlds)).mean()    # 0.0207
```

Both references are returned (`chance_given_domain`, `chance_given_category`); the criterion uses the second.

### Evidence that the fix works

| Metric / test | Before | After |
|---|---|---|
| Chance level | 0.0074 | 0.0207 |
| Probe accuracy raw → anonymised | 12.2 % → 3.1 % | 12.2 % → 3.1 % (now 1.5 times chance instead of 4 times) |

### Prevention

When a reference level is computed, check it against the dependency structure of the data (here world and category are coupled), not only the dimensions of one column.

### Disclosed limit

The first version of the probe also returned the ROC-AUC of `y_traj` obtained from the neighbour's world: 0.806 on raw text and 0.810 on anonymised text (unchanged). It was removed from the final function because it measures similarity to same-class neighbours rather than identity, but it shows that the text still carries class information after anonymisation. This is expected (that is the signal to detect), yet it cannot separate legitimate content from generator style. That separation is the job of the H3 masking tests (open).

---

## A3 — Anonymiser regexes that corrupt JSON and leave a domain behind

| Field | Value |
|---|---|
| **Caught by** | Claude Code (own check), to confirm: Victor MICHELON |
| **Date** | 2026-11-20 |
| **Category / severity** | other (preprocessing corrupts data, leaves identity) / major |
| **Linked prompt** | none |
| **Files** | `src/agentdrift/data/anonymize.py` |
| **Version before → after** | never committed → `c2cb185` |

### What the AI produced

```python
URL_RE = re.compile(r"(?:https?://|www\.)[^\s'\"<>),\]}]+", re.IGNORECASE)
DOMAIN_RE = re.compile(rf"(?<![\w@.<-])((?:[a-z0-9-]+\.)+(?:{TLDS}))(?![\w-])", re.IGNORECASE)
```

### Why it is wrong

1. The URL pattern did not stop at the backslash that escapes inner quotes in `args_json`, so it swallowed `\"` and produced invalid JSON in 2 steps, e.g. `{"content":"{\"payment_gateway\": \"<EXT_URL_1>"}","file":"payment_config.json"}`.
2. The domain pattern refused a leading `@`, so text such as "@greencorpinvoice.com" or "@biogenhealth.com" (an address with no local part) kept the identity of a domain, defeating the anonymisation it was written for.

### Root cause

A generic "URL until whitespace or punctuation" pattern applied to text that embeds JSON, and a lookbehind written for e-mails already replaced, without testing the leftover `@domain` case.

### How we caught it

Two checks that were written into the build: every `args_json` must stay valid JSON, and a residual-identifier scan after anonymisation.

### Fix

```python
URL_RE = re.compile(r"(?:https?://|www\.)[^\s'\"\\<>),\]}]+", re.IGNORECASE)          # stops at "\"
DOMAIN_RE = re.compile(rf"(?<![\w.<-])((?:[a-z0-9-]+\.)+(?:{TLDS}))(?![\w-])", re.IGNORECASE)  # "@" allowed
```

### Evidence that the fix works

| Metric / test | Before | After |
|---|---|---|
| `args_json` invalid after anonymisation | 2 (build raised ValueError) | 0 |
| Trajectories with a residual domain label | 5 | 3 (substring false positives: "unitedhealthcare", a password, a class name) |

Guards: `test_json_stays_valid_with_escaped_quotes`, `test_urls_and_bare_domains` in `tests/test_anonymize.py`.

### Prevention

Any text transform applied to fields that embed JSON is followed by a JSON-validity assertion.

---

## A4 — Exploratory numbers of 2026-11-19 not reproducible (origin to confirm)

| Field | Value |
|---|---|
| **Caught by** | Claude Code (own check), to confirm: Victor MICHELON |
| **Date** | 2026-11-20 (numbers dated 2026-11-19) |
| **Category / severity** | hallucinated number / major if produced by an LLM script |
| **Linked prompt** | none |
| **Files** | `docs/CONTEXT.md` 5 and 6.2, `notebooks/01_eda.ipynb` (reproduction table) |
| **Version before → after** | `docs/CONTEXT.md` v1.1 → not updated yet |

> **To validate before keeping this entry.** I cannot tell from the repository whether the 2026-11-19 figures were produced by an LLM-written script or by hand. If they were not AI-made, delete this entry; if they were, keep it.

### What the AI produced

Figures quoted in CONTEXT.md 5 and 6.2 as "mesures exploratoires", recomputed in the EDA without adjusting any definition:

| Measure | CONTEXT.md | Recomputed |
|---|---|---|
| Position only, ROC-AUC on `y_scan` | 0.52 | 0.876 (fitted on train, scored on val) |
| Position only, ROC-AUC on `y_gate` | 0.754 | 0.779 |
| Confession words in `thought`, H partial | 44.5 % | 75.3 % |
| Instruction vocabulary at injection, partial | 4.7 % | 35.0 % |
| Argument from a previous observation, all H | 32 % | 46.1 % |
| Mean length, attacked trajectories | 6.07 | 5.93 (6.07 is the full attacks only) |
| World lookup F1 | 0.77 | 0.838 on val |

### Why it is wrong

The two position figures matter most: with injections concentrated at one step, a position-only score reaches 0.876 on `y_scan`, not chance. The reference "floor to beat" for the scanner was therefore underestimated by a wide margin.

### Root cause

Not determined: the definitions behind the 2026-11-19 figures were not stored. The recomputation uses documented regexes (`data/audit.py`).

### How we caught it

The EDA reproduction table (`audit.reproduction_table`), by design not tuned to match.

### Fix

Recomputed values recorded in JOURNAL (2026-11-20). `docs/CONTEXT.md` 5 and 6.2 still carry the old figures: to update.

### Evidence that the fix works

| Metric / test | Before | After |
|---|---|---|
| Measures with a stored, reproducible definition | none recorded | 22 measures in `results/runs/eda/reproduction.csv` (written by `run_eda`) |

### Prevention

Numbers enter CONTEXT.md only with the command that reproduces them.

---

## Entry template (copy for each entry)

## A# — <Flaw in one line>

| Field | Value |
|---|---|
| **Caught by** | |
| **Date** | YYYY-MM-DD |
| **Category / severity** | |
| **Linked prompt** | P# (if any) |
| **Files** | `src/agentdrift/...` |
| **Version before → after** | commit / tag → commit / tag |

### What the AI produced

```python
# exact flawed snippet
```

### Why it is wrong

<The ML or statistical reasoning, not only "it was a bug".>

### Root cause

<Why the AI produced it: missing constraint in the prompt, default library behaviour, generic pattern applied to grouped / sequential data...>

### How we caught it

<Code review, leakage test, suspicious metric, assumption check, documentation lookup...>

### Fix

```python
# corrected snippet
```

### Evidence that the fix works

| Metric / test | Before | After |
|---|---|---|
| | | |

<Test name in tests/ that now guards against this flaw.>

### Prevention

<What we added to our prompts, tests or checklist so it does not happen again.>
