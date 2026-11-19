# AI_AUDIT.md — Catch & Correct Report

> Concrete flaws in AI-generated Machine Learning logic that we caught, with root cause, fix and **verified** evidence that the fix works.
> Target: 6–8 substantive flaws (minimum 4), at least one per student. Only real flaws, recorded when they happen.

## Summary

| # | Date | Caught by | Category | Severity | File / step | Found via | Fixed in |
|---|---|---|---|---|---|---|---|
| A1 | | | | | | | |
| A2 | | | | | | | |
| A3 | | | | | | | |
| A4 | | | | | | | |

**Categories:** preprocessing leakage · split / group leakage · temporal leakage (future steps) · shortcut / artifact feature · wrong metric under imbalance · threshold or calibration chosen on test · violated statistical assumption · invented API or wrong library behaviour · hallucinated reference or number · other.
**Severity:** critical (invalidates results) · major (biases results) · minor (no effect on reported numbers).

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
