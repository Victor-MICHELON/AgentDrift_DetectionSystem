# PROMPT_LOG.md — Critical AI Interactions

> The 4–6 most important LLM interactions of the project, one or more per student.
> Each entry documents a **high-stakes decision** (leakage prevention, statistical test selection, model or metric choice), the **iteration from a bare prompt to a constrained prompt**, what the AI got wrong or left out, and how we verified the final answer.
> Errors found during these interactions are detailed in [AI_AUDIT.md](AI_AUDIT.md) and cross-referenced here.

## Index

| # | Date | Author | Decision | Project step | Version | Linked audit |
|---|---|---|---|---|---|---|
| P1 | | | | | | |
| P2 | | | | | | |
| P3 | | | | | | |
| P4 | | | | | | |
| P5 | | | | | | |
| P6 | | | | | | |

<!--
Planned high-stakes interactions (delete this comment once filled):
- Grouped split design and leakage tests (task templates, future steps, fitting inside folds)       -> Data & protocol owner
- Feature engineering under the "observation x_t only" constraint + artifact masking (H1, H3)      -> Features & AutoML owner
- Metric choice under imbalance and early-detection metrics (PR-AUC, recall@FPR, stop-before-action) -> Decision & evaluation owner
- Statistical comparison of models with dependent steps (cluster bootstrap, paired tests, DeLong validity) -> Decision & evaluation owner
- AutoML search space and nested hyperparameter tuning                                             -> Features & AutoML owner
- Out-of-course model choice and fine-tuning protocol (DeBERTa-v3), with its reference paper       -> Advanced models owner
-->

---

## Entry template (copy for each entry)

## P# — <Decision in one line>

| Field | Value |
|---|---|
| **Author** | |
| **Date** | YYYY-MM-DD |
| **Tool / model** | |
| **Project step** | Step 2.x / Step 3.x / Phase 4 |
| **Pipeline version** | v0.x (git commit / tag) |
| **Files affected** | `src/agentdrift/...` |
| **Why this is high-stakes** | What goes wrong for the project if the AI answer is wrong |

### 1. Iteration 0 — Bare prompt

```text
<exact prompt as first typed>
```

**What the answer got wrong or left out:** <specific flaws: leakage, wrong metric, unverified assumption, invented API...>

### 2. Iteration 1..n — Intermediate refinements (optional)

| Iteration | What we added to the prompt | Why (flaw it targeted) |
|---|---|---|
| 1 | | |

### 3. Final constrained prompt

```text
# ROLE
You are a <precise expertise, e.g. senior ML engineer specialised in evaluating imbalanced, grouped, sequential classifiers>.

# CONTEXT
Project: early guardrail detecting indirect prompt injection in AI agent trajectories (AgentDrift dataset).
Current step: <...>. Pipeline version: <...>. What already exists: <modules, results>.

# TASK
<One precise sentence: what to produce.>

# OBJECTIVE AND SUCCESS CRITERIA
<Why we need it; measurable criteria the answer must meet.>

# DATA SCHEMA
- Unit of observation: one agent step t, scored BEFORE the proposed tool call a_t is executed.
- Columns: <name: type, meaning> ...
- Group key: <task template id>. Domain key: <domain>. Target: <y_scan / y_gate / y_traj>.
- Prevalence: <positive rate per step / per trajectory>.

# LEAKAGE CONSTRAINTS (hard rules)
- Never use information from steps > t (no trajectory length, no trajectory label, no future tool calls).
- Every fitted transform (TF-IDF, scaler, imputer, encoder, feature selection, resampling) lives inside a scikit-learn Pipeline fitted on the training fold only.
- Splits are grouped by task template; no template in two folds.
- Thresholds, calibration and hyperparameters are chosen on validation folds, never on the test set.
- <task-specific rules>

# STATISTICAL ASSUMPTIONS
- Steps within a trajectory are NOT independent: any CI or test must resample tasks (cluster level).
- <assumptions to check, how, and what to do if violated>

# TECHNICAL CONSTRAINTS
- Python 3.12, library versions from requirements.txt: <relevant ones>. Use only APIs that exist in these versions.
- Fixed seed via agentdrift.utils.seed.set_seed. Relative paths via agentdrift.config.
- <style: functions with docstrings, no notebook logic, type hints>

# OUTPUT FORMAT
<exact structure: code block + table + justification of each choice, etc.>

# VERIFICATION AND FALLBACK TESTS
- The answer must include tests that fail if <leakage condition>.
- If <assumption> does not hold, propose <fallback> instead.

# SELF-CRITIQUE
Before answering, list the assumptions you made that we cannot verify from this prompt, the main risks of your solution, and anything in this prompt you think is wrong.
```

### 4. AI answer (summary)

<Key points of the answer. Link to the resulting commit.>

### 5. Our audit of the answer

| Check | Method | Result |
|---|---|---|
| Leakage | <test in tests/, manual review> | pass / fail → AI_AUDIT A# |
| APIs exist in pinned versions | <run, docs> | |
| Statistical validity | <assumption check> | |
| Result sanity | <comparison with expected value or baseline> | |

**Accepted:** <what we kept>
**Rejected or modified:** <what we changed and why>
**Linked audit entry:** A#
