# PROMPT_LOG.md — Critical AI Interactions

> The 4–6 most important LLM interactions of the project, one or more per student.
> Each entry keeps the **final robust prompt** used for a high-stakes task (leakage prevention, statistical test selection, model or metric choice), a short summary of what the AI produced, and how we checked it.
> We do not document the iteration from a bare prompt to the final one. Errors found are recorded independently in [AI_AUDIT.md](AI_AUDIT.md) and linked here only when relevant.

## Index

| # | Date | Author | Decision | Project step | Linked audit |
|---|---|---|---|---|---|
| P1 | 2026-11-20 | Victor MICHELON | Scope and leakage rules of the exploratory data analysis | Step 2.1 | — |
| P2 | | | | | |
| P3 | | | | | |
| P4 | | | | | |
| P5 | | | | | |
| P6 | | | | | |

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
| **Author / date** | |
| **Tool / model** | |
| **Project step / files** | Step 2.x, `src/agentdrift/...` |
| **Why high-stakes** | What goes wrong if the AI answer is wrong |

### Final prompt

```text
# ROLE
<precise expertise>

# CONTEXT
<project, current step, what already exists>

# TASK
<one precise sentence>

# DATA SCHEMA
<unit of observation, columns, group key, target, prevalence>

# LEAKAGE CONSTRAINTS (hard rules)
- Never use information from steps > t.
- Every fitted transform lives inside a Pipeline fitted on the training fold only.
- Splits grouped by task template; thresholds and hyperparameters chosen on validation, never on test.

# STATISTICAL AND TECHNICAL CONSTRAINTS
<dependence between steps, pinned versions, seed, style>

# OUTPUT FORMAT AND VERIFICATION
<expected structure, tests that must fail on leakage>

# SELF-CRITIQUE
List unverifiable assumptions, main risks, and anything wrong in this prompt.
```

### Result and check

<2–5 lines: what was produced, what we verified, what we changed. Link to the commit and to an A# entry if a real flaw was found.>

---

## P1 — Scope and leakage rules of the exploratory data analysis

| Field | Value |
|---|---|
| **Author / date** | Victor MICHELON, 2026-11-20 |
| **Tool / model** | Claude Code (Claude Opus 5.5 / Sonnet 5.5) |
| **Project step / files** | Step 2.1; `src/agentdrift/data/audit.py`, `src/agentdrift/viz/plots.py`, `src/agentdrift/pipelines/run_eda.py`, `notebooks/01_eda.ipynb` |
| **Why high-stakes** | The EDA justifies every preprocessing decision (anonymisation, masking, excluded columns). If it peeks at the fixed test set, scores shortcut baselines on their own training data, or presents generator artefacts as signal, preprocessing and hypotheses H1–H3 rest on biased evidence. |

### Final prompt

```text
# ROLE
You are a senior data scientist specialised in auditing synthetic, grouped, sequential datasets for
supervised security classifiers, and in spotting dataset artefacts and leakage before any modelling.

# CONTEXT
Project: early guardrail detecting indirect prompt injection in AI agent trajectories (AgentDrift,
12,536 synthetic trajectories, 71,024 labelled steps, 5 domains, 250 task templates, generator
Llama-3.3-70B). Current step: Step 2.1, part A of the EDA (raw data, before anonymisation).
Pipeline version: v0.0. Already existing: pinned download with checksums
(agentdrift.data.download), interim tables validated by pandera (agentdrift.data.load / schema):
trajectories.parquet, steps.parquet, templates.parquet. External test set: ATBench (1,000
trajectories, trajectory-level labels only).

# TASK
Produce the computations (src/agentdrift/data/audit.py), the figures (src/agentdrift/viz/plots.py),
a headless runner (src/agentdrift/pipelines/run_eda.py) and a presentable English notebook
(notebooks/01_eda.ipynb) covering sections A0–A10 of docs/PLAN_DONNEES.md.

# OBJECTIVE AND SUCCESS CRITERIA
Each section answers one question with one figure and a table, then states key findings and the
consequence for the project (preprocessing decision, leakage risk L1–L12, hypothesis H1–H8).
Success: the exploratory numbers of 19/11 (CONTEXT.md sections 5 and 6.2) are reproduced or
explicitly corrected; every number quoted in the notebook text is computed in a cell;
"Restart & Run All" works in a few minutes; run_eda reproduces the same metrics.json.

# DATA SCHEMA
- Unit of observation: one agent step t (0-based), scored BEFORE the proposed tool call a_t runs.
- steps: traj_id, t, origin_split {train,val,test}, domain, template_id, world_id, source_category,
  thought, tool, args_json, obs, label {benign, injection_point, hijacked, failed_injection},
  y_scan = label(t-1) in {injection_point, failed_injection}, y_gate = label(t) == hijacked, is_first_h.
- trajectories: category, source_category, attack_type, compliance, task, template_id, complexity,
  world_id, world_json, n_steps, injection_step, first_h_step, y_traj (provisional).
- templates: template_id, expected_tools (oracle only).
- Group key: template_id. Domain key: domain. Prevalence per step: y_gate 17.4 %, y_scan 9.9 %.

# LEAKAGE CONSTRAINTS (hard rules)
- The fixed test set (origin_split == test) never drives a choice. Descriptive statistics may show
  train / val / test side by side; anything that looks like a score is computed on val.
- Shortcut baselines (world lookup, position only, trajectory length) are fitted on train and scored
  on val, never scored on their own training data.
- TF-IDF / SVD / t-SNE in the EDA are descriptive only, never reused as model features, and are
  labelled as such.
- Generator metadata (category, source_category, attack_type, compliance, world_id, expected_tools,
  n_steps) are analysed as potential leaks, never presented as usable features.
- Nothing in the notebook modifies data/interim.

# STATISTICAL ASSUMPTIONS
- Steps of one trajectory are dependent; rates are reported at the step AND trajectory level.
- Association measures (Cramér's V) are descriptive; no p-value on 71k dependent steps.
- Small cells (e.g. 3 delayed trajectories with gap 4) are reported as counts, not percentages only.

# TECHNICAL CONSTRAINTS
- Python 3.12, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2 (no new
  dependency). Fixed seed (agentdrift.utils.seed). Relative paths (agentdrift.config).
- Logic in src/, the notebook only calls and displays. Colour-blind-safe categorical palette in a
  fixed order; one colour per entity across all figures; figures exported to results/figures and
  report/latex/figures (PDF).

# OUTPUT FORMAT
Python modules with docstrings; notebook with, per section: question (markdown), computation cell,
figure, table, "Key findings" and "Consequence" generated from computed values.

# VERIFICATION AND FALLBACK TESTS
- Totals match the schema validation (71,024 steps, 5,536 first H, 7,036 y_scan positives).
- If a 19/11 number is not reproduced, report both values and the definition used; do not tune the
  definition until it matches.
- Shortcut baselines must be at or above chance on val; a shortcut above 0.9 AUC is flagged.

# SELF-CRITIQUE
List the definitions you had to choose (e.g. "unknown address", "injection marker"), how sensitive
the reported numbers are to them, and which conclusions would change if they were defined otherwise.
```

### Result and check

- Produced: one function per table in `audit.py`, a validated colour-blind-safe palette in `plots.py`, a headless runner writing `metrics.json`, CSV tables and 17 figures, and a notebook (sections A0–A10) that runs in about 12 s.
- Checked: shortcut baselines read train only and are scored on val; t-SNE on dev steps only; totals equal the schema validation; full execution on the pinned versions (2 pandas 3 / regex failures fixed).
- Changed after review: several generated findings were not supported by the computed tables (e.g. ATBench tool counts inverted, "never" contradicted by 29 occurrences) and some numbers were typed by hand; all rewritten from computed values.
- Linked audit: none yet.
