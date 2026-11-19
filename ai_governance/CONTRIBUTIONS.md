# CONTRIBUTIONS.md — Ownership Matrix

> Group DIA 7 — Oscar DUJARDIN · Bertrand MAIGRE · Victor MICHELON · Jean-Baptiste PASTEAU
> Each module has one **owner** (defends it at the oral, explains every function and hyperparameter) and one **reviewer**.

## Module ownership

| Work package | Scope | Modules | Owner | Reviewer |
|---|---|---|---|---|
| **A — Data & protocol** | Data audit, EDA, schema, splits, external dataset adapters | `src/agentdrift/data/`, `src/agentdrift/splits/`, `src/agentdrift/pipelines/run_audit.py`, `run_eda.py`, `run_external.py` | | |
| **B — Features & AutoML** | Scanner / gate views, text, structural and provenance features, artifact masking, baselines, AutoML | `src/agentdrift/features/`, `models/baselines.py`, `models/automl.py`, `pipelines/run_baseline.py`, `run_automl.py` | | |
| **C — Advanced models** | Boosting, ensembles, transformer, sequence model, anomaly detection, SOTA benchmarks | `models/boosting.py`, `ensemble.py`, `transformer.py`, `sequence.py`, `anomaly.py`, `sota.py` | | |
| **D — Decision & evaluation** | Calibration, thresholds, sequential decision, metrics, bootstrap, statistical tests | `src/agentdrift/decision/`, `src/agentdrift/evaluation/`, `pipelines/run_experiment.py` | | |
| **Shared** | Config, seeding, tests, figures | `config.py`, `utils/`, `viz/`, `tests/` | | |

## Report sections

| Section | Author | Reviewer |
|---|---|---|
| Business scope & problem formalisation | | |
| Data description & exploration | | |
| Methodology (splits, leakage, imbalance) | | |
| Models & hyperparameters | | |
| Results, statistics & comparison | | |
| Discussion, limitations, conclusion | | |

## Governance entries

| Student | PROMPT_LOG entries | AI_AUDIT entries |
|---|---|---|
| Oscar DUJARDIN | | |
| Bertrand MAIGRE | | |
| Victor MICHELON | | |
| Jean-Baptiste PASTEAU | | |

## Model versions

| Version | Content | Owner |
|---|---|---|
| v0.0 | Trivial and shortcut baselines | |
| v0.1 | TF-IDF + logistic regression | |
| v0.2 | AutoML | |
| v0.3 | Structural / provenance features, masking, boosting | |
| v0.4 | Ensemble | |
| v0.5 | Out-of-course model (DeBERTa-v3 / sequence) | |
| v0.6 | Sequential decision (H6) / anomaly detection (H7) | |
| v1.0 | Final calibrated model, external evaluation | |
