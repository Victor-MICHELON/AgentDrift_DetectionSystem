# AgentDrift Detection — Early guardrail against AI agent hijacking

ESILV 2026 Machine Learning project — Group DIA 7 (Oscar Dujardin, Bertrand Maigre, Victor Michelon, Jean-Baptiste Pasteau).

A step-level detector that scores each proposed tool call of an AI agent **before it is executed** and decides to allow, send to human review, or block, in order to stop indirect prompt injection attacks before the first dangerous action.

- Project foundation: [docs/CONTEXT.md](docs/CONTEXT.md)
- Course requirements: [docs/INSTRUCTIONS.md](docs/INSTRUCTIONS.md)
- Decision log: [docs/JOURNAL.md](docs/JOURNAL.md) · References: [docs/REFERENCE.md](docs/REFERENCE.md)
- AI governance: [ai_governance/](ai_governance/)

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
pip install -e . --no-deps         # makes `import agentdrift` work everywhere
pytest
```

macOS: LightGBM and XGBoost need OpenMP → `brew install libomp`.

## Layout

| Path | Content |
|---|---|
| `docs/` | `CONTEXT.md`, `INSTRUCTIONS.md`, `REFERENCE.md`, `JOURNAL.md` |
| `src/agentdrift/` | Package: `data`, `splits`, `features`, `models`, `decision`, `evaluation`, `pipelines` (modules are added as the project advances) |
| `configs/` | YAML configs (data sources, split protocol, one file per model version) |
| `data/` | Raw / interim / processed / external data (not committed) |
| `notebooks/` | Exploration and figures; logic stays in `src/` |
| `tests/` | Leakage, split and metric tests |
| `results/runs/<version>/` | Metrics, config, seed and git hash of each run |
| `report/` | LaTeX sources and `Report_Group_XX.pdf` |
| `ai_governance/` | `CONTRIBUTIONS.md`, `PROMPT_LOG.md`, `AI_AUDIT.md` |

## Reproducing results

| Result | Command |
|---|---|
| Data download + checksums | `python -m agentdrift.data.download --atbench` |
| Data audit | `python -m agentdrift.pipelines.run_audit` |
| v0.1 baseline | `python -m agentdrift.pipelines.run_baseline` |

More commands are added with each new pipeline.

All runs use a fixed seed (`agentdrift.utils.seed.set_seed`) and relative paths (`agentdrift.config`).
