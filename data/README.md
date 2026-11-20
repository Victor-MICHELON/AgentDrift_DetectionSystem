# Data

Raw and processed data are **not committed** (see `.gitignore`). Sources, licenses and checksums are in `configs/data.yaml`.

| Folder | Content |
|---|---|
| `raw/` | AgentDrift as downloaded (CC BY 4.0) |
| `interim/` | Step-level table parsed from raw trajectories |
| `processed/` | Features per view (scanner / gate) |
| `external/` | ATBench (Apache-2.0), AgentDojo (optional) |

Download and verify (pinned versions and SHA-256 in `configs/data.yaml`):

```bash
python -m agentdrift.data.download               # AgentDrift @ 014a514 -> data/raw/agentdrift
python -m agentdrift.data.download --atbench     # + ATBench (HF revision 4476ef9) -> data/external/atbench
python -m agentdrift.data.download --verify-only # offline check: commit, no modified file, checksums
```

The command is idempotent and exits with code 1 if a file is missing, modified or at the wrong version.

The base split is `data_taskdisjoint/` (no task template shared between parts). Read the dataset README section "Known regularities you must control for" before any modelling.
