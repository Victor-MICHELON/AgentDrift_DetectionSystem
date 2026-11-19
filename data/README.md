# Data

Raw and processed data are **not committed** (see `.gitignore`). Sources, licenses and checksums are in `configs/data.yaml`.

| Folder | Content |
|---|---|
| `raw/` | AgentDrift as downloaded (CC BY 4.0) |
| `interim/` | Step-level table parsed from raw trajectories |
| `processed/` | Features per view (scanner / gate) |
| `external/` | ATBench (Apache-2.0), AgentDojo (optional) |

Download AgentDrift (pinned commit, see `configs/data.yaml`):

```bash
git clone https://github.com/Asif-0209/AgentDrift data/raw/agentdrift
git -C data/raw/agentdrift checkout 014a514fa998b4ac4519579fceb8a5884b379bda
shasum -a 256 data/raw/agentdrift/data_taskdisjoint/*.jsonl   # compare with configs/data.yaml
```

The base split is `data_taskdisjoint/` (no task template shared between parts). Read the dataset README section "Known regularities you must control for" before any modelling. A scripted download with checksum verification will live in `src/agentdrift/data/download.py`.
