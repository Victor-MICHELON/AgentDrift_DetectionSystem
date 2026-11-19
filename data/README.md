# Data

Raw and processed data are **not committed** (see `.gitignore`). Sources, licenses and checksums are in `configs/data.yaml`.

| Folder | Content |
|---|---|
| `raw/` | AgentDrift as downloaded (CC BY 4.0) |
| `interim/` | Step-level table parsed from raw trajectories |
| `processed/` | Features per view (scanner / gate) |
| `external/` | ATBench (Apache-2.0), AgentDojo (optional) |

Download: `python -m agentdrift.pipelines.run_audit` (to be implemented in `src/agentdrift/data/download.py`).
