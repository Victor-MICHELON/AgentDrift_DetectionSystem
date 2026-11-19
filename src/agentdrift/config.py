"""Project paths and YAML config loading. All paths are relative to the repository root."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW, INTERIM, PROCESSED, EXTERNAL = (DATA / d for d in ("raw", "interim", "processed", "external"))
RESULTS = ROOT / "results"
RUNS, FIGURES = RESULTS / "runs", RESULTS / "figures"


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml (name may include a subfolder, e.g. 'models/v0_1_baseline')."""
    path = CONFIGS / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
