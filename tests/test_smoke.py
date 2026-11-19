"""Smoke tests: the package imports and seeding is deterministic."""

import numpy as np

from agentdrift.config import CONFIGS, ROOT
from agentdrift.utils.seed import set_seed


def test_paths_are_inside_repo():
    assert CONFIGS.parent == ROOT
    assert (ROOT / "pyproject.toml").exists()


def test_seed_is_deterministic():
    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    assert np.array_equal(a, np.random.rand(3))
