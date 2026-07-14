"""Shared test fixtures and markers for S-Stream test suite."""

from __future__ import annotations

import pytest

from engines.lbm2d import LBM2D
from engines.lbm3d import LBM3D


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow-running")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU/CuPy")


@pytest.fixture
def sim_2d() -> LBM2D:
    """A fresh 64x64 2D engine (jitted, not stepped)."""
    return LBM2D(width=64, height=64)


@pytest.fixture
def sim_2d_32() -> LBM2D:
    """A fresh 32x32 2D engine for fast tests."""
    return LBM2D(width=32, height=32)


@pytest.fixture
def sim_3d() -> LBM3D:
    """A fresh 16x16x16 3D engine."""
    return LBM3D(width=16, height=16, depth=16)
