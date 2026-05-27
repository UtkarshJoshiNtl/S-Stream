from __future__ import annotations

import numpy as np
from cpu_lbm import CPULBM2D


def test_lbm_creation() -> None:
    sim = CPULBM2D(64, 64, 0.02)
    assert sim.width == 64
    assert sim.height == 64


def test_initialization() -> None:
    sim = CPULBM2D(64, 64, 0.02)
    sim.initialize(1.0, 0.1, 0.0)
    density = sim.get_density()
    assert density.shape == (64, 64)
    assert np.allclose(density, 1.0)


def test_step() -> None:
    sim = CPULBM2D(64, 64, 0.02)
    sim.initialize(1.0, 0.1, 0.0)
    f_before = sim.f.copy()
    sim.step()
    assert not np.allclose(sim.f, f_before)


def test_obstacles() -> None:
    sim = CPULBM2D(64, 64, 0.02)
    obstacles = np.zeros((64, 64), dtype=bool)
    obstacles[32:40, 32:40] = True
    sim.obstacles = obstacles
    sim.initialize(1.0, 0.1, 0.0)
    f_before = sim.f.copy()
    sim.apply_obstacles()
    assert np.any(sim.f != f_before)


def test_add_obstacle() -> None:
    sim = CPULBM2D(64, 64, 0.02)
    sim.add_obstacle(32, 32, radius=5)
    assert sim.obstacles[32, 32]
    assert not sim.obstacles[0, 0]
