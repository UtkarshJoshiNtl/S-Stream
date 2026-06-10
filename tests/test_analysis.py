from __future__ import annotations

import numpy as np

from analysis.physics import drag_coefficient, reynolds_number, strouhal_number
from engines.lbm2d import LBM2D


def test_reynolds_number() -> None:
    sim = LBM2D(64, 64, viscosity=0.02)
    sim.u_inflow = 0.15
    Re = reynolds_number(sim)
    expected = 0.15 * 64 / 0.02
    assert abs(Re - expected) < 0.01


def test_reynolds_number_with_obstacle() -> None:
    sim = LBM2D(64, 64, viscosity=0.01)
    sim.u_inflow = 0.1
    Re = reynolds_number(sim, obstacle_diameter=10.0)
    expected = 0.1 * 10.0 / 0.01
    assert abs(Re - expected) < 0.01


def test_reynolds_number_zero_inflow() -> None:
    sim = LBM2D(32, 32)
    sim.u_inflow = 0.0
    assert reynolds_number(sim) == 0.0


def test_drag_coefficient_no_obstacle() -> None:
    sim = LBM2D(32, 32)
    assert drag_coefficient(sim) == 0.0


def test_drag_coefficient_cylinder() -> None:
    sim = LBM2D(64, 64, viscosity=0.02)
    sim.u_inflow = 0.1
    sim.initialize(rho=1.0, u=0.1, v=0.0)
    sim.add_obstacle(32, 32, radius=5)
    for _ in range(2000):
        sim.step()
    Cd = drag_coefficient(sim)
    assert Cd > 0.0
    # cylinder at Re ≈ 32 should have Cd > 1.0
    assert Cd > 0.5


def test_strouhal_number_regular_sine() -> None:
    n = 256
    dt = 0.1
    freq_bin = 5
    freq = freq_bin / (n * dt)  # exactly aligns with FFT bin
    t = np.arange(n) * dt
    v = np.sin(2 * np.pi * freq * t)
    St = strouhal_number(list(v), dt=dt, diameter=1.0, velocity=1.0)
    assert St is not None
    assert abs(St - freq) < 0.01


def test_strouhal_number_noise() -> None:
    rng = np.random.default_rng(42)
    noise = list(rng.normal(0, 0.01, 200))
    St = strouhal_number(noise, dt=0.01)
    assert St is None


def test_strouhal_number_short_history() -> None:
    assert strouhal_number([1.0, 2.0, 3.0], dt=1.0) is None
