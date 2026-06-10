from __future__ import annotations

import numpy as np

from engines.base import SimEngine
from engines.lbm_common import LATTICE_2D


def reynolds_number(
    sim: SimEngine, obstacle_diameter: float | None = None
) -> float:
    L = obstacle_diameter or float(sim.grid_shape[1])
    U = sim.u_inflow
    nu = sim.viscosity
    if nu <= 0 or U <= 0:
        return 0.0
    return U * L / nu


def drag_coefficient(sim: SimEngine) -> float:
    """Momentum-exchange method for drag on 2D obstacles."""
    lattice = LATTICE_2D
    obs = sim.obstacles  # type: ignore[attr-defined]
    if not np.any(obs):
        return 0.0

    Fx = 0.0
    for i in range(9):
        opp = lattice.opp[i]
        cx = lattice.cx[i]
        cy = lattice.cy[i]
        shifted = np.roll(obs, shift=(-cy, -cx), axis=(0, 1))
        boundary = (~obs) & shifted
        if not np.any(boundary):
            continue
        f_i = sim.f[i][boundary]  # type: ignore[attr-defined]
        f_opp = sim.f[opp][boundary]  # type: ignore[attr-defined]
        Fx += cx * float(np.sum(f_i + f_opp))

    U = sim.u_inflow
    rho0 = 1.0
    A = float(np.sum(obs))
    D = 2.0 * np.sqrt(A / np.pi) if A > 0 else 1.0
    if U <= 0 or D <= 0:
        return 0.0
    Cd = 2.0 * abs(Fx) / (rho0 * U * U * D)
    return float(Cd)


def strouhal_number(
    v_history: list[float],
    dt: float = 1.0,
    diameter: float = 1.0,
    velocity: float = 1.0,
) -> float | None:
    if len(v_history) < 32:
        return None
    arr = np.array(v_history[-256:])
    arr = arr - np.mean(arr)
    n = len(arr)
    if n < 4:
        return None
    window = np.hanning(n)
    fft = np.fft.rfft(arr * window)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(fft)
    low = max(1, int(0.02 * n))
    mag[:low] = 0.0
    peak_idx = int(np.argmax(mag))
    peak_mag = mag[peak_idx]
    total = float(np.sum(mag))
    if total < 1e-10 or peak_mag / total < 0.1:
        return None
    f_peak = freqs[peak_idx]
    if velocity <= 0:
        return None
    return f_peak * diameter / velocity
