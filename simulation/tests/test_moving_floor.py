"""Moving-floor wall mode (Step 2).

The floor of a vehicle tunnel travels with the inlet stream so the body does
not sit in a thick no-slip boundary layer.  The moving wall is a domain-boundary
reflection (bounce-back plus ``2 w_i rho (c_i . u_wall) / cs2``), so it must not
appear in the body force, and the empty tunnel must hold a steady, uniform
stream with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from sstream.lbm import (
    TUNNEL_OUTLET_MODE,
    TUNNEL_WALL_MODES,
    LbmConfig,
    LbmSolver,
    pick_device,
)

DEVICE = pick_device()
SHAPE = (96, 58, 58)
U = 0.05
TAU = 0.514369
CS = 0.12
STEPS = 20_000


def _tunnel_solver() -> LbmSolver:
    config = LbmConfig(
        shape=SHAPE,
        tau=TAU,
        smagorinsky=CS,
        periodic=(False, False, False),
        inlet_speed=U,
        wall_modes=TUNNEL_WALL_MODES,
        outlet_mode=TUNNEL_OUTLET_MODE,
        moving_wall_speed=U,
        device=DEVICE,
    )
    solver = LbmSolver(config)
    solver.initialize(ux=U)
    return solver


def _tunnel_state(solver: LbmSolver):
    """Mean density and mean speed from live populations, not cached fields."""
    from sstream.lattice import DIRECTIONS

    f = np.asarray(solver.f.numpy(), dtype=np.float64)
    rho = f.sum(axis=0)
    ux = np.einsum("i,ixyz->xyz", DIRECTIONS[:, 0].astype(np.float64), f) / rho
    uy = np.einsum("i,ixyz->xyz", DIRECTIONS[:, 1].astype(np.float64), f) / rho
    uz = np.einsum("i,ixyz->xyz", DIRECTIONS[:, 2].astype(np.float64), f) / rho
    return float(rho.mean()), float(np.sqrt(ux**2 + uy**2 + uz**2).mean())


@pytest.mark.slow
def test_moving_floor_empty_tunnel_is_uniform_and_steady():
    """Profile + steadiness over 20k steps with the moving floor."""
    solver = _tunnel_solver()

    samples = {}
    for step in range(1, STEPS + 1):
        solver.step()
        if step % 100 == 0 and step >= 2000:
            samples[step] = _tunnel_state(solver)

    rho = np.asarray([s[0] for s in samples.values()])
    speed = np.asarray([s[1] for s in samples.values()])
    assert max(abs(rho - 1.0)) < 0.005, "empty-tunnel density left the 0.5% band"
    assert max(abs(speed - U)) < 0.02 * U, "empty-tunnel speed left the 2% band"
    assert abs(rho[-1] - rho[0]) < 0.005
    assert abs(speed[-1] - speed[0]) < 0.02 * U

    # --- profile: lowest 3 cells above the floor vs the core -------------
    velocity = solver.velocity()
    nx, ny, nz = SHAPE
    deviations = []
    for frac in (0.25, 0.50, 0.75):
        ix = int(round(frac * (nx - 1)))
        core = float(np.mean(velocity[ix, :, nz // 2 :, 0]))
        near_floor = float(np.mean(velocity[ix, :, 1:4, 0]))
        deviations.append(abs(near_floor - core) / core)
    print(
        "moving-floor profile (core vs lowest 3 cells): "
        + ", ".join(f"{d:.2%}" for d in deviations)
    )
    assert max(deviations) < 0.03, (
        f"floor-layer speed deviates from core by {max(deviations):.2%} (limit 3%)"
    )

    # The moving wall is a boundary reflection, not a solid: it must not
    # contribute to the body force.
    force = np.asarray(solver.total_wall_force())
    assert np.allclose(force, 0.0, atol=1e-9), f"moving floor leaked into body force: {force}"
