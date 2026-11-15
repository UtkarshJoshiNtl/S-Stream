"""Solver tests: uniform flow, plane Poiseuille, forcing, and stability.

The Poiseuille case is the important one.  Its exact solution fixes both the
viscosity and the wall location, so a wrong table, a missing force term, or a
misplaced bounce-back plane all show up as a profile error.  Half-way
bounce-back places the no-slip plane halfway along the link, i.e. at y=0.5 and
y=ny-1.5 for solid layers at y=0 and y=ny-1.
"""

import numpy as np
import pytest

from sstream.lattice import CS2
from sstream.lbm import LbmConfig, LbmSolver, pick_device

DEVICE = pick_device()


def channel_solver(ny=20, tau=0.65, target_speed=0.04, smagorinsky=0.0):
    """A wall-bounded channel driven by a body force, periodic along x and z."""
    nx, nz = 4, 4
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    nu = CS2 * (tau - 0.5)
    height = ny - 2
    force = 8.0 * nu * target_speed / (height * height)
    config = LbmConfig(
        shape=(nx, ny, nz),
        tau=tau,
        smagorinsky=smagorinsky,
        body_force=(force, 0.0, 0.0),
        periodic=(True, False, True),
        device=DEVICE,
    )
    return LbmSolver(config, solid), solid, force, nu, height


def test_uniform_flow_is_preserved():
    solver = LbmSolver(
        LbmConfig(shape=(12, 10, 8), tau=0.6, periodic=(True, True, True), device=DEVICE)
    )
    solver.initialize(ux=0.02)
    solver.run(200)
    velocity = solver.velocity()
    assert np.allclose(velocity[..., 0], 0.02, atol=1e-5)
    assert np.allclose(velocity[..., 1:], 0.0, atol=1e-6)
    assert np.allclose(solver.density(), 1.0, atol=1e-6)


def test_inlet_and_outlet_hold_a_uniform_stream():
    solver = LbmSolver(
        LbmConfig(
            shape=(24, 10, 10),
            tau=0.6,
            periodic=(False, True, True),
            inlet_speed=0.02,
            device=DEVICE,
        )
    )
    solver.initialize(ux=0.02)
    solver.run(200)
    core = solver.velocity()[4:-4, :, :, 0]
    assert np.allclose(core, 0.02, atol=1e-5)


def test_solid_cells_are_inert():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[3:5, 3:5, 3:5] = True
    solver = LbmSolver(
        LbmConfig(shape=(8, 8, 8), tau=0.6, periodic=(True, True, True), device=DEVICE),
        mask,
    )
    solver.run(50)
    velocity = solver.velocity()[mask]
    assert np.allclose(velocity, 0.0)
    assert np.allclose(solver.density()[mask], 1.0)
    assert solver.fluid_cells == 8**3 - 8


def test_poiseuille_profile_matches_the_exact_solution():
    solver, _, force, nu, height = channel_solver(ny=20, tau=0.65)
    solver.run(12000)

    ny = solver.shape[1]
    u = solver.velocity()[0, :, 0, 0]
    y = np.arange(ny)
    fluid = (y >= 1) & (y <= ny - 2)
    # Walls sit at y=0.5 and y=ny-1.5, the mid-link positions.
    exact = force / (2.0 * nu) * (y - 0.5) * (ny - 1.5 - y)

    error = np.abs(u[fluid] - exact[fluid]).max() / exact[fluid].max()
    assert error < 0.015
    # The profile must be symmetric to machine precision.
    assert np.abs(u[fluid][::-1] - u[fluid]).max() < 1e-6
    # And the walls must carry exactly the momentum the body force adds.
    total = solver.total_wall_force()[0]
    assert total == pytest.approx(force * solver.shape[0] * height * solver.shape[2], rel=0.01)


def test_poiseuille_is_tau_independent():
    """The same physical channel with a different relaxation time."""
    results = []
    for tau in (0.6, 0.7):
        solver, _, _, _, _ = channel_solver(ny=20, tau=tau)
        solver.run(12000)
        results.append(solver.velocity()[0, :, 0, 0].copy())
    assert np.abs(results[0] - results[1]).max() < 1e-4


def test_channel_mass_is_conserved():
    solver, _, _, _, _ = channel_solver(ny=20, tau=0.65)
    initial = solver.total_mass()
    solver.run(12000)
    drift = abs(solver.total_mass() - initial) / initial
    # Float32 accumulation and the Guo force leave a small residual; a leak
    # through the bounce-back walls would be far larger than this.
    assert drift < 1e-3


def test_smagorinsky_stays_finite_and_bounded():
    solver, _, _, _, _ = channel_solver(ny=20, tau=0.65, smagorinsky=0.1)
    solver.run(2000)
    velocity = solver.velocity()
    density = solver.density()
    assert np.isfinite(velocity).all()
    assert np.isfinite(density).all()
    assert solver.max_speed() < 0.2
    assert density.min() > 0.5 and density.max() < 1.5
