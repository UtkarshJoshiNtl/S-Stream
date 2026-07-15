"""Phase D feature tests: halfway BB, IBM, free-surface, adjoint, FSI."""

from __future__ import annotations

import numpy as np

from engines.adjoint import finite_difference_gradient
from engines.free_surface import FreeSurfaceTracker
from engines.fsi import FSIRigidCircle
from engines.ibm import RigidIBM
from engines.lbm2d import LBM2D


def test_halfway_bb_no_nan() -> None:
    sim = LBM2D(width=64, height=32, viscosity=0.05)
    sim.obstacle_bc = "halfway"
    sim.u_inflow = 0.05
    sim.initialize(rho=1.0, u=0.05, v=0.0)
    sim.add_obstacle(20, 16, radius=6)
    sim.run(200, physics_only=True)
    assert np.isfinite(sim.f).all()
    assert float(sim.u.max()) > 0.0


def test_ibm_force_nonzero() -> None:
    sim = LBM2D(width=48, height=48, viscosity=0.08)
    sim.u_inflow = 0.05
    sim.initialize(rho=1.0, u=0.05, v=0.0)
    ibm = RigidIBM(sim.width, sim.height, stiffness=0.5)
    ibm.set_circle(24, 24, radius=6.0, u_body=0.0, v_body=0.0)
    sim.ibm = ibm
    sim.run(50, physics_only=True)
    fx, fy = ibm.compute_force(sim.u, sim.v)
    assert float(np.max(np.abs(fx)) + np.max(np.abs(fy))) > 0.0


def test_free_surface_mass_stable() -> None:
    tracker = FreeSurfaceTracker(32, 32)
    tracker.fill_bottom(10)
    m0 = tracker.mass()
    u = np.zeros((32, 32), dtype=np.float32)
    v = np.zeros((32, 32), dtype=np.float32)
    u[:, :] = 0.02
    for _ in range(50):
        tracker.step(u, v)
    m1 = tracker.mass()
    assert abs(m1 - m0) / m0 < 0.05


def test_finite_difference_gradient_smoke() -> None:
    def obj(x: np.ndarray) -> float:
        return float((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

    g = finite_difference_gradient(obj, np.array([1.0, -2.0]), eps=1e-4)
    assert np.allclose(g, 0.0, atol=1e-2)


def test_fsi_rigid_circle_moves() -> None:
    ibm = RigidIBM(40, 40, stiffness=0.2)
    fsi = FSIRigidCircle(ibm, radius=5.0, mass=5.0)
    fsi.reset(20.0, 30.0)
    y0 = fsi.body.y
    ibm.fx[:] = 0.0
    ibm.fy[:] = 0.1  # upward Eulerian force → body force opposite
    fsi.step(gravity=-0.01)
    assert fsi.body.y != y0
