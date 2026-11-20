"""Gauge-pressure momentum exchange (Step 1).

The momentum-exchange force used to count the full bounce-back population,
so a body touching a wall felt the ambient pressure cs2*rho (~0.33) with no
balancing term from the other side.  That pushed Cz for a floor-resting sphere
into the tens and made Cl/Cz meaningless.  The gauge form subtracts the
rest-state equilibrium population ``w_i * rho_ref`` from every link:

    force = -2 * c_i * (f_bounced - w_i * rho_ref)

For a closed body floating in the flow the extra term cancels around the
surface (drag unchanged to floating-point precision); for a body touching a
wall the unbalanced ambient pressure is removed, leaving only the dynamic
contribution.

Tests:
  (i)   a solid block resting on a wall in still fluid at rho=1 gives zero
        force in all components;
  (ii)  a floating sphere in uniform flow keeps the same drag within 1%;
  (iii) a sphere resting on the floor drops from Cz ~ -54 to order 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sstream.lbm import TUNNEL_OUTLET_MODE, LbmConfig, LbmSolver, pick_device
from sstream.mesh import build_model, load_surface
from sstream.pipeline import freestream_probe_x
from sstream.schema import FluidSpec, MeshSpec, MeshTransform, TunnelSpec
from sstream.voxelize import voxelize

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "3d-files"
DEVICE = pick_device()
SPEED = 0.05
REYNOLDS = 100.0
SMAGORINSKY = 0.12
STEPS = 6000
SAMPLE_EVERY = 100
ALL_NOSLIP = {face: "no_slip" for face in ("x0", "x1", "y0", "y1", "z0", "z1")}
# The Step 1 gauge comparison was calibrated with the historical floor
# (no-slip, free-slip sides/roof); pin it so the moving floor's extra underbody
# shear does not change what these tests measure.
GAUGE_TEST_WALL_MODES = {
    "y0": "free_slip",
    "y1": "free_slip",
    "z0": "no_slip",
    "z1": "free_slip",
}


def _still_block_force(gauge: bool) -> np.ndarray:
    """Total force on a block resting on the floor of a still fluid."""
    shape = (16, 12, 10)
    solid = np.zeros(shape, dtype=bool)
    solid[4:8, 3:7, 0:2] = True  # block on the floor, merged with the wall
    solver = LbmSolver(
        LbmConfig(
            shape=shape,
            tau=0.6,
            periodic=(False, False, False),
            inlet_speed=0.0,
            wall_modes=ALL_NOSLIP,
            gauge_pressure=gauge,
            device=DEVICE,
        ),
        solid,
    )
    solver.initialize(ux=0.0, uy=0.0, uz=0.0)
    solver.run(50)
    return np.asarray(solver.total_wall_force(), dtype=float)


def test_still_block_on_wall_has_zero_gauge_force():
    """(i) still fluid, rho=1: the gauge force on a wall-resting block is zero."""
    force_on = _still_block_force(True)
    # float32 roundoff accumulates to ~1e-6 over 50 steps; the exact-rest
    # identity bounced == w_i holds only in exact arithmetic.
    assert np.allclose(force_on, 0.0, atol=1e-5), f"gauge force is not zero: {force_on}"

    # Non-vacuous check: without the gauge term the ambient pressure imbalance
    # is real and the force must not vanish.
    force_off = _still_block_force(False)
    print(f"still-block force: gauge on={force_on}, gauge off={force_off}")
    assert np.linalg.norm(force_off) > 1e-4, (
        "the block produced no force even without the gauge term; the test geometry "
        "is not exercising the bounce-back force path"
    )


def _sphere_case(position_z: float):
    """A D=10 sphere case; position_z=3 floats it at mid-height, 0 rests it on the floor."""
    tunnel = TunnelSpec(length=10.0, width=6.0, height=6.0, floor_z=0.0, cells_streamwise=100)
    spec = MeshSpec(
        orient="none",
        fraction=0.1,
        transform=MeshTransform(position_x=3.0, position_z=position_z),
    )
    build = build_model(load_surface(ASSETS / "sphere.stl"), spec, tunnel)
    mask = voxelize(build.mesh, tunnel)
    assert not mask.empty

    bounds = np.asarray(build.mesh.bounds, dtype=float).reshape(3, 2)
    diameter = float((bounds[:, 1] - bounds[:, 0]).mean())
    fluid = FluidSpec(speed=SPEED, kinematic_viscosity=SPEED * diameter / REYNOLDS,
                      reference_length=diameter)
    area = int(np.asarray(mask.solid, dtype=bool).any(axis=0).sum())
    return tunnel, mask, fluid, area


def _run_sphere(tunnel, mask, fluid, area, gauge: bool):
    """Run the case and return (Cd, Cy, Cz) averaged over the late window."""
    from sstream.lattice import LatticeUnits

    units = LatticeUnits.from_physical(
        fluid.speed, fluid.kinematic_viscosity, tunnel.cell_size,
        u_lattice=SPEED, smagorinsky=SMAGORINSKY,
    )
    solver = LbmSolver(
        LbmConfig(
            shape=tunnel.grid_shape,
            tau=units.tau,
            smagorinsky=SMAGORINSKY,
            periodic=(False, False, False),
            inlet_speed=units.u_lattice,
            wall_modes=GAUGE_TEST_WALL_MODES,
            outlet_mode=TUNNEL_OUTLET_MODE,
            gauge_pressure=gauge,
            moving_wall_speed=0.0,
            device=DEVICE,
        ),
        mask.solid,
    )
    solver.initialize(ux=units.u_lattice)
    x_probe = freestream_probe_x(mask.solid)
    cds, cys, czs = [], [], []
    for step in range(1, STEPS + 1):
        solver.step()
        if step % SAMPLE_EVERY == 0 and step > STEPS // 2:
            force = solver.total_wall_force()
            u = solver.plane_speed(x_probe)
            scale = 2.0 / (u * u * area)
            cds.append(force[0] * scale)
            cys.append(force[1] * scale)
            czs.append(force[2] * scale)
    return float(np.mean(cds)), float(np.mean(cys)), float(np.mean(czs))


@pytest.mark.slow
def test_floating_sphere_drag_unchanged_by_gauge():
    """(ii) a floating sphere keeps its drag within 1%."""
    tunnel, mask, fluid, area = _sphere_case(position_z=3.0)
    cd_before, _, _ = _run_sphere(tunnel, mask, fluid, area, gauge=False)
    cd_after, _, _ = _run_sphere(tunnel, mask, fluid, area, gauge=True)
    change = abs(cd_after - cd_before) / abs(cd_before)
    print(f"floating sphere Cd: gauge off={cd_before:.4f} on={cd_after:.4f} change={change:.2%}")
    assert change < 0.01, f"gauge changed drag by {change:.2%} (limit 1%)"


@pytest.mark.slow
def test_floor_sphere_cz_is_order_one_with_gauge():
    """(iii) a floor-resting sphere drops from Cz ~ -54 to order 1."""
    tunnel, mask, fluid, area = _sphere_case(position_z=0.0)
    _, _, cz_before = _run_sphere(tunnel, mask, fluid, area, gauge=False)
    _, _, cz_after = _run_sphere(tunnel, mask, fluid, area, gauge=True)
    print(f"floor-resting sphere Cz: gauge off={cz_before:.3f} on={cz_after:.3f}")
    # The "before" number must actually show the ambient-pressure artifact, so
    # the test cannot pass vacuously.
    assert abs(cz_before) > 10.0, f"expected the pre-gauge artifact (|Cz|>10), got {cz_before:.3f}"
    assert abs(cz_after) < 5.0, f"gauge Cz={cz_after:.3f} is not order 1"
