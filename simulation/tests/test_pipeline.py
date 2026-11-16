"""End-to-end pipeline tests on a tiny analytic sphere."""

import numpy as np
import pytest

from sstream.pipeline import SolveCancelled, SolvePlan, run_case
from sstream.preview import RunPreset
from sstream.schema import FluidSpec, TunnelSpec
from sstream.voxelize import VoxelResult


def test_solve_returns_finite_fields_with_the_mask_shape(tiny_result):
    result = tiny_result
    assert result.velocity.shape == result.tunnel.grid_shape + (3,)
    assert result.vorticity.shape == result.velocity.shape
    assert result.dye.shape == result.tunnel.grid_shape
    assert np.isfinite(result.velocity).all()
    assert np.isfinite(result.density).all()
    assert np.isfinite(result.vorticity).all()


def test_force_coefficients_are_reported_from_the_wall_force(tiny_result):
    # The shared air fixture sits at tau ~ 0.5000007 (air at this grid), so
    # the 640-step force window is noise near the stability edge; the sign
    # is covered by the stable high-viscosity test below instead.
    result = tiny_result
    assert np.isfinite(result.cd)
    assert np.isfinite(result.cl)
    assert result.cd_trace.size > 10
    assert result.cd_trace.size == result.cl_trace.size
    assert result.reynolds == pytest.approx(result.fluid.reynolds)


def test_dye_marks_the_wake_and_stays_out_of_the_solid(tiny_result):
    result = tiny_result
    assert result.dye.max() == pytest.approx(1.0)
    assert np.allclose(result.dye[result.mask.solid], 0.0)
    downstream = result.dye[int(result.tunnel.grid_shape[0] * 0.6):]
    assert downstream.max() > 0.1


def test_progress_is_reported_and_ends_at_100(tiny_result):
    seen = []

    def progress(percent, message):
        seen.append((percent, message))

    from sstream.pipeline import SolvePlan
    plan = SolvePlan(
        mask=tiny_result.mask,
        tunnel=tiny_result.tunnel,
        fluid=tiny_result.fluid,
        preset=RunPreset("Tiny", 48, 32, "progress test"),
    )
    run_case(plan, progress=progress)
    assert seen[0][0] == 1
    assert seen[-1][0] == 100
    assert any("Solving" in message for _, message in seen)


def test_cancel_stops_the_run():
    tunnel = TunnelSpec(cells_streamwise=32)
    shape = tunnel.grid_shape
    solid = np.zeros(shape, dtype=bool)
    solid[10:13, 8:11, 4:7] = True
    mask = VoxelResult(
        solid=solid,
        boundary=solid,
        origin=tunnel.grid_origin,
        spacing=tunnel.grid_spacing,
        dimensions=shape,
    )
    plan = SolvePlan(
        mask=mask,
        tunnel=tunnel,
        fluid=FluidSpec(),
        preset=RunPreset("Tiny", 32, 400, "cancel test"),
    )
    with pytest.raises(SolveCancelled):
        run_case(plan, should_stop=lambda: True)


def test_a_body_in_a_positive_x_flow_reports_positive_drag(tmp_path):
    """Known-sign test: force on the body in +x is positive drag.

    The shared air fixture sits at tau ~ 0.5000007 where this is noise, so
    this test uses a viscous fluid (tau ~ 0.58) where the sign is physical
    and every sampled step is positive.
    """
    from sstream.pipeline import SolvePlan
    from sstream.preview import RunPreset
    from sstream.schema import FluidSpec, TunnelSpec
    from sstream.voxelize import VoxelResult

    tunnel = TunnelSpec(cells_streamwise=48)
    shape = tunnel.grid_shape
    x, y, z = np.indices(shape)
    solid = (
        (x - shape[0] * 0.35) ** 2
        + (y - shape[1] * 0.5) ** 2
        + (z - shape[2] * 0.45) ** 2
    ) < (min(shape) * 0.18) ** 2
    mask = VoxelResult(
        solid=solid,
        boundary=solid,
        origin=tunnel.grid_origin,
        spacing=tunnel.grid_spacing,
        dimensions=shape,
    )
    plan = SolvePlan(
        mask=mask,
        tunnel=tunnel,
        fluid=FluidSpec(kinematic_viscosity=1.0),
        preset=RunPreset("Test", 48, 640, "sign test"),
        u_lattice=0.1,
    )
    result = run_case(plan)
    assert result.units.tau > 0.55
    assert result.cd > 0.0
    assert np.mean(result.cd_trace) > 0.0


def test_convergence_detector_reports_never_converged_when_tolerance_unmet():
    tunnel = TunnelSpec(cells_streamwise=48)
    shape = tunnel.grid_shape
    x, y, z = np.indices(shape)
    solid = (
        (x - shape[0] * 0.35) ** 2
        + (y - shape[1] * 0.5) ** 2
        + (z - shape[2] * 0.45) ** 2
    ) < (min(shape) * 0.18) ** 2
    mask = VoxelResult(
        solid=solid,
        boundary=solid,
        origin=tunnel.grid_origin,
        spacing=tunnel.grid_spacing,
        dimensions=shape,
    )
    plan = SolvePlan(
        mask=mask,
        tunnel=tunnel,
        fluid=FluidSpec(kinematic_viscosity=1.0),
        preset=RunPreset("Test", 48, 400, "convergence test"),
        u_lattice=0.1,
        convergence_tol=1e-9,
        convergence_window=10,
    )
    result = run_case(plan)
    assert result.converged is False
    assert any("Convergence: never converged" in w for w in result.warnings)
