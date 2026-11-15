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
    result = tiny_result
    assert np.isfinite(result.cd) and result.cd > 0.0
    assert np.isfinite(result.cl)
    assert result.drag_trace.size > 10
    assert result.drag_trace.size == result.lift_trace.size
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
