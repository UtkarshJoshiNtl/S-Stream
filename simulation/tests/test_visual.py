"""Visualisation mode: stable lattice params and an honest report."""

import numpy as np
import pytest

from sstream.lattice import (
    CS2,
    VISUAL_SMAGORINSKY,
    VISUAL_TAU_MAX,
    VISUAL_TAU_MIN,
    VISUAL_U_LATTICE,
    clamp_visual_tau,
)
from sstream.pipeline import SolvePlan, run_case
from sstream.preview import RunPreset
from sstream.report import summary
from sstream.schema import FluidSpec, TunnelSpec
from sstream.voxelize import VoxelResult


def test_clamp_visual_tau_pins_into_the_stable_band():
    assert clamp_visual_tau(0.4) == VISUAL_TAU_MIN
    assert clamp_visual_tau(0.5001) == VISUAL_TAU_MIN
    assert clamp_visual_tau(0.51) == 0.51
    assert clamp_visual_tau(0.9) == VISUAL_TAU_MAX


def test_with_tau_keeps_the_conversion_and_flags_the_clamp():
    from sstream.lattice import LatticeUnits

    base = LatticeUnits(dx=0.1, dt=0.001, u_lattice=0.06, tau=0.5001)
    pinned = base.with_tau(VISUAL_TAU_MIN)
    assert pinned.tau == VISUAL_TAU_MIN
    assert pinned.dx == base.dx and pinned.dt == base.dt
    assert pinned.viscosity_clamped is True
    # nu_lattice = cs^2 (tau - 1/2): the pinned band is a small, known viscosity
    assert pinned.nu_lattice == pytest.approx(CS2 * (VISUAL_TAU_MIN - 0.5))


def _tiny_plan(visual: bool) -> SolvePlan:
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
    return SolvePlan(
        mask=mask,
        tunnel=tunnel,
        fluid=FluidSpec(),
        preset=RunPreset("Test", 48, 96, "visual test"),
        u_lattice=VISUAL_U_LATTICE,
        smagorinsky=VISUAL_SMAGORINSKY,
        visual=visual,
    )


@pytest.fixture(scope="module")
def visual_result():
    return run_case(_tiny_plan(visual=True))


def test_visual_run_pins_tau_into_the_band(visual_result):
    assert VISUAL_TAU_MIN <= visual_result.units.tau <= VISUAL_TAU_MAX
    assert visual_result.units.viscosity_clamped is True
    assert visual_result.visual is True


def test_visual_report_says_not_calibrated(visual_result):
    text = summary(visual_result, "sphere.obj")
    assert "visualization run, effective Re" in text
    assert "Cd not calibrated" in text
    assert "No validation" in text
    eff = visual_result.fluid.speed * visual_result.fluid.reference_length / (
        visual_result.units.nu_physical
    )
    assert f"{eff:,.0f}" in text
