"""Thermal / Boussinesq coupling must actually run, and point the right way.

Regression cover for PLAN.md C11:

  - ``LBM3D.step()`` never called the thermal kernels at all, so
    ``temperature`` stayed exactly 0.0 while the engine still advertised a
    ``temperature`` field in ``get_field_names()``.
  - ``_apply_buoyancy_2d_nb`` applied ``g_y`` to D2Q9 directions 3 and 4,
    which are ``(-1, 0)`` and ``(0, -1)`` — so gravity partly drove the
    *streamwise* axis. Heated from below, the 2D engine produced a mean
    vertical velocity of **-1.1e-3**: the fluid sank when it should rise.
  - ``_apply_buoyancy_3d_nb`` only touched the six axis-aligned populations,
    so 12 of the 19 D3Q19 directions received no body force at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm3d import LBM3D


def _warm_cavity_3d(steps: int = 8) -> LBM3D:
    """Uniformly warm 3D cavity: a pure body force with no gradient.

    Steps are kept short deliberately. A closed box cannot absorb the momentum
    a sustained body force adds, so a long run rings like a mass on a spring
    and the mean velocity oscillates through zero regardless of correctness.
    """
    sim = LBM3D(width=16, height=16, depth=8)
    sim.u_inflow = 0.0
    sim.init_thermal(thermal_diffusivity=0.02, beta=0.01, g_y=-1.0, T_ref=0.0)
    sim.f_T[:] = sim.lattice.w[:, None, None, None] * 1.0
    sim.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
    sim.run(steps)
    return sim


def _warm_cavity_2d(steps: int = 8) -> LBM2D:
    sim = LBM2D(width=24, height=24, viscosity=0.02)
    sim.u_inflow = 0.0
    sim.init_thermal(thermal_diffusivity=0.02, beta=0.01, g_y=-1.0, T_ref=0.0)
    sim.f_T[:] = sim.lattice.w[:, None, None] * 1.0
    sim.initialize(rho=1.0, u=0.0, v=0.0)
    sim.run(steps)
    return sim


class TestThermalStepPath3D:
    def test_temperature_actually_evolves(self) -> None:
        """Before the fix, LBM3D never touched f_T so this stayed exactly 0.0."""
        sim = LBM3D(width=12, height=12, depth=6)
        sim.u_inflow = 0.0
        sim.init_thermal(thermal_diffusivity=0.05, beta=0.0)
        sim.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
        sim.set_temperature_boundary(1.0, region="bottom")
        assert float(sim.temperature.max()) == 0.0
        sim.run(10)
        assert float(sim.temperature.max()) > 0.05

    def test_temperature_distribution_is_advanced(self) -> None:
        """f_T must be collided and streamed, not left at its seeded values."""
        sim = LBM3D(width=12, height=12, depth=6)
        sim.u_inflow = 0.0
        sim.init_thermal(thermal_diffusivity=0.05, beta=0.0)
        sim.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
        sim.set_temperature_boundary(1.0, region="bottom")
        seeded = sim.f_T.copy()
        sim.run(10)
        assert not np.allclose(sim.f_T, seeded, atol=1e-6)

    def test_hot_fluid_is_driven_upward(self) -> None:
        """Gravity is -y and the cavity is warm, so the mean v must be positive."""
        sim = _warm_cavity_3d()
        assert float(sim.get_velocity()[..., 1].mean()) > 0.0

    def test_buoyancy_does_not_drive_unforced_axes(self) -> None:
        """A y-only body force must not inject momentum along x or z."""
        vel = _warm_cavity_3d().get_velocity()
        driven = float(vel[..., 1].mean())
        assert abs(float(vel[..., 0].mean())) < abs(driven) * 1e-3
        assert abs(float(vel[..., 2].mean())) < abs(driven) * 1e-3

    def test_step_is_stable(self) -> None:
        sim = _warm_cavity_3d()
        assert np.all(np.isfinite(sim.rho))
        assert np.all(np.isfinite(sim.temperature))
        assert np.all(np.isfinite(sim.f_T))
        assert float(sim.rho.min()) > 0.0

    def test_thermal_disabled_is_a_no_op(self) -> None:
        sim = LBM3D(width=12, height=12, depth=6)
        sim.u_inflow = 0.0
        sim.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
        assert not hasattr(sim, "temperature")
        sim.run(5)
        assert np.all(np.isfinite(sim.rho))

    def test_temperature_field_is_offered_only_when_enabled(self) -> None:
        off = LBM3D(width=12, height=12, depth=6)
        assert "temperature" not in off.get_field_names()
        on = LBM3D(width=12, height=12, depth=6)
        on.init_thermal()
        assert "temperature" in on.get_field_names()


class TestBuoyancyDirection:
    def test_2d_warm_cavity_is_driven_upward(self) -> None:
        """Catches the g_y -> f[3]/f[4] misindexing (directions -x and -y)."""
        assert float(_warm_cavity_2d().get_velocity()[:, :, 1].mean()) > 0.0

    def test_2d_buoyancy_momentum_is_axis_aligned(self) -> None:
        """A y-only force must not add momentum to the streamwise axis."""
        vel = _warm_cavity_2d().get_velocity()
        driven = float(vel[:, :, 1].mean())
        assert abs(float(vel[:, :, 0].mean())) < abs(driven) * 1e-3

    def test_3d_buoyancy_reaches_every_live_direction(self) -> None:
        """Exactly the directions with a y component must receive the force.

        The old kernel touched only the six axis-aligned populations, leaving
        12 of the 19 D3Q19 directions with no body force at all.
        """
        sim = LBM3D(width=8, height=8, depth=4)
        sim.u_inflow = 0.0
        sim.init_thermal(beta=0.01, g_y=-1.0, T_ref=0.0)
        sim.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
        sim.temperature[:, :, :] = 1.0
        before = sim.f.copy()
        sim.apply_buoyancy()
        delta = sim.f - before
        untouched = set(np.flatnonzero(np.all(np.abs(delta) < 1e-12, axis=(1, 2, 3))))
        expected = {i for i in range(19) if sim.lattice.cy[i] == 0}
        assert untouched == expected
        assert len(expected) < 19

    def test_2d_buoyancy_reaches_every_live_direction(self) -> None:
        sim = LBM2D(width=8, height=8)
        sim.init_thermal(beta=0.01, g_y=-1.0, T_ref=0.0)
        sim.temperature[:, :] = 1.0
        before = sim.f.copy()
        sim.apply_buoyancy()
        delta = sim.f - before
        untouched = set(np.flatnonzero(np.all(np.abs(delta) < 1e-12, axis=(1, 2))))
        expected = {i for i in range(9) if sim.lattice.cy[i] == 0}
        assert untouched == expected
        assert len(expected) < 9

    def test_buoyancy_respects_obstacles(self) -> None:
        sim = LBM3D(width=8, height=8, depth=4)
        sim.u_inflow = 0.0
        sim.init_thermal(beta=0.01, g_y=-1.0, T_ref=0.0)
        sim.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
        sim.temperature[:, :, :] = 1.0
        sim.obstacles[:, :, :] = True
        before = sim.f.copy()
        sim.apply_buoyancy()
        np.testing.assert_allclose(sim.f, before, atol=1e-12)

    def test_reference_temperature_produces_no_force(self) -> None:
        sim = LBM2D(width=8, height=8)
        sim.init_thermal(beta=0.01, g_y=-1.0, T_ref=0.7)
        sim.temperature[:, :] = 0.7
        before = sim.f.copy()
        sim.apply_buoyancy()
        np.testing.assert_allclose(sim.f, before, atol=1e-12)


class TestBuoyancyMomentumBalance:
    @pytest.mark.parametrize("direction", ["x", "y"])
    def test_added_momentum_matches_the_requested_force(self, direction: str) -> None:
        """The discrete force must deliver exactly F, not a distorted version.

        This is the property the hand-indexed populations broke: summing
        w_i * (F . c_i) * c_i over i must reproduce F for any direction.
        """
        beta, magnitude = 0.01, 1.0
        sim = LBM2D(width=8, height=8)
        g_x = magnitude if direction == "x" else 0.0
        g_y = magnitude if direction == "y" else 0.0
        sim.init_thermal(beta=beta, g_x=g_x, g_y=g_y, T_ref=0.0)
        sim.temperature[:, :] = 1.0
        before = sim.f.copy()
        sim.apply_buoyancy()
        delta = sim.f - before
        cells = 8 * 8
        added_x = float(np.sum(delta * sim.lattice.cx[:, None, None])) / cells
        added_y = float(np.sum(delta * sim.lattice.cy[:, None, None])) / cells
        expected = -beta * (1.0 - 0.0)
        assert added_x == pytest.approx(expected * g_x, rel=1e-5)
        assert added_y == pytest.approx(expected * g_y, rel=1e-5)


class TestTemperatureRegionNaming:
    """'top' and 'bottom' must mean the same rows as the moving-wall helpers.

    They were swapped: asking for a heated floor set the ceiling instead, so
    every Rayleigh-Benard setup was stratified upside down.
    """

    @staticmethod
    def _row_temperature(sim, row: int) -> float:
        # The temperature is the zeroth moment of f_T, not a weighted sum.
        return float(np.sum(sim.f_T[:, row, sim.width // 2]))

    def test_2d_bottom_is_row_zero(self) -> None:
        sim = LBM2D(width=16, height=16)
        sim.init_thermal()
        sim.set_temperature_boundary(1.0, region="bottom")
        assert self._row_temperature(sim, 0) == pytest.approx(1.0, rel=1e-5)
        assert self._row_temperature(sim, 15) == pytest.approx(0.0, abs=1e-6)

    def test_2d_top_is_the_last_row(self) -> None:
        sim = LBM2D(width=16, height=16)
        sim.init_thermal()
        sim.set_temperature_boundary(1.0, region="top")
        assert self._row_temperature(sim, 15) == pytest.approx(1.0, rel=1e-5)
        assert self._row_temperature(sim, 0) == pytest.approx(0.0, abs=1e-6)

    def test_3d_bottom_is_row_zero(self) -> None:
        # f_T is indexed (19, depth, height, width), so the height row is
        # axis 2 and 'bottom'/'top' are rows within it.
        sim = LBM3D(width=12, height=12, depth=6)
        sim.init_thermal()
        sim.set_temperature_boundary(1.0, region="bottom")
        assert float(np.sum(sim.f_T[:, 0, 0, 6])) == pytest.approx(1.0, rel=1e-5)
        assert float(np.sum(sim.f_T[:, 0, -1, 6])) == pytest.approx(0.0, abs=1e-6)

    def test_3d_top_is_the_last_row(self) -> None:
        sim = LBM3D(width=12, height=12, depth=6)
        sim.init_thermal()
        sim.set_temperature_boundary(1.0, region="top")
        assert float(np.sum(sim.f_T[:, 0, -1, 6])) == pytest.approx(1.0, rel=1e-5)
        assert float(np.sum(sim.f_T[:, 0, 0, 6])) == pytest.approx(0.0, abs=1e-6)

    def test_3d_lid_row_matches_the_moving_wall_lid(self) -> None:
        """The lid used for cavities must be the row 'top' refers to."""
        sim = LBM3D(width=12, height=12, depth=6)
        sim.init_thermal()
        sim.set_temperature_boundary(1.0, region="top")
        assert sim.f_T[:, :, -1, :].any()
        assert not sim.f_T[:, :, 0, :].any()
