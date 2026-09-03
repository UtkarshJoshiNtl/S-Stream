from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d import LBM2D


@pytest.fixture
def sim() -> LBM2D:
    return LBM2D(width=32, height=32, viscosity=0.02)


class TestInit:
    def test_default_grid(self) -> None:
        s = LBM2D()
        assert s.width == 128
        assert s.height == 128

    def test_custom_grid(self, sim: LBM2D) -> None:
        assert sim.width == 32
        assert sim.height == 32

    def test_omega_from_viscosity(self) -> None:
        s = LBM2D(viscosity=0.1)
        expected = 1.0 / (3.0 * 0.1 + 0.5)
        assert s.omega == pytest.approx(expected)

    def test_initial_fields_not_nan(self, sim: LBM2D) -> None:
        assert not np.any(np.isnan(sim.f))
        assert not np.any(np.isnan(sim.rho))
        assert not np.any(np.isnan(sim.u))
        assert not np.any(np.isnan(sim.v))

    def test_initial_density_uniform(self, sim: LBM2D) -> None:
        assert np.allclose(sim.rho, 1.0)

    def test_initial_velocity_uniform(self, sim: LBM2D) -> None:
        # Quiescent init is the default (low-Mach channel drives via BC).
        assert np.allclose(sim.u, 0.0)
        assert np.allclose(sim.v, 0.0)


class TestEquilibrium:
    def test_equilibrium_total_density(self, sim: LBM2D) -> None:
        rho = np.ones((32, 32), dtype=np.float32)
        u = np.zeros((32, 32), dtype=np.float32)
        v = np.zeros((32, 32), dtype=np.float32)
        feq = sim.lattice.equilibrium(rho, u, v)
        rho_sum = np.sum(feq, axis=0)
        assert np.allclose(rho_sum, 1.0, atol=1e-6)

    def test_equilibrium_x_momentum(self, sim: LBM2D) -> None:
        rho = np.ones((32, 32), dtype=np.float32)
        u_fld = np.full((32, 32), 0.1, dtype=np.float32)
        v_fld = np.zeros((32, 32), dtype=np.float32)
        feq = sim.lattice.equilibrium(rho, u_fld, v_fld)
        u_comp = np.sum(feq * sim.lattice.cx[:, np.newaxis, np.newaxis], axis=0)
        assert np.allclose(u_comp, 0.1, atol=1e-6)


class TestCollision:
    def test_mass_conserved(self, sim: LBM2D) -> None:
        mass_before = np.sum(sim.f)
        sim.collision()
        mass_after = np.sum(sim.f)
        assert mass_after == pytest.approx(mass_before, rel=1e-5)

    def test_output_shape(self, sim: LBM2D) -> None:
        sim.collision()
        assert sim.f.shape == (9, 32, 32)


class TestStreaming:
    def test_mass_conserved(self, sim: LBM2D) -> None:
        mass_before = np.sum(sim.f)
        sim.streaming()
        mass_after = np.sum(sim.f)
        assert mass_after == pytest.approx(mass_before)


class TestBoundaries:
    def test_inflow_sets_left_bc(self, sim: LBM2D) -> None:
        sim.initialize(rho=1.0, u=0.0, v=0.0)
        sim.u_inflow = 0.15
        sim.apply_inflow()
        c = sim.lattice.cx[:, np.newaxis, np.newaxis]
        u_left = np.sum(sim.f * c, axis=0) / np.sum(sim.f, axis=0)
        assert np.allclose(u_left[:, 0], 0.15, atol=1e-6)

    def test_outflow_zero_gradient(self, sim: LBM2D) -> None:
        sim.apply_outflow()
        for i in range(9):
            assert np.allclose(sim.f[i, :, -1], sim.f[i, :, -2])

    def test_walls_bounce_back_top(self, sim: LBM2D) -> None:
        sim.initialize(rho=1.0, u=0.1, v=0.0)
        f_before = sim.f.copy()
        sim.apply_walls()
        for i in range(9):
            opp_i = sim.lattice.opp[i]
            np.testing.assert_allclose(
                sim.f[i, 0, :],
                f_before[opp_i, 0, :],
                err_msg=f"f[{i}] at top wall should equal original f[{opp_i}]",
            )


class TestObstacles:
    def test_add_obstacle(self, sim: LBM2D) -> None:
        sim.add_obstacle(16, 16, radius=5)
        assert sim.obstacles[16, 16]

    def test_clear_obstacles(self, sim: LBM2D) -> None:
        sim.add_obstacle(16, 16, radius=5)
        sim.clear_obstacles()
        assert not np.any(sim.obstacles)

    def test_obstacle_bounce_back(self, sim: LBM2D) -> None:
        sim.obstacles[16, 16] = True
        f_before = sim.f.copy()
        sim.apply_obstacles()
        for i in range(9):
            opp_i = sim.lattice.opp[i]
            if i != opp_i:
                np.testing.assert_allclose(
                    sim.f[i, 16, 16],
                    f_before[opp_i, 16, 16],
                    err_msg=f"f[{i}] should equal original f[{opp_i}] at obstacle",
                )


class TestFusedPathContract:
    """The fused fast path must never substitute a different obstacle BC."""

    @staticmethod
    def _run(fused: bool, obstacle_bc: str) -> np.ndarray:
        sim = LBM2D(width=48, height=40, viscosity=0.02)
        sim.u_inflow = 0.05
        sim.initialize()
        sim.add_obstacle(24, 20, radius=5)
        sim.obstacle_bc = obstacle_bc
        sim.use_fused = fused
        sim.run(30)
        return sim.get_density()

    def test_obstacle_scene_never_uses_fused_kernel(self) -> None:
        sim = LBM2D(width=32, height=32)
        assert sim._can_use_fused() is True
        sim.add_obstacle(16, 16, radius=4)
        assert sim._can_use_fused() is False

    def test_fused_flag_does_not_change_obstacle_physics(self) -> None:
        """use_fused must be a performance switch only, never a physics one.

        The fused kernel performed an on-node swap while the configured default
        is halfway bounce-back, so enabling the fast path silently moved the
        no-slip surface and changed drag.
        """
        np.testing.assert_allclose(
            self._run(True, "halfway"),
            self._run(False, "halfway"),
            atol=1e-6,
            err_msg="fused flag changed the physics of an obstacle scene",
        )

    def test_configured_obstacle_bc_is_actually_applied(self) -> None:
        """halfway and fullway must still give different answers.

        Guards against 'fixing' the contract by making obstacle_bc inert.
        """
        assert not np.allclose(
            self._run(False, "halfway"),
            self._run(False, "fullway"),
            atol=1e-6,
        )

    def test_empty_scene_still_uses_fused_kernel(self) -> None:
        sim = LBM2D(width=32, height=32)
        sim.clear_obstacles()
        assert sim._can_use_fused() is True


class TestPeriodicDomain:
    def test_obstacles_apply_in_periodic_mode(self) -> None:
        """Periodic streaming must not pass through a solid.

        apply_boundary_conditions() used to return before obstacle handling in
        periodic mode, so the obstacle mask was a render-only overlay there.
        """
        sim = LBM2D(width=32, height=32, viscosity=0.02)
        sim.domain_mode = "periodic"
        sim.initialize()
        sim.obstacles[12, 16] = True  # (x=16, y=12)
        assert not sim.obstacles[11, 16]
        direction = 2  # (0, +1) points at the solid
        sim.f[direction, 11, 16] = 0.123
        sim.apply_boundary_conditions()
        opp = int(sim.lattice.opp[direction])
        assert sim.f[opp, 11, 16] == pytest.approx(sim.f[direction, 11, 16], abs=1e-5)
        assert sim.f[direction, 11, 16] == pytest.approx(0.123, abs=1e-6)

    def test_periodic_run_keeps_obstacle_solid(self) -> None:
        sim = LBM2D(width=32, height=32, viscosity=0.02)
        sim.domain_mode = "periodic"
        sim.initialize()
        sim.add_obstacle(16, 16, radius=4)
        sim.run(40)
        assert np.all(np.isfinite(sim.rho))
        assert np.all(sim.rho > 0)


class TestHalfwayBounceBack:
    def test_opposite_solid_neighbours_do_not_alias(self) -> None:
        """A fluid cell walled on both sides must reflect, not swap to zero.

        The kernel wrote f[opp[i]] = f[i] in direction order, so for a cell
        with solids on both sides the second write read the value the first
        write had just produced. Sources are now read before any write.
        """
        sim = LBM2D(width=32, height=32)
        sim.initialize()
        sim.obstacles[16, 15] = True  # (x=15, y=16)
        sim.obstacles[16, 17] = True  # (x=17, y=16)
        sim.f[1, 16, 16] = 0.7  # +x heading toward the solid at x=17
        sim.f[3, 16, 16] = 0.2  # -x heading toward the solid at x=15
        sim.apply_obstacles()
        assert sim.f[3, 16, 16] == pytest.approx(0.7, abs=1e-6)
        assert sim.f[1, 16, 16] == pytest.approx(0.2, abs=1e-6)


class TestStep:
    def test_step_changes_state(self, sim: LBM2D) -> None:
        f_before = sim.f.copy()
        sim.step()
        assert not np.allclose(sim.f, f_before)

    def test_multiple_steps(self, sim: LBM2D) -> None:
        sim.run(10)
        assert sim.f.shape == (9, 32, 32)


class TestGetter:
    def test_get_density(self, sim: LBM2D) -> None:
        d = sim.get_density()
        assert d.shape == (32, 32)
        assert np.allclose(d, sim.rho)

    def test_get_velocity(self, sim: LBM2D) -> None:
        v = sim.get_velocity()
        assert v.shape == (32, 32, 2)
        assert np.allclose(v[:, :, 0], sim.u)
        assert np.allclose(v[:, :, 1], sim.v)

    def test_get_smoke(self, sim: LBM2D) -> None:
        s = sim.get_smoke()
        assert s.shape == (32, 32)
        assert np.allclose(s, sim.smoke)


class TestSmoke:
    def test_initial_smoke_zero(self, sim: LBM2D) -> None:
        assert np.allclose(sim.smoke, 0.0)

    def test_add_emitter(self, sim: LBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.05)
        assert len(sim.emitters) == 1
        assert sim.emitters[0] == (16, 16, 0.05)

    def test_clear_emitters(self, sim: LBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.05)
        sim.clear_emitters()
        assert len(sim.emitters) == 0

    def test_emitter_injects_smoke(self, sim: LBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.1)
        sim.apply_emitters()
        assert sim.smoke[16, 16] == pytest.approx(0.1)

    def test_smoke_advection_preserves_mass(self, sim: LBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.u[:] = 0.0
        sim.v[:] = 0.0
        mass_before = np.sum(sim.smoke)
        sim.advect_smoke()
        mass_after = np.sum(sim.smoke)
        assert mass_after == pytest.approx(mass_before, rel=1e-10)

    def test_smoke_advection_moves_downstream(self, sim: LBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.u[:] = 1.0
        sim.v[:] = 0.0
        sim.advect_smoke()
        assert sim.smoke[16, 17] > 0.1
        assert sim.smoke[16, 16] < 1.0

    def test_decay_reduces_smoke(self, sim: LBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.decay_smoke()
        assert sim.smoke[16, 16] == pytest.approx(0.999)

    def test_diffusion_spreads_smoke(self, sim: LBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.diffuse_smoke()
        assert sim.smoke[15, 16] > 0
        assert sim.smoke[16, 15] > 0
        assert sim.smoke[17, 16] > 0
        assert sim.smoke[16, 17] > 0

    def test_obstacles_clear_smoke(self, sim: LBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.obstacles[16, 16] = True
        sim.step()
        assert sim.smoke[16, 16] == 0.0

    def test_smoke_step_integration(self, sim: LBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.1)
        sim.step()
        assert np.sum(sim.smoke) > 0
        assert np.all(sim.smoke >= 0)

    def test_initialize_clears_smoke(self, sim: LBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.emitters.append((16, 16, 0.1))
        sim.initialize(rho=1.0, u=0.1, v=0.0)
        assert np.allclose(sim.smoke, 0.0)
        assert len(sim.emitters) == 0


class TestBoundaryConditionsRespectObstacles:
    """No domain boundary condition may write populations into a solid.

    A blockage that touches the inlet used to receive a fresh equilibrium
    every step, which made the solid a mass source and quietly corrupted
    both blockage ratio and drag.
    """

    @staticmethod
    def _channel(**kwargs) -> LBM2D:
        sim = LBM2D(width=32, height=32, viscosity=0.02)
        for key, value in kwargs.items():
            setattr(sim, key, value)
        sim.initialize()
        return sim

    def test_inflow_skips_solid_inlet_cells(self) -> None:
        sim = self._channel(u_inflow=0.1)
        sim.obstacles[10, 0] = True  # (x=0, y=10)
        sim.f[:, 10, 0] = 0.321
        sim.apply_inflow()
        assert sim.f[0, 10, 0] == pytest.approx(0.321, abs=1e-6)
        assert sim.f[1, 11, 0] != pytest.approx(0.321, abs=1e-6)

    def test_inflow_zou_he_skips_solid_inlet_cells(self) -> None:
        sim = self._channel(u_inflow=0.1, use_zou_he=True)
        sim.obstacles[10, 0] = True
        sim.f[:, 10, 0] = 0.321
        sim.apply_inflow()
        assert sim.f[0, 10, 0] == pytest.approx(0.321, abs=1e-6)

    def test_outflow_skips_solid_outlet_cells(self) -> None:
        sim = self._channel()
        sim.obstacles[20, 31] = True  # (x=31, y=20)
        sim.f[:, 20, 31] = 0.654
        sim.apply_outflow()
        assert sim.f[0, 20, 31] == pytest.approx(0.654, abs=1e-6)

    def test_outflow_zou_he_skips_solid_outlet_cells(self) -> None:
        sim = self._channel(use_zou_he=True)
        sim.obstacles[20, 31] = True
        # Zou-He writes f3/f6/f7 at the outlet, so assert on one of those.
        sim.f[3, 20, 31] = 0.654
        sim.f[6, 20, 31] = 0.654
        sim.f[7, 20, 31] = 0.654
        sim.apply_outflow()
        assert sim.f[3, 20, 31] == pytest.approx(0.654, abs=1e-6)
        assert sim.f[6, 20, 31] == pytest.approx(0.654, abs=1e-6)

    def test_walls_skip_solid_edge_cells(self) -> None:
        sim = self._channel()
        sim.obstacles[0, 5] = True  # (x=5, y=0) on the bottom wall
        # Break the resting symmetry: on a symmetric equilibrium the wall
        # swap is a no-op and would hide a missing mask.
        sim.f[1, 0, 5] = 0.31
        sim.f[3, 0, 5] = 0.02
        before = sim.f[:, 0, 5].copy()
        sim.apply_walls()
        np.testing.assert_allclose(sim.f[:, 0, 5], before, atol=1e-6)
        # Control: the same swap does happen on the neighbouring free cell.
        sim.f[1, 0, 6] = 0.31
        sim.f[3, 0, 6] = 0.02
        sim.apply_walls()
        assert sim.f[1, 0, 6] == pytest.approx(0.02, abs=1e-6)

    def test_cavity_lid_skips_solid_cells(self) -> None:
        sim = self._channel(domain_mode="cavity", lid_velocity=0.1)
        sim.obstacles[31, 7] = True  # (x=7, y=31) on the lid row
        sim.f[2, 31, 7] = 0.44
        before = sim.f[:, 31, 7].copy()
        sim.apply_boundary_conditions()
        np.testing.assert_allclose(sim.f[:, 31, 7], before, atol=1e-6)

    def test_cavity_side_wall_skips_solid_cells(self) -> None:
        sim = self._channel(domain_mode="cavity", lid_velocity=0.1)
        sim.obstacles[9, 0] = True  # (x=0, y=9) on the left side wall
        sim.f[1, 9, 0] = 0.31
        sim.f[3, 9, 0] = 0.02
        before = sim.f[:, 9, 0].copy()
        sim.apply_boundary_conditions()
        np.testing.assert_allclose(sim.f[:, 9, 0], before, atol=1e-6)

    def test_channel_boundaries_leave_solid_cells_untouched(self) -> None:
        """A full boundary pass must not alter a single solid population.

        The inlet used to rewrite the whole left column with a fresh
        equilibrium, so a blockage touching x=0 acted as a mass source rather
        than a wall.
        """
        sim = self._channel(u_inflow=0.08)
        sim.f += np.linspace(0.01, 0.09, 9, dtype=np.float32)[:, None, None]
        sim.obstacles[10:20, 0] = True
        sim.obstacles[20:26, 31] = True
        sim.obstacles[0, 3:9] = True
        sim.obstacles[31, 3:9] = True
        before = sim.f[:, sim.obstacles].copy()
        sim.apply_boundary_conditions()
        np.testing.assert_allclose(sim.f[:, sim.obstacles], before, atol=1e-6)
