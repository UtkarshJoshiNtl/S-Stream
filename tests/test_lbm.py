from __future__ import annotations

import numpy as np
import pytest
from cpu_lbm import CPULBM2D


@pytest.fixture
def sim() -> CPULBM2D:
    return CPULBM2D(width=32, height=32, viscosity=0.02)


class TestInit:
    def test_default_grid(self) -> None:
        s = CPULBM2D()
        assert s.width == 128
        assert s.height == 128

    def test_custom_grid(self, sim: CPULBM2D) -> None:
        assert sim.width == 32
        assert sim.height == 32

    def test_omega_from_viscosity(self) -> None:
        s = CPULBM2D(viscosity=0.1)
        expected = 1.0 / (3.0 * 0.1 + 0.5)
        assert s.omega == pytest.approx(expected)

    def test_initial_fields_not_nan(self, sim: CPULBM2D) -> None:
        assert not np.any(np.isnan(sim.f))
        assert not np.any(np.isnan(sim.rho))
        assert not np.any(np.isnan(sim.u))
        assert not np.any(np.isnan(sim.v))

    def test_initial_density_uniform(self, sim: CPULBM2D) -> None:
        assert np.allclose(sim.rho, 1.0)

    def test_initial_velocity_uniform(self, sim: CPULBM2D) -> None:
        assert np.allclose(sim.u, 0.1)
        assert np.allclose(sim.v, 0.0)


class TestEquilibrium:
    def test_equilibrium_total_density(self, sim: CPULBM2D) -> None:
        rho = np.ones((32, 32))
        u = np.zeros((32, 32))
        v = np.zeros((32, 32))
        feq = sim.equilibrium(rho, u, v)
        rho_sum = np.sum(feq, axis=0)
        assert np.allclose(rho_sum, 1.0)

    def test_equilibrium_x_momentum(self, sim: CPULBM2D) -> None:
        rho = np.ones((32, 32))
        u_fld = np.full((32, 32), 0.1)
        v_fld = np.zeros((32, 32))
        feq = sim.equilibrium(rho, u_fld, v_fld)
        u_comp = np.sum(feq * sim.cx[:, np.newaxis, np.newaxis], axis=0)
        assert np.allclose(u_comp, 0.1)


class TestCollision:
    def test_mass_conserved(self, sim: CPULBM2D) -> None:
        mass_before = np.sum(sim.f)
        sim.collision()
        mass_after = np.sum(sim.f)
        assert mass_after == pytest.approx(mass_before, rel=1e-10)

    def test_output_shape(self, sim: CPULBM2D) -> None:
        sim.collision()
        assert sim.f.shape == (9, 32, 32)


class TestStreaming:
    def test_mass_conserved(self, sim: CPULBM2D) -> None:
        mass_before = np.sum(sim.f)
        sim.streaming()
        mass_after = np.sum(sim.f)
        assert mass_after == pytest.approx(mass_before)


class TestBoundaries:
    def test_inflow_sets_left_bc(self, sim: CPULBM2D) -> None:
        sim.initialize(rho=1.0, u=0.0, v=0.0)
        sim.apply_inflow(u_inflow=0.15)
        u_left = np.sum(sim.f * sim.cx[:, np.newaxis, np.newaxis], axis=0) / np.sum(
            sim.f, axis=0
        )
        assert np.allclose(u_left[:, 0], 0.15, atol=1e-6)

    def test_outflow_zero_gradient(self, sim: CPULBM2D) -> None:
        sim.apply_outflow()
        for i in range(9):
            assert np.allclose(sim.f[i, :, -1], sim.f[i, :, -2])

    def test_walls_bounce_back_top(self, sim: CPULBM2D) -> None:
        sim.initialize(rho=1.0, u=0.1, v=0.0)
        sim.apply_walls()
        for i in range(9):
            assert np.allclose(sim.f[i, 0, :], sim.f[sim.opp[i], 0, :])


class TestObstacles:
    def test_add_obstacle(self, sim: CPULBM2D) -> None:
        sim.add_obstacle(16, 16, radius=5)
        assert sim.obstacles[16, 16]

    def test_clear_obstacles(self, sim: CPULBM2D) -> None:
        sim.add_obstacle(16, 16, radius=5)
        sim.clear_obstacles()
        assert not np.any(sim.obstacles)

    def test_obstacle_bounce_back(self, sim: CPULBM2D) -> None:
        sim.obstacles[16, 16] = True
        f_before = sim.f.copy()
        sim.apply_obstacles()
        assert np.any(sim.f != f_before)


class TestStep:
    def test_step_changes_state(self, sim: CPULBM2D) -> None:
        f_before = sim.f.copy()
        sim.step()
        assert not np.allclose(sim.f, f_before)

    def test_multiple_steps(self, sim: CPULBM2D) -> None:
        sim.run(10)
        assert sim.f.shape == (9, 32, 32)


class TestGetter:
    def test_get_density(self, sim: CPULBM2D) -> None:
        d = sim.get_density()
        assert d.shape == (32, 32)
        assert np.allclose(d, sim.rho)

    def test_get_velocity(self, sim: CPULBM2D) -> None:
        v = sim.get_velocity()
        assert v.shape == (32, 32, 2)
        assert np.allclose(v[:, :, 0], sim.u)
        assert np.allclose(v[:, :, 1], sim.v)

    def test_get_smoke(self, sim: CPULBM2D) -> None:
        s = sim.get_smoke()
        assert s.shape == (32, 32)
        assert np.allclose(s, sim.smoke)


class TestSmoke:
    def test_initial_smoke_zero(self, sim: CPULBM2D) -> None:
        assert np.allclose(sim.smoke, 0.0)

    def test_add_emitter(self, sim: CPULBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.05)
        assert len(sim.emitters) == 1
        assert sim.emitters[0] == (16, 16, 0.05)

    def test_clear_emitters(self, sim: CPULBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.05)
        sim.clear_emitters()
        assert len(sim.emitters) == 0

    def test_emitter_injects_smoke(self, sim: CPULBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.1)
        sim.apply_emitters()
        assert sim.smoke[16, 16] == pytest.approx(0.1)

    def test_smoke_advection_preserves_mass(self, sim: CPULBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.u[:] = 0.0
        sim.v[:] = 0.0
        mass_before = np.sum(sim.smoke)
        sim.advect_smoke()
        mass_after = np.sum(sim.smoke)
        assert mass_after == pytest.approx(mass_before, rel=1e-10)

    def test_smoke_advection_moves_downstream(self, sim: CPULBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.u[:] = 1.0
        sim.v[:] = 0.0
        sim.advect_smoke()
        assert sim.smoke[16, 17] > 0.1
        assert sim.smoke[16, 16] < 1.0

    def test_decay_reduces_smoke(self, sim: CPULBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.decay_smoke()
        assert sim.smoke[16, 16] == pytest.approx(0.999)

    def test_diffusion_spreads_smoke(self, sim: CPULBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.diffuse_smoke()
        assert sim.smoke[15, 16] > 0
        assert sim.smoke[16, 15] > 0
        assert sim.smoke[17, 16] > 0
        assert sim.smoke[16, 17] > 0

    def test_obstacles_clear_smoke(self, sim: CPULBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.obstacles[16, 16] = True
        sim.step()
        assert sim.smoke[16, 16] == 0.0

    def test_smoke_step_integration(self, sim: CPULBM2D) -> None:
        sim.add_emitter(16, 16, strength=0.1)
        sim.step()
        assert np.sum(sim.smoke) > 0
        assert np.all(sim.smoke >= 0)

    def test_initialize_clears_smoke(self, sim: CPULBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        sim.emitters.append((16, 16, 0.1))
        sim.initialize(rho=1.0, u=0.1, v=0.0)
        assert np.allclose(sim.smoke, 0.0)
        assert len(sim.emitters) == 0
