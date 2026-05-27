from __future__ import annotations

import numpy as np
import pytest
from cpu_lbm3d import CPULBM3D


@pytest.fixture
def sim() -> CPULBM3D:
    return CPULBM3D(width=16, height=16, depth=16, viscosity=0.02)


class TestInit:
    def test_default_grid(self) -> None:
        s = CPULBM3D()
        assert s.width == 128
        assert s.height == 128
        assert s.depth == 64

    def test_custom_grid(self, sim: CPULBM3D) -> None:
        assert sim.width == 16
        assert sim.height == 16
        assert sim.depth == 16

    def test_omega_from_viscosity(self) -> None:
        s = CPULBM3D(viscosity=0.1)
        expected = 1.0 / (3.0 * 0.1 + 0.5)
        assert s.omega == pytest.approx(expected)

    def test_initial_fields_not_nan(self, sim: CPULBM3D) -> None:
        assert not np.any(np.isnan(sim.f))
        assert not np.any(np.isnan(sim.rho))
        assert not np.any(np.isnan(sim.u))
        assert not np.any(np.isnan(sim.v))
        assert not np.any(np.isnan(sim.w_vel))

    def test_initial_density_uniform(self, sim: CPULBM3D) -> None:
        assert np.allclose(sim.rho, 1.0)

    def test_initial_velocity_uniform(self, sim: CPULBM3D) -> None:
        assert np.allclose(sim.u, 0.1)
        assert np.allclose(sim.v, 0.0)
        assert np.allclose(sim.w_vel, 0.0)

    def test_f_shape(self, sim: CPULBM3D) -> None:
        assert sim.f.shape == (19, 16, 16, 16)

    def test_omega_stable_assert(self) -> None:
        with pytest.raises(AssertionError):
            CPULBM3D(viscosity=0.0)

    def test_grid_shapes(self, sim: CPULBM3D) -> None:
        assert sim.u.shape == (16, 16, 16)
        assert sim.v.shape == (16, 16, 16)
        assert sim.w_vel.shape == (16, 16, 16)
        assert sim.rho.shape == (16, 16, 16)
        assert sim.obstacles.shape == (16, 16, 16)
        assert sim.smoke.shape == (16, 16, 16)


class TestEquilibrium:
    def test_equilibrium_total_density(self, sim: CPULBM3D) -> None:
        rho = np.ones((16, 16, 16))
        u = np.zeros((16, 16, 16))
        v = np.zeros((16, 16, 16))
        w = np.zeros((16, 16, 16))
        feq = sim.equilibrium(rho, u, v, w)
        rho_sum = np.sum(feq, axis=0)
        assert np.allclose(rho_sum, 1.0)

    def test_equilibrium_x_momentum(self, sim: CPULBM3D) -> None:
        rho = np.ones((16, 16, 16))
        u_fld = np.full((16, 16, 16), 0.1)
        v_fld = np.zeros((16, 16, 16))
        w_fld = np.zeros((16, 16, 16))
        feq = sim.equilibrium(rho, u_fld, v_fld, w_fld)
        u_comp = np.sum(feq * sim.cx[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        assert np.allclose(u_comp, 0.1, atol=1e-12)

    def test_equilibrium_y_momentum(self, sim: CPULBM3D) -> None:
        rho = np.ones((16, 16, 16))
        u_fld = np.zeros((16, 16, 16))
        v_fld = np.full((16, 16, 16), 0.05)
        w_fld = np.zeros((16, 16, 16))
        feq = sim.equilibrium(rho, u_fld, v_fld, w_fld)
        v_comp = np.sum(feq * sim.cy[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        assert np.allclose(v_comp, 0.05, atol=1e-12)

    def test_equilibrium_z_momentum(self, sim: CPULBM3D) -> None:
        rho = np.ones((16, 16, 16))
        u_fld = np.zeros((16, 16, 16))
        v_fld = np.zeros((16, 16, 16))
        w_fld = np.full((16, 16, 16), 0.03)
        feq = sim.equilibrium(rho, u_fld, v_fld, w_fld)
        w_comp = np.sum(feq * sim.cz[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        assert np.allclose(w_comp, 0.03, atol=1e-12)


class TestCollision:
    def test_mass_conserved(self, sim: CPULBM3D) -> None:
        mass_before = np.sum(sim.f)
        sim.collision()
        mass_after = np.sum(sim.f)
        assert mass_after == pytest.approx(mass_before, rel=1e-10)

    def test_output_shape(self, sim: CPULBM3D) -> None:
        sim.collision()
        assert sim.f.shape == (19, 16, 16, 16)

    def test_momentum_conserved(self, sim: CPULBM3D) -> None:
        c = sim.cx[:, np.newaxis, np.newaxis, np.newaxis]
        mx_before = np.sum(sim.f * c)
        c = sim.cy[:, np.newaxis, np.newaxis, np.newaxis]
        my_before = np.sum(sim.f * c)
        c = sim.cz[:, np.newaxis, np.newaxis, np.newaxis]
        mz_before = np.sum(sim.f * c)
        sim.collision()
        c = sim.cx[:, np.newaxis, np.newaxis, np.newaxis]
        mx_after = np.sum(sim.f * c)
        c = sim.cy[:, np.newaxis, np.newaxis, np.newaxis]
        my_after = np.sum(sim.f * c)
        c = sim.cz[:, np.newaxis, np.newaxis, np.newaxis]
        mz_after = np.sum(sim.f * c)
        assert mx_after == pytest.approx(mx_before, rel=1e-10)
        assert my_after == pytest.approx(my_before, rel=1e-10)
        assert mz_after == pytest.approx(mz_before, rel=1e-10)


class TestStreaming:
    def test_mass_conserved(self, sim: CPULBM3D) -> None:
        mass_before = np.sum(sim.f)
        sim.streaming()
        mass_after = np.sum(sim.f)
        assert mass_after == pytest.approx(mass_before)


class TestBoundaries:
    def test_inflow_sets_left_bc(self, sim: CPULBM3D) -> None:
        sim.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
        sim.apply_inflow(u_inflow=0.15)
        c = sim.cx[:, np.newaxis, np.newaxis, np.newaxis]
        u_left = np.sum(sim.f * c, axis=0) / np.sum(sim.f, axis=0)
        assert np.allclose(u_left[:, :, 0], 0.15, atol=1e-6)

    def test_outflow_zero_gradient(self, sim: CPULBM3D) -> None:
        sim.apply_outflow()
        for i in range(19):
            assert np.allclose(sim.f[i, :, :, -1], sim.f[i, :, :, -2])

    def test_walls_bounce_back_top_bottom(self, sim: CPULBM3D) -> None:
        sim.initialize(rho=1.0, u=0.1, v=0.0, w=0.0)
        sim.apply_walls()
        for i in range(19):
            assert np.allclose(sim.f[i, :, 0, :], sim.f[sim.opp[i], :, 0, :])
            assert np.allclose(sim.f[i, :, -1, :], sim.f[sim.opp[i], :, -1, :])

    def test_walls_bounce_back_front_back(self, sim: CPULBM3D) -> None:
        sim.initialize(rho=1.0, u=0.1, v=0.0, w=0.0)
        sim.apply_walls()
        for i in range(19):
            assert np.allclose(sim.f[i, 0, :, :], sim.f[sim.opp[i], 0, :, :])
            assert np.allclose(sim.f[i, -1, :, :], sim.f[sim.opp[i], -1, :, :])


class TestObstacles:
    def test_add_obstacle_sphere(self, sim: CPULBM3D) -> None:
        sim.add_obstacle_sphere(8, 8, 8, radius=3)
        assert sim.obstacles[8, 8, 8]

    def test_clear_obstacles(self, sim: CPULBM3D) -> None:
        sim.add_obstacle_sphere(8, 8, 8, radius=3)
        sim.clear_obstacles()
        assert not np.any(sim.obstacles)

    def test_obstacle_bounce_back(self, sim: CPULBM3D) -> None:
        sim.obstacles[8, 8, 8] = True
        f_before = sim.f.copy()
        sim.apply_obstacles()
        assert np.any(sim.f != f_before)

    def test_obstacle_smoke_cleared(self, sim: CPULBM3D) -> None:
        sim.smoke[8, 8, 8] = 1.0
        sim.obstacles[8, 8, 8] = True
        sim.step()
        assert sim.smoke[8, 8, 8] == 0.0


class TestStep:
    def test_step_changes_state(self, sim: CPULBM3D) -> None:
        f_before = sim.f.copy()
        sim.step()
        assert not np.allclose(sim.f, f_before)

    def test_multiple_steps(self, sim: CPULBM3D) -> None:
        sim.run(10)
        assert sim.f.shape == (19, 16, 16, 16)

    def test_no_nan_after_steps(self, sim: CPULBM3D) -> None:
        sim.run(50)
        assert not np.any(np.isnan(sim.f))
        assert not np.any(np.isnan(sim.rho))
        assert not np.any(np.isnan(sim.u))
        assert not np.any(np.isnan(sim.v))
        assert not np.any(np.isnan(sim.w_vel))
        assert not np.any(np.isnan(sim.smoke))


class TestGetter:
    def test_get_density(self, sim: CPULBM3D) -> None:
        d = sim.get_density()
        assert d.shape == (16, 16, 16)
        assert np.allclose(d, sim.rho)

    def test_get_velocity(self, sim: CPULBM3D) -> None:
        v = sim.get_velocity()
        assert v.shape == (16, 16, 16, 3)
        assert np.allclose(v[:, :, :, 0], sim.u)
        assert np.allclose(v[:, :, :, 1], sim.v)
        assert np.allclose(v[:, :, :, 2], sim.w_vel)

    def test_get_smoke(self, sim: CPULBM3D) -> None:
        s = sim.get_smoke()
        assert s.shape == (16, 16, 16)
        assert np.allclose(s, sim.smoke)


class TestSmoke:
    def test_initial_smoke_zero(self, sim: CPULBM3D) -> None:
        assert np.allclose(sim.smoke, 0.0)

    def test_add_emitter(self, sim: CPULBM3D) -> None:
        sim.add_emitter(8, 8, 8, strength=0.05)
        assert len(sim.emitters) == 1
        assert sim.emitters[0] == (8, 8, 8, 0.05)

    def test_clear_emitters(self, sim: CPULBM3D) -> None:
        sim.add_emitter(8, 8, 8, strength=0.05)
        sim.clear_emitters()
        assert len(sim.emitters) == 0

    def test_emitter_injects_smoke(self, sim: CPULBM3D) -> None:
        sim.add_emitter(8, 8, 8, strength=0.1)
        sim.apply_emitters()
        assert sim.smoke[8, 8, 8] == pytest.approx(0.1)

    def test_smoke_advection_preserves_mass_at_zero_vel(self, sim: CPULBM3D) -> None:
        sim.smoke[8, 8, 8] = 1.0
        sim.u[:] = 0.0
        sim.v[:] = 0.0
        sim.w_vel[:] = 0.0
        mass_before = np.sum(sim.smoke)
        sim.advect_smoke()
        mass_after = np.sum(sim.smoke)
        assert mass_after == pytest.approx(mass_before, rel=1e-10)

    def test_smoke_advection_moves_downstream_x(self, sim: CPULBM3D) -> None:
        sim.smoke[8, 8, 8] = 1.0
        sim.u[:] = 1.0
        sim.v[:] = 0.0
        sim.w_vel[:] = 0.0
        sim.advect_smoke()
        assert sim.smoke[8, 8, 9] > 0.1
        assert sim.smoke[8, 8, 8] < 1.0

    def test_smoke_advection_moves_downstream_z(self, sim: CPULBM3D) -> None:
        sim.smoke[4, 8, 8] = 1.0
        sim.u[:] = 0.0
        sim.v[:] = 0.0
        sim.w_vel[:] = 1.0
        sim.advect_smoke()
        assert sim.smoke[5, 8, 8] > 0.1
        assert sim.smoke[4, 8, 8] < 1.0

    def test_decay_reduces_smoke(self, sim: CPULBM3D) -> None:
        sim.smoke[8, 8, 8] = 1.0
        sim.decay_smoke()
        assert sim.smoke[8, 8, 8] == pytest.approx(0.999)

    def test_diffusion_spreads_smoke(self, sim: CPULBM3D) -> None:
        sim.smoke[8, 8, 8] = 1.0
        sim.diffuse_smoke()
        assert sim.smoke[7, 8, 8] > 0
        assert sim.smoke[9, 8, 8] > 0
        assert sim.smoke[8, 7, 8] > 0
        assert sim.smoke[8, 9, 8] > 0
        assert sim.smoke[8, 8, 7] > 0
        assert sim.smoke[8, 8, 9] > 0

    def test_smoke_step_integration(self, sim: CPULBM3D) -> None:
        sim.add_emitter(8, 8, 8, strength=0.1)
        sim.step()
        assert np.sum(sim.smoke) > 0
        assert np.all(sim.smoke >= 0)

    def test_initialize_clears_smoke(self, sim: CPULBM3D) -> None:
        sim.smoke[8, 8, 8] = 1.0
        sim.emitters.append((8, 8, 8, 0.1))
        sim.initialize(rho=1.0, u=0.1, v=0.0, w=0.0)
        assert np.allclose(sim.smoke, 0.0)
        assert len(sim.emitters) == 0
