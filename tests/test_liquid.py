from __future__ import annotations

import numpy as np
import pytest
from engines.lbm2d_liquid import LBM2DLiquid


@pytest.fixture
def sim() -> LBM2DLiquid:
    return LBM2DLiquid(width=32, height=32, viscosity=0.02, g=-5.0, g_adhesion=-5.0)


@pytest.fixture
def liq() -> LBM2DLiquid:
    """Larger grid with Shan-Chen-friendly omega (~1.0)."""
    return LBM2DLiquid(
        width=64, height=64, viscosity=1.0 / 6.0, g=-8.0, g_adhesion=-5.0
    )


class TestInit:
    def test_default_grid(self) -> None:
        s = LBM2DLiquid()
        assert s.width == 128
        assert s.height == 128

    def test_custom_grid(self, sim: LBM2DLiquid) -> None:
        assert sim.width == 32
        assert sim.height == 32

    def test_initial_fields_not_nan(self, sim: LBM2DLiquid) -> None:
        assert not np.any(np.isnan(sim.f))
        assert not np.any(np.isnan(sim.rho))

    def test_droplet_initialized(self, sim: LBM2DLiquid) -> None:
        assert np.any(sim.rho > 1.5), "droplet not initialized"
        assert np.any(sim.rho < 0.1), "vapor background not initialized"

    def test_no_obstacles_by_default(self, sim: LBM2DLiquid) -> None:
        assert not np.any(sim.obstacles)

    def test_no_smoke_by_default(self, sim: LBM2DLiquid) -> None:
        assert np.all(sim.smoke == 0.0)


class TestForce:
    def test_force_computed_on_step(self, sim: LBM2DLiquid) -> None:
        sim.step()
        assert not np.any(np.isnan(sim.fx))
        assert not np.any(np.isnan(sim.fy))
        assert np.any(sim.fx != 0.0) or np.any(sim.fy != 0.0)

    def test_force_zero_inside_obstacles(self, sim: LBM2DLiquid) -> None:
        sim.add_obstacle(sim.width // 2, sim.height // 2, 5)
        sim.step()
        mask = sim.obstacles
        assert np.all(sim.fx[mask] == 0.0)
        assert np.all(sim.fy[mask] == 0.0)


class TestStep:
    def test_step_updates_rho(self, sim: LBM2DLiquid) -> None:
        rho_before = sim.rho.copy()
        sim.step()
        assert not np.allclose(rho_before, sim.rho, atol=1e-6)

    def test_step_updates_velocity(self, sim: LBM2DLiquid) -> None:
        u_before = sim.u.copy()
        v_before = sim.v.copy()
        sim.step()
        assert not np.allclose(u_before, sim.u, atol=1e-6)
        assert not np.allclose(v_before, sim.v, atol=1e-6)

    def test_step_no_nan(self, liq: LBM2DLiquid) -> None:
        for _ in range(100):
            liq.step()
        assert not np.any(np.isnan(liq.rho))
        assert not np.any(np.isnan(liq.u))
        assert not np.any(np.isnan(liq.v))
        assert np.all(liq.rho > 0)

    def test_mass_conserved_without_obstacles(self, liq: LBM2DLiquid) -> None:
        mass_before = np.sum(liq.rho)
        for _ in range(100):
            liq.step()
        mass_after = np.sum(liq.rho)
        assert abs(mass_after - mass_before) / mass_before < 0.10

    def test_liquid_phase_separates(self, liq: LBM2DLiquid) -> None:
        for _ in range(200):
            liq.step()
        assert not np.any(np.isnan(liq.rho))
        assert np.any(liq.rho > 1.5), f"max rho={liq.rho.max():.4f}"
        assert np.any(liq.rho < 0.5), "vapor phase not present"


class TestObstacles:
    def test_add_obstacle_circle(self, sim: LBM2DLiquid) -> None:
        sim.add_obstacle(10, 10, 3)
        assert sim.obstacles[10, 10]
        assert not sim.obstacles[0, 0]

    def test_clear_obstacles(self, sim: LBM2DLiquid) -> None:
        sim.add_obstacle(10, 10, 3)
        assert np.any(sim.obstacles)
        sim.clear_obstacles()
        assert not np.any(sim.obstacles)

    def test_density_zero_inside_obstalces(self, sim: LBM2DLiquid) -> None:
        sim.add_obstacle(sim.width // 2, sim.height // 2, 8)
        for _ in range(20):
            sim.step()
        mask = sim.obstacles
        assert np.all(sim.rho[mask] > 0)

    def test_obstacle_smoke_cleared(self, sim: LBM2DLiquid) -> None:
        sim.add_obstacle(10, 10, 5)
        sim.smoke[10, 10] = 1.0
        sim.step()
        assert sim.smoke[10, 10] == 0.0


class TestEmitters:
    def test_add_emitter(self, sim: LBM2DLiquid) -> None:
        sim.add_emitter(5, 5, 0.1)
        assert sim.get_emitter_count() == 1

    def test_clear_emitters(self, sim: LBM2DLiquid) -> None:
        sim.add_emitter(5, 5, 0.1)
        sim.clear_emitters()
        assert sim.get_emitter_count() == 0

    def test_emitter_increases_smoke(self, sim: LBM2DLiquid) -> None:
        sim.add_emitter(5, 5, 0.1)
        sm_before = sim.smoke[5, 5]
        sim.step()
        assert sim.smoke[5, 5] > sm_before

    def test_emitter_out_of_bounds(self, sim: LBM2DLiquid) -> None:
        sim.add_emitter(-1, -1, 0.1)
        sim.step()
        assert np.all(sim.smoke == 0.0)

    def test_multiple_emitters(self, sim: LBM2DLiquid) -> None:
        sim.add_emitter(5, 5, 0.1)
        sim.add_emitter(10, 10, 0.1)
        assert sim.get_emitter_count() == 2


class TestGetDensity:
    def test_get_density_shape(self, sim: LBM2DLiquid) -> None:
        d = sim.get_density()
        assert d.shape == (32, 32)
        assert d.dtype == np.float32

    def test_get_density_values_match(self, sim: LBM2DLiquid) -> None:
        d = sim.get_density()
        assert np.allclose(d, sim.rho)

    def test_get_density_is_copy(self, sim: LBM2DLiquid) -> None:
        d = sim.get_density()
        d[0, 0] = 999
        assert sim.rho[0, 0] != 999


class TestGetVelocity:
    def test_get_velocity_shape(self, sim: LBM2DLiquid) -> None:
        vel = sim.get_velocity()
        assert vel.shape == (32, 32, 2)
        assert vel.dtype == np.float32

    def test_get_velocity_values(self, sim: LBM2DLiquid) -> None:
        vel = sim.get_velocity()
        assert np.allclose(vel[:, :, 0], sim.u)
        assert np.allclose(vel[:, :, 1], sim.v)

    def test_get_velocity_is_copy(self, sim: LBM2DLiquid) -> None:
        vel = sim.get_velocity()
        vel[0, 0, 0] = 999
        assert sim.u[0, 0] != 999
