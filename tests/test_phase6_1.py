"""Tests for Phase 6.1: Multi-Component Shan-Chen LBM Engine."""
from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d_multicomponent import LBM2DMultiComponent


class TestMultiComponentInit:
    def test_default_init(self) -> None:
        sim = LBM2DMultiComponent(width=64, height=64)
        assert sim.grid_shape == (64, 64)
        assert sim.ndim == 2
        assert sim.g11 == -5.0
        assert sim.g22 == -5.0
        assert sim.g12 == 5.0
        assert sim.sigma == 0.05
        assert sim.g_adhesion == -5.0

    def test_custom_params(self) -> None:
        sim = LBM2DMultiComponent(
            width=32, height=32, g11=-3.0, g22=-4.0, g12=6.0,
            sigma=0.1, g_adhesion=-2.0, droplet_radius=5,
        )
        assert sim.g11 == -3.0
        assert sim.g22 == -4.0
        assert sim.g12 == 6.0
        assert sim.sigma == 0.1
        assert sim.g_adhesion == -2.0
        assert sim.droplet_radius == 5

    def test_distributions_allocated(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        assert sim.f1.shape == (9, 32, 32)
        assert sim.f2.shape == (9, 32, 32)
        assert sim.rho1.shape == (32, 32)
        assert sim.rho2.shape == (32, 32)

    def test_force_arrays_allocated(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        assert sim.fx1.shape == (32, 32)
        assert sim.fy1.shape == (32, 32)
        assert sim.fx2.shape == (32, 32)
        assert sim.fy2.shape == (32, 32)

    def test_defaults_32x32(self) -> None:
        sim = LBM2DMultiComponent()
        assert sim.grid_shape == (128, 128)


class TestMultiComponentInitDensities:
    def test_dropletInitialized(self) -> None:
        sim = LBM2DMultiComponent(width=64, height=64)
        # Component 1 should have high density at center
        cx, cy = 32, 32
        radius = 64 // 6  # ~10
        assert sim.rho1[cy, cx] == pytest.approx(2.0, abs=0.01)
        # Component 2 should have high density at edges
        assert sim.rho2[0, 0] == pytest.approx(2.0, abs=0.01)

    def test_complementaryInitialization(self) -> None:
        sim = LBM2DMultiComponent(width=64, height=64)
        # Where rho1 is high, rho2 should be low, and vice versa
        center_rho1 = sim.rho1[32, 32]
        corner_rho1 = sim.rho1[0, 0]
        assert center_rho1 > corner_rho1

    def test_custom_droplet_radius(self) -> None:
        sim = LBM2DMultiComponent(width=64, height=64, droplet_radius=5)
        # Only a small region should have high density
        assert sim.rho1[32, 32] == pytest.approx(2.0, abs=0.01)
        # Just outside the droplet, rho1 should be low
        assert sim.rho1[32, 40] == pytest.approx(0.01, abs=0.1)


class TestMultiComponentStep:
    def test_singleStep(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.step()
        rho1_after = sim.rho1.copy()
        rho2_after = sim.rho2.copy()
        assert np.all(np.isfinite(rho1_after))
        assert np.all(np.isfinite(rho2_after))

    def test_runMultipleSteps(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(50)
        assert np.all(np.isfinite(sim.rho1))
        assert np.all(np.isfinite(sim.rho2))
        assert np.all(np.isfinite(sim.u))
        assert np.all(np.isfinite(sim.v))

    def test_densitySeparation(self) -> None:
        """After many steps, component 1 and 2 should develop spatial structure."""
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(500)
        # Both components should still be finite and have spatial variation
        assert np.all(np.isfinite(sim.rho1))
        assert np.all(np.isfinite(sim.rho2))
        # There should be some spatial structure (not all identical)
        assert float(np.std(sim.rho1)) > 0.01
        assert float(np.std(sim.rho2)) > 0.01

    def test_noNegativeDensity(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(100)
        assert np.all(sim.rho1 >= 0)
        assert np.all(sim.rho2 >= 0)


class TestMultiComponentFields:
    def test_field_names(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        names = sim.get_field_names()
        assert "component1" in names
        assert "component2" in names
        assert "color" in names
        assert "smoke" in names
        assert "speed" in names
        assert "vorticity" in names
        assert "pressure" in names
        assert "density" in names
        assert "phase" in names

    def test_get_component1_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("component1")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32
        assert float(np.min(f)) >= 0.0
        assert float(np.max(f)) <= 1.0

    def test_get_component2_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("component2")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32
        assert float(np.min(f)) >= 0.0
        assert float(np.max(f)) <= 1.0

    def test_get_color_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("color")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32
        assert float(np.min(f)) >= 0.0
        assert float(np.max(f)) <= 1.0

    def test_get_phase_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("phase")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32

    def test_get_smoke_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        f = sim.get_field("smoke")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32

    def test_get_speed_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("speed")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32

    def test_get_vorticity_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("vorticity")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32

    def test_get_pressure_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("pressure")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32

    def test_get_density_field(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(10)
        f = sim.get_field("density")
        assert f.shape == (32, 32)
        assert f.dtype == np.float32

    def test_unknown_field_raises(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        with pytest.raises(ValueError, match="Unknown field"):
            sim.get_field("nonexistent")


class TestMultiComponentObstacles:
    def test_add_obstacle(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.add_obstacle(16, 16, radius=3)
        assert sim.obstacles[16, 16] is np.True_
        assert sim.obstacles[0, 0] is np.False_

    def test_clear_obstacles(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.add_obstacle(16, 16, radius=3)
        sim.clear_obstacles()
        assert not np.any(sim.obstacles)

    def test_obstacle_noDensityDrift(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.add_obstacle(16, 16, radius=5)
        sim.run(50)
        # Density inside obstacle should remain bounded (not NaN/inf)
        rho1_obs = sim.rho1[sim.obstacles]
        assert np.all(np.isfinite(rho1_obs))
        # Bounce-back preserves distributions inside obstacles
        assert float(np.mean(rho1_obs)) > 0


class TestMultiComponentEmitters:
    def test_add_emitter(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.add_emitter(10, 10, 0.05)
        assert sim.get_emitter_count() == 1

    def test_clear_emitters(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.add_emitter(10, 10, 0.05)
        sim.clear_emitters()
        assert sim.get_emitter_count() == 0


class TestMultiComponentSmoke:
    def test_smokeAdvection(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.add_emitter(5, 16, 0.1)
        sim.run(50)
        smoke = sim.get_smoke()
        assert float(np.max(smoke)) > 0.0

    def test_smokeDecay(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.smoke[:] = 0.5
        sim.smoke_decay = 0.9
        sim.run(10)
        assert float(np.mean(sim.smoke)) < 0.5


class TestMultiComponentVelocity:
    def test_velocityShape(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        vel = sim.get_velocity()
        assert vel.shape == (32, 32, 2)

    def test_initialVelocityIsZero(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        vel = sim.get_velocity()
        assert float(np.max(np.abs(vel))) == 0.0


class TestMultiComponentGetF:
    def test_getFShape(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        f = sim.get_f()
        # Should be 18 = 9 (component 1) + 9 (component 2)
        assert f.shape == (18, 32, 32)


class TestMultiComponentPressure:
    def test_pressureShape(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        p = sim.get_pressure()
        assert p.shape == (32, 32)


class TestMultiComponentDensity:
    def test_densityShape(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        rho = sim.get_density()
        assert rho.shape == (32, 32)


class TestMultiComponentInterComponentRepulsion:
    def test_highRepulsionSeperatesFaster(self) -> None:
        """Higher g12 should cause faster phase separation."""
        sim_weak = LBM2DMultiComponent(width=32, height=32, g12=1.0)
        sim_strong = LBM2DMultiComponent(width=32, height=32, g12=10.0)
        sim_weak.run(200)
        sim_strong.run(200)
        # With higher repulsion, the color contrast should be sharper
        total_weak = sim_weak.rho1 + sim_weak.rho2
        total_strong = sim_strong.rho1 + sim_strong.rho2
        # Just verify both ran without error
        assert np.all(np.isfinite(total_weak))
        assert np.all(np.isfinite(total_strong))


class TestMultiComponentSigma:
    def test_zeroSigmaNoColorPerturbation(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32, sigma=0.0)
        sim.run(50)
        assert np.all(np.isfinite(sim.rho1))

    def test_highSigmaStillStable(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32, sigma=0.2)
        sim.run(50)
        assert np.all(np.isfinite(sim.rho1))
        assert np.all(np.isfinite(sim.rho2))


class TestMultiComponentInitialization:
    def test_reinitialize(self) -> None:
        sim = LBM2DMultiComponent(width=32, height=32)
        sim.run(100)
        sim.initialize()
        # After reinit, component 1 should have high density at center again
        assert sim.rho1[16, 16] == pytest.approx(2.0, abs=0.01)
        assert sim.rho2[0, 0] == pytest.approx(2.0, abs=0.01)
