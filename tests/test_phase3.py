"""Tests for Phase 3: Turbulence, Thermal, and Non-Newtonian models.

Tests cover:
- Smagorinsky SGS collision operator
- WALE collision operator
- Thermal LBM with Boussinesq buoyancy
- Non-Newtonian viscosity models (Power-Law, Carreau, Bingham)
"""

import numpy as np
import pytest

from engines.collision import (
    BGKCollision,
    SmagorinskyCollision,
    WaleCollision,
)
from engines.lbm2d import LBM2D
from engines.lbm3d import LBM3D
from engines.lbm_common import LATTICE_2D, LATTICE_3D_Q19
from engines.non_newtonian import (
    BinghamModel,
    CarreauModel,
    NonNewtonianCollision,
    PowerLawModel,
)
from engines.thermal_mixin import ThermalMixin


# =============================================================================
# Smagorinsky SGS Tests
# =============================================================================
class TestSmagorinskyCollision:
    """Tests for Smagorinsky SGS turbulence model."""

    def test_smagorinsky_produces_valid_output_2d(self) -> None:
        """Smagorinsky collision produces finite density/velocity in 2D."""
        sim = LBM2D(width=32, height=32, collision=SmagorinskyCollision(cs=0.1))
        sim.run(100)
        rho = sim.get_density()
        u = sim.get_velocity()
        assert np.all(np.isfinite(rho))
        assert np.all(np.isfinite(u))
        assert 0.9 < rho.mean() < 1.2

    def test_smagorinsky_produces_valid_output_3d(self) -> None:
        """Smagorinsky collision produces finite density/velocity in 3D."""
        sim = LBM3D(width=16, height=16, depth=16, collision=SmagorinskyCollision(cs=0.1))
        sim.run(50)
        rho = sim.get_density()
        u = sim.get_velocity()
        assert np.all(np.isfinite(rho))
        assert np.all(np.isfinite(u))

    def test_smagorinsky_different_cs_values(self) -> None:
        """Different Smagorinsky constants produce different results."""
        sim1 = LBM2D(width=32, height=32, collision=SmagorinskyCollision(cs=0.05))
        sim2 = LBM2D(width=32, height=32, collision=SmagorinskyCollision(cs=0.2))
        sim1.run(100)
        sim2.run(100)
        u1 = sim1.get_velocity()
        u2 = sim2.get_velocity()
        # Different cs should produce different velocity fields
        assert not np.allclose(u1, u2, atol=1e-6)

    def test_smagorinsky_stability_at_high_re(self) -> None:
        """Smagorinsky remains stable at higher Reynolds numbers."""
        sim = LBM2D(
            width=64, height=64, viscosity=0.005,
            collision=SmagorinskyCollision(cs=0.12),
        )
        sim.run(500)
        rho = sim.get_density()
        assert np.all(np.isfinite(rho))
        assert 0.85 < rho.mean() < 1.2


# =============================================================================
# WALE Tests
# =============================================================================
class TestWaleCollision:
    """Tests for WALE turbulence model."""

    def test_wale_produces_valid_output_2d(self) -> None:
        """WALE collision produces finite density/velocity in 2D."""
        sim = LBM2D(width=32, height=32, collision=WaleCollision(cs=0.1))
        sim.run(100)
        rho = sim.get_density()
        u = sim.get_velocity()
        assert np.all(np.isfinite(rho))
        assert np.all(np.isfinite(u))

    def test_wale_produces_valid_output_3d(self) -> None:
        """WALE collision produces finite density/velocity in 3D."""
        sim = LBM3D(width=16, height=16, depth=16, collision=WaleCollision(cs=0.1))
        sim.run(50)
        rho = sim.get_density()
        assert np.all(np.isfinite(rho))

    def test_wale_vs_smagorinsky_different(self) -> None:
        """WALE and Smagorinsky produce different results."""
        sim_s = LBM2D(width=32, height=32, collision=SmagorinskyCollision(cs=0.1))
        sim_w = LBM2D(width=32, height=32, collision=WaleCollision(cs=0.1))
        sim_s.run(100)
        sim_w.run(100)
        u_s = sim_s.get_velocity()
        u_w = sim_w.get_velocity()
        assert not np.allclose(u_s, u_w, atol=1e-6)


# =============================================================================
# Thermal LBM Tests
# =============================================================================
class TestThermalLBM:
    """Tests for thermal LBM with Boussinesq buoyancy."""

    def test_thermal_fields_initialized(self) -> None:
        """Thermal fields are properly initialized."""
        sim = LBM2D(width=32, height=32)
        sim.init_thermal(thermal_diffusivity=0.02, beta=0.001)
        assert hasattr(sim, 'f_T')
        assert hasattr(sim, 'temperature')
        assert sim.f_T.shape == (9, 32, 32)
        assert sim.temperature.shape == (32, 32)
        assert np.all(sim.temperature == 0.0)

    def test_thermal_fields_3d(self) -> None:
        """Thermal fields are properly initialized in 3D."""
        sim = LBM3D(width=16, height=16, depth=16)
        sim.init_thermal(thermal_diffusivity=0.02, beta=0.001)
        assert sim.f_T.shape == (19, 16, 16, 16)
        assert sim.temperature.shape == (16, 16, 16)

    def test_temperature_boundary_condition(self) -> None:
        """Temperature boundary conditions are applied."""
        sim = LBM2D(width=32, height=32)
        sim.init_thermal()
        sim.set_temperature_boundary(1.0, region="left")
        # Left column should have temperature
        assert np.all(sim.f_T[:, :, 0] != 0)

    def test_thermal_step_runs(self) -> None:
        """Thermal simulation runs without errors."""
        sim = LBM2D(width=32, height=32)
        sim.init_thermal(thermal_diffusivity=0.02, beta=0.001)
        sim.set_temperature_boundary(1.0, region="bottom")
        sim.set_temperature_boundary(0.0, region="top")
        # Modified step to include thermal
        for _ in range(10):
            sim.streaming()
            sim.apply_boundary_conditions()
            sim.apply_buoyancy()
            sim.collision()
            sim.collision_temperature()
            sim.apply_outflow()
        rho = sim.get_density()
        T = sim.get_temperature()
        assert np.all(np.isfinite(rho))
        assert np.all(np.isfinite(T))

    def test_buoyancy_creates_convection(self) -> None:
        """Boussinesq buoyancy creates vertical motion when heated from below."""
        sim = LBM2D(width=32, height=32, viscosity=0.02)
        sim.init_thermal(thermal_diffusivity=0.02, beta=0.01, g_y=-1.0)
        sim.set_temperature_boundary(1.0, region="bottom")
        sim.set_temperature_boundary(0.0, region="top")
        sim.initialize(rho=1.0, u=0.0, v=0.0)

        # Run with thermal coupling
        for _ in range(200):
            sim.streaming()
            sim.apply_boundary_conditions()
            sim.apply_buoyancy()
            sim.collision()
            sim.collision_temperature()
            sim.apply_outflow()

        # Should have some vertical velocity due to buoyancy
        u = sim.get_velocity()
        v_vertical = u[:, :, 1]
        assert np.max(np.abs(v_vertical)) > 1e-6


# =============================================================================
# Non-Newtonian Model Tests
# =============================================================================
class TestPowerLawModel:
    """Tests for Power-Law viscosity model."""

    def test_power_law_creates_non_uniform_viscosity(self) -> None:
        """Power-law model produces spatially varying viscosity."""
        model = PowerLawModel(n=0.5)
        shear = np.array([0.1, 1.0, 10.0])
        viscosity = model.compute_viscosity(shear, base_viscosity=0.02)
        # Viscosity should decrease with shear for n < 1
        assert viscosity[0] > viscosity[1] > viscosity[2]

    def test_power_law_newtonian_limit(self) -> None:
        """Power-law with n=1 reduces to Newtonian."""
        model = PowerLawModel(n=1.0)
        shear = np.array([0.1, 1.0, 10.0])
        viscosity = model.compute_viscosity(shear, base_viscosity=0.02)
        assert np.allclose(viscosity, 0.02, rtol=1e-6)

    def test_shear_thickening(self) -> None:
        """Power-law with n > 1 is shear-thickening."""
        model = PowerLawModel(n=1.5)
        shear = np.array([0.1, 1.0, 10.0])
        viscosity = model.compute_viscosity(shear, base_viscosity=0.02)
        assert viscosity[0] < viscosity[1] < viscosity[2]


class TestCarreauModel:
    """Tests for Carreau viscosity model."""

    def test_carreau_newtonian_plateaus(self) -> None:
        """Carreau model has Newtonian plateaus at low and high shear."""
        model = CarreauModel(n=0.5, lambda_val=1.0, nu_inf_ratio=0.01)
        # Low shear: should be near base viscosity
        low_shear = np.array([1e-6])
        vis_low = model.compute_viscosity(low_shear, base_viscosity=0.02)
        assert vis_low[0] == pytest.approx(0.02, rel=0.01)

        # High shear: should approach nu_inf
        high_shear = np.array([10000.0])
        vis_high = model.compute_viscosity(high_shear, base_viscosity=0.02)
        assert vis_high[0] < 0.005  # Should be much less than base viscosity


class TestBinghamModel:
    """Tests for Bingham plastic model."""

    def test_bingham_yield_stress(self) -> None:
        """Bingham model has yield stress behavior."""
        model = BinghamModel(tau_y=0.1, mu_p_ratio=0.1)
        shear = np.array([0.01, 0.1, 1.0])
        viscosity = model.compute_viscosity(shear, base_viscosity=0.02)
        # At low shear, viscosity should be dominated by yield stress
        assert viscosity[0] > viscosity[1] > viscosity[2]


class TestNonNewtonianCollision:
    """Tests for NonNewtonianCollision wrapper."""

    def test_non_newtonian_collision_runs(self) -> None:
        """NonNewtonianCollision with Power-Law runs without errors."""
        model = PowerLawModel(n=0.5)
        collision = NonNewtonianCollision(BGKCollision(), model, base_viscosity=0.02)
        sim = LBM2D(width=32, height=32, collision=collision)
        sim.run(100)
        rho = sim.get_density()
        assert np.all(np.isfinite(rho))
        assert 0.9 < rho.mean() < 1.2

    def test_non_newtonian_3d_runs(self) -> None:
        """NonNewtonianCollision works in 3D."""
        model = PowerLawModel(n=0.5)
        collision = NonNewtonianCollision(BGKCollision(), model, base_viscosity=0.02)
        sim = LBM3D(width=16, height=16, depth=16, collision=collision)
        sim.run(20)
        rho = sim.get_density()
        assert np.all(np.isfinite(rho))
