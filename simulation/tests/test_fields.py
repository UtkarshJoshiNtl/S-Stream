"""Derived-field tests: curl, reference area, Cp, and force conversion.

These are pure NumPy checks.  They pin the parts that are easy to get subtly
wrong: axis order in the curl, the choice of silhouette area, and the
``rho * dx^4 / dt^2`` force scale.
"""

import numpy as np
import pytest

from sstream import fields
from sstream.lattice import CS2, LatticeUnits
from sstream.schema import FluidSpec


def test_vorticity_of_solid_body_rotation_is_2_omega():
    nx, ny, nz = 14, 12, 10
    x, y, z = np.indices((nx, ny, nz)).astype(float)
    omega = 2.0
    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    velocity[..., 0] = -omega * (y * 0.5)
    velocity[..., 1] = omega * (x * 0.5)

    curl = fields.vorticity(velocity, spacing=(0.5, 0.5, 0.5))
    interior = curl[1:-1, 1:-1, 1:-1]
    assert np.allclose(interior[..., 0], 0.0, atol=1e-5)
    assert np.allclose(interior[..., 1], 0.0, atol=1e-5)
    assert np.allclose(interior[..., 2], 2.0 * omega, atol=1e-4)


def test_frontal_area_uses_the_silhouette_not_the_wetted_area():
    solid = np.zeros((4, 3, 2), dtype=bool)
    solid[0:2, 1, 1] = True          # two cells along x, same (y, z) column
    solid[3, 0, 0] = True            # one separate column
    area = fields.frontal_area(solid, spacing=(0.5, 0.25, 0.2))
    assert area == pytest.approx(2 * 0.25 * 0.2)


def test_pressure_coefficient_follows_the_lattice_equation_of_state():
    density = np.array([1.0, 1.01, 0.99], dtype=np.float32)
    cp = fields.pressure_coefficient(density, u_lattice=0.1)
    assert cp[0] == 0.0
    expected = (CS2 * 0.01) / (0.5 * 0.1 * 0.1)
    assert cp[1] == pytest.approx(expected, rel=1e-5)
    assert cp[2] == pytest.approx(-expected, rel=1e-5)


def test_force_conversion_returns_cd_and_vertical_cl():
    units = LatticeUnits(dx=1.0, dt=1.0, u_lattice=0.1, tau=0.6)
    fluid = FluidSpec(speed=10.0, density=1.0, kinematic_viscosity=1e-5)
    area = 2.0
    cd, cl = fields.force_coefficients(np.array([5.0, 99.0, 2.0]), units, fluid, area)
    dynamic = 0.5 * 1.0 * 10.0**2 * area
    assert cd == pytest.approx(5.0 / dynamic)
    assert cl == pytest.approx(2.0 / dynamic)


def test_force_conversion_scales_with_cell_and_time_size():
    units = LatticeUnits(dx=0.5, dt=0.25, u_lattice=0.05, tau=0.6)
    fluid = FluidSpec(speed=20.0, density=2.0, kinematic_viscosity=1e-5)
    cd, _ = fields.force_coefficients(np.array([3.0, 0.0, 0.0]), units, fluid, 1.5)
    physical = 3.0 * 2.0 * 0.5**4 / 0.25**2
    assert cd == pytest.approx(physical / (0.5 * 2.0 * 20.0**2 * 1.5))
