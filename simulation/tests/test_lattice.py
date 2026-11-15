"""Constants, isotropy, and the SI-to-lattice conversion.

A flipped diagonal direction in a hand-typed D3Q19 table still produces a
smooth-looking simulation that slowly diverges.  These checks are cheap and
catch it immediately.
"""

import numpy as np
import pytest

from sstream.lattice import (
    CS2,
    DIRECTIONS,
    OPPOSITE,
    TAU_MIN,
    WEIGHTS,
    LatticeUnits,
    equilibrium,
)


def test_table_has_the_right_shells():
    assert DIRECTIONS.shape == (19, 3)
    speed2 = (DIRECTIONS.astype(np.int64) ** 2).sum(axis=1)
    assert speed2.tolist() == [0] + [1] * 6 + [2] * 12
    assert WEIGHTS.shape == (19,)
    assert OPPOSITE.shape == (19,)


def test_opposites_negate_and_involute():
    for i, j in enumerate(OPPOSITE):
        assert np.array_equal(DIRECTIONS[j], -DIRECTIONS[i])
        assert OPPOSITE[j] == i


def test_weights_and_isotropy():
    assert WEIGHTS.sum() == pytest.approx(1.0, abs=1e-6)
    c = DIRECTIONS.astype(np.float64)
    w = WEIGHTS.astype(np.float64)
    # Second moment must be isotropic with cs2 on the diagonal...
    stress = np.einsum("i,ia,ib->ab", w, c, c)
    assert np.allclose(stress, CS2 * np.eye(3), atol=1e-7)
    # ...and the fourth moment must be the lattice value, not 3 * cs2**2.
    assert (w * c[:, 0] ** 4).sum() == pytest.approx(CS2, abs=1e-6)
    assert (w * c[:, 0] ** 2 * c[:, 1] ** 2).sum() == pytest.approx(CS2**2, abs=1e-6)
    assert (w * c[:, 0] * c[:, 1]).sum() == pytest.approx(0.0, abs=1e-7)


def test_equilibrium_recovers_the_moments():
    u = np.array([0.01, -0.02, 0.03])
    rho = 1.2
    f = equilibrium(u[0], u[1], u[2], rho)
    c = DIRECTIONS.astype(np.float64)
    assert f.shape == (19,)
    assert f.sum() == pytest.approx(rho, abs=1e-6)
    assert np.allclose(c.T @ f, rho * u, atol=1e-6)
    # Stress includes the convective term; the diagonal still carries cs2.
    stress = np.einsum("i,ia,ib->ab", f, c, c)
    assert np.allclose(stress - rho * CS2 * np.eye(3), rho * np.outer(u, u), atol=1e-6)


def test_equilibrium_broadcasts_over_a_grid():
    ux = np.zeros((4, 5, 6), dtype=np.float32)
    f = equilibrium(ux, 0.0, 0.0)
    assert f.shape == (4, 5, 6, 19)
    assert f.sum(axis=-1) == pytest.approx(1.0, abs=1e-6)


def test_units_convert_both_ways():
    # A viscosity that keeps tau above TAU_MIN, so the conversion round-trips
    # exactly.  The clamped case is covered by test_units_floor_is_tau_min.
    units = LatticeUnits.from_physical(
        speed=30.0, kinematic_viscosity=1.5e-2, dx=6.0 / 256
    )
    assert units.u_lattice == pytest.approx(0.05)
    assert units.dt == pytest.approx(0.05 * (6.0 / 256) / 30.0)
    assert units.tau > 0.5
    assert units.nu_physical == pytest.approx(1.5e-2, rel=1e-6)
    assert units.nu_lattice == pytest.approx(CS2 * (units.tau - 0.5))


def test_units_floor_is_tau_min():
    # Air on a fine grid wants tau ~ 0.5000007; the shared TAU_MIN floor
    # (the same one LbmConfig validates and the step kernel clamps to)
    # raises it, and the conversion records that it did.
    units = LatticeUnits.from_physical(
        speed=30.0, kinematic_viscosity=1.5e-5, dx=6.0 / 256
    )
    assert units.tau == pytest.approx(TAU_MIN)
    assert units.viscosity_clamped


def test_units_reject_nonsense():
    with pytest.raises(ValueError):
        LatticeUnits.from_physical(0.0, 1.5e-5, 0.01)
    with pytest.raises(ValueError):
        LatticeUnits.from_physical(30.0, 0.0, 0.01)
    with pytest.raises(ValueError):
        LatticeUnits.from_physical(30.0, 1.5e-5, 0.0)
    # A lattice speed near 0.2 would put the Mach number where the
    # incompressible assumption stops holding.
    with pytest.raises(ValueError):
        LatticeUnits.from_physical(30.0, 1.5e-5, 0.01, u_lattice=0.2)
