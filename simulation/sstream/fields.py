"""Derived fields and the numbers reported after a solve.

The LBM is weakly compressible, so pressure comes straight from density:
``p = cs^2 * rho``.

Nothing here imports Qt, Warp, or PyVista, so every conversion can be tested
with plain arrays.
"""

from __future__ import annotations

import numpy as np

from .lattice import CS2, LatticeUnits
from .schema import FluidSpec


def speed(velocity: np.ndarray) -> np.ndarray:
    """Velocity magnitude, ``(..., 3) -> (...)``."""
    return np.sqrt(np.sum(np.asarray(velocity) ** 2, axis=-1))


def vorticity(velocity: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Curl of the velocity field as ``(nx, ny, nz, 3)``.

    ``spacing`` is needed because a real tunnel grid has metres per cell; the
    solver itself is dimensionless.  Central differences on the cell centres
    are enough for a visual, and the same field drives the volume colours.
    """
    velocity = np.asarray(velocity, dtype=np.float32)
    vx, vy, vz = (velocity[..., i] for i in range(3))
    dvx = np.gradient(vx, *spacing, axis=(0, 1, 2))
    dvy = np.gradient(vy, *spacing, axis=(0, 1, 2))
    dvz = np.gradient(vz, *spacing, axis=(0, 1, 2))
    out = np.empty_like(velocity)
    out[..., 0] = dvz[1] - dvy[2]
    out[..., 1] = dvx[2] - dvz[0]
    out[..., 2] = dvy[0] - dvx[1]
    return out


def frontal_area(solid: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Area of the solid silhouette seen by the wind, in square metres.

    Every (y, z) column that contains any solid contributes one cell face.
    Using the silhouette instead of the wetted area is what makes ``Cd`` a
    conventional drag coefficient rather than a surface-area number.
    """
    silhouette = np.asarray(solid, dtype=bool).any(axis=0)
    return float(silhouette.sum()) * float(spacing[1]) * float(spacing[2])


def force_coefficients(
    force_lattice: np.ndarray,
    units: LatticeUnits,
    fluid: FluidSpec,
    area: float,
) -> tuple[float, float]:
    """Convert lattice momentum per step to ``(Cd, Cl)``.

    ``force_lattice`` is the sum handed to the solid, in lattice units where
    density is one and one length/time is one cell/step.  The physical force
    is ``F_lat * rho * dx^4 / dt^2``; the rest is the usual dynamic pressure
    and reference area.  ``Cl`` is the vertical (world Z) component because
    that is the lift a car or a wing actually cares about.
    """
    if area <= 0.0:
        return (0.0, 0.0)
    force = np.asarray(force_lattice, dtype=np.float64).reshape(3)
    scale = fluid.density * units.dx**4 / units.dt**2
    dynamic = fluid.dynamic_pressure * area
    return (float(force[0] * scale / dynamic), float(force[2] * scale / dynamic))


def pressure_coefficient(density: np.ndarray, u_lattice: float) -> np.ndarray:
    """``Cp`` from the LBM equation of state, with the free stream at rest."""
    if u_lattice <= 0.0:
        return np.zeros_like(density, dtype=np.float32)
    return (CS2 * (np.asarray(density, dtype=np.float32) - 1.0)) / (0.5 * u_lattice * u_lattice)


_Q_SLAB_CELLS = 4_000_000  # max cells per x-slab for the S/Omega assembly


def q_criterion(velocity: np.ndarray, dx: float | tuple[float, float, float] = 1.0) -> np.ndarray:
    """Second invariant of the velocity gradient, ``Q``.

    ``Q = 0.5 (|Omega|^2 - |S|^2)`` with ``S`` and ``Omega`` the symmetric
    and antisymmetric parts of the velocity gradient ``A_ij = du_i/dx_j``
    (Frobenius norms).  ``velocity`` is ``(nx, ny, nz, 3)`` and ``dx`` is a
    cell size, either one scalar for isotropic cells or a ``(dx, dy, dz)``
    spacing.  Positive ``Q`` marks rotation-dominated regions (vortices);
    a pure shear has ``S`` and ``Omega`` equally strong and gives ``Q = 0``.

    The six gradient arrays are built once with central differences (edge
    cells fall back to one-sided differences, like ``numpy.gradient``), and
    the symmetric/antisymmetric temporaries are summed in x-slabs so they
    never exceed one slab instead of the whole domain.  The result is
    float32.
    """
    u = np.ascontiguousarray(velocity, dtype=np.float32)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("velocity must be (nx, ny, nz, 3)")
    spacing = (
        (float(dx),) * 3 if np.isscalar(dx) else tuple(float(v) for v in dx)
    )
    if len(spacing) != 3 or min(spacing) <= 0.0:
        raise ValueError("dx must be a positive scalar or a positive (dx, dy, dz)")
    vx, vy, vz = (u[..., i] for i in range(3))
    dux = np.gradient(vx, *spacing, axis=(0, 1, 2))
    duy = np.gradient(vy, *spacing, axis=(0, 1, 2))
    duz = np.gradient(vz, *spacing, axis=(0, 1, 2))

    nx, ny, nz = u.shape[:3]
    out = np.empty((nx, ny, nz), dtype=np.float32)
    slab = max(1, _Q_SLAB_CELLS // (ny * nz))
    for x0 in range(0, nx, slab):
        x1 = min(nx, x0 + slab)
        sx, sy, sz = (slice(x0, x1),) * 3

        def g(component, axis):
            return np.asarray(
                (dux if component == 0 else duy if component == 1 else duz)[axis][sx, sy, sz],
                dtype=np.float32,
            )

        dux_x, dux_y, dux_z = g(0, 0), g(0, 1), g(0, 2)
        duy_x, duy_y, duy_z = g(1, 0), g(1, 1), g(1, 2)
        duz_x, duz_y, duz_z = g(2, 0), g(2, 1), g(2, 2)

        s_xx, s_yy, s_zz = dux_x, duy_y, duz_z
        strace_xy = 0.5 * (dux_y + duy_x)
        strace_xz = 0.5 * (dux_z + duz_x)
        strace_yz = 0.5 * (duy_z + duz_y)
        o_xy = 0.5 * (dux_y - duy_x)
        o_xz = 0.5 * (dux_z - duz_x)
        o_yz = 0.5 * (duy_z - duz_y)

        omega2 = 2.0 * (o_xy * o_xy + o_xz * o_xz + o_yz * o_yz)
        strain2 = (
            s_xx * s_xx
            + s_yy * s_yy
            + s_zz * s_zz
            + 2.0 * (strace_xy * strace_xy + strace_xz * strace_xz + strace_yz * strace_yz)
        )
        out[sx, sy, sz] = 0.5 * (omega2 - strain2)
    return out
