"""Derived fields and the numbers reported after a solve.

The LBM is weakly compressible, so pressure comes straight from density:
``p = cs^2 * rho``.  That means a surface pressure coefficient needs no extra
Poisson solve.  Sampling is nearest-fluid-cell, which is honest about the
resolution: the heatmap is a cell-scale field, not sub-cell wall data.

Nothing here imports Qt, Warp, or PyVista, so every conversion can be tested
with plain arrays.
"""

from __future__ import annotations

from itertools import product

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


def sample_nearest_fluid(
    points: np.ndarray,
    values: np.ndarray,
    solid: np.ndarray,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Sample ``values`` at the nearest fluid cell to each point.

    Surface vertices sit on the solid boundary, so rounding them into the grid
    often lands on a solid cell.  Searching the 3 x 3 x 3 neighbourhood in
    order of distance finds the closest fluid sample instead of inventing a
    value.  Points that are completely buried stay zero.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values)
    solid = np.asarray(solid, dtype=bool)
    shape = np.asarray(solid.shape)

    index = np.rint((points - np.asarray(origin)) / np.asarray(spacing)).astype(int)
    index = np.clip(index, 0, shape - 1)
    tail = values.shape[3:]
    out = np.zeros((len(points),) + tail, dtype=np.float32)
    found = np.zeros(len(points), dtype=bool)

    offsets = sorted(product((-1, 0, 1), repeat=3), key=lambda o: sum(v * v for v in o))
    for offset in offsets:
        if found.all():
            break
        cell = index + np.asarray(offset)
        inside = np.all((cell >= 0) & (cell < shape), axis=1)
        candidates = np.flatnonzero(~found & inside)
        if candidates.size == 0:
            continue
        cells = cell[candidates]
        fluid = ~solid[cells[:, 0], cells[:, 1], cells[:, 2]]
        if not fluid.any():
            continue
        chosen = candidates[fluid]
        cells = cells[fluid]
        out[chosen] = values[cells[:, 0], cells[:, 1], cells[:, 2]]
        found[chosen] = True
    return out


def surface_pressure(
    points: np.ndarray,
    mask,
    density: np.ndarray,
    units: LatticeUnits,
) -> np.ndarray:
    """``Cp`` at every surface vertex, sampled from the neighbouring fluid."""
    cp = pressure_coefficient(density, units.u_lattice)
    return sample_nearest_fluid(points, cp, mask.solid, mask.origin, mask.spacing)
