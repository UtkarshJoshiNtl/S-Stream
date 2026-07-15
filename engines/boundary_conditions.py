"""D2Q9 boundary conditions for S-Stream.

Index convention (engines/lbm_common.LATTICE_2D)::

    i:   0   1   2   3   4   5   6   7   8
    cx:  0   1   0  -1   0   1  -1  -1   1
    cy:  0   0   1   0  -1   1   1  -1  -1
    opp: 0   3   4   1   2   7   8   5   6

    0 rest, 1 E, 2 N, 3 W, 4 S, 5 NE, 6 NW, 7 SW, 8 SE

Moving-wall momentum exchange assumes cs2 = 1/3
(so 2 * w_i * rho * (c·u_wall) / cs2 = 6 * w_i * rho * (c·u_wall)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange

from engines.lbm_common import Lattice2D


class BoundaryCondition(ABC):
    @abstractmethod
    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Bounce-back (obstacles)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _bounce_back_nb(
    f: np.ndarray,
    mask: np.ndarray,
    opp: np.ndarray,
    height: int,
    width: int,
) -> None:
    for i in range(9):
        opp_i = opp[i]
        if i < opp_i:
            for y in prange(height):
                for x in range(width):
                    if mask[y, x]:
                        tmp = f[i, y, x]
                        f[i, y, x] = f[opp_i, y, x]
                        f[opp_i, y, x] = tmp


class BounceBack(BoundaryCondition):
    """Full-way bounce-back on obstacle mask cells."""

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        _bounce_back_nb(f, obstacles, lattice.opp, f.shape[1], f.shape[2])


# ---------------------------------------------------------------------------
# Simple equilibrium inflow / open outflow (legacy)
# ---------------------------------------------------------------------------


class EquilibriumInflow(BoundaryCondition):
    """Set left column (x=0) to equilibrium at prescribed ux (uy=0)."""

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        u_in = float(kwargs.get("u_inflow", 0.15))
        u2 = u_in * u_in
        cu = lattice.cx * u_in
        feq = lattice.w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
        f[:, :, 0] = feq[:, np.newaxis].astype(f.dtype, copy=False)


class OpenOutflow(BoundaryCondition):
    """Zero-gradient (copy) outlet on the right column."""

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        f[:, :, -1] = f[:, :, -2]


# ---------------------------------------------------------------------------
# Zou–He velocity / pressure (Zou & He, 1997)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, boundscheck=False)
def _zou_he_left_nb(f: np.ndarray, ux: float, uy: float, height: int) -> None:
    """Zou–He velocity BC on left inlet x=0.

    After streaming, unknowns pointing into the domain: f1 (E), f5 (NE),
    f8 (SE).  Density is recovered from known populations + prescribed ux.
    """
    for y in range(height):
        f0 = f[0, y, 0]
        f2 = f[2, y, 0]
        f3 = f[3, y, 0]
        f4 = f[4, y, 0]
        f6 = f[6, y, 0]
        f7 = f[7, y, 0]
        denom = 1.0 - ux
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0.0 else -1e-12
        rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / denom
        f[1, y, 0] = f3 + (2.0 / 3.0) * rho * ux
        f[5, y, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux + 0.5 * rho * uy
        f[8, y, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux - 0.5 * rho * uy


@njit(cache=True, fastmath=True, boundscheck=False)
def _zou_he_right_pressure_nb(
    f: np.ndarray,
    rho_out: float,
    height: int,
) -> None:
    """Zou–He pressure BC on right outlet x = width-1.

    Prescribes density ``rho_out`` (lattice pressure p = rho / 3).
    Unknowns after streaming: f3 (W), f6 (NW), f7 (SW).
    Transverse velocity is taken as uy = 0 at the outlet.
    """
    x = f.shape[2] - 1
    rho = rho_out if rho_out > 0.0 else 1.0
    uy = 0.0
    for y in range(height):
        f0 = f[0, y, x]
        f1 = f[1, y, x]
        f2 = f[2, y, x]
        f4 = f[4, y, x]
        f5 = f[5, y, x]
        f8 = f[8, y, x]
        ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho
        f[3, y, x] = f1 - (2.0 / 3.0) * rho * ux
        f[7, y, x] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * ux - 0.5 * rho * uy
        f[6, y, x] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * ux + 0.5 * rho * uy


def apply_zou_he_left(
    f: np.ndarray,
    ux: float,
    uy: float,
    lattice: Lattice2D,
) -> None:
    """Apply Zou–He velocity inlet on the left column (x=0)."""
    del lattice  # D2Q9 indices are hard-coded to LATTICE_2D ordering
    _zou_he_left_nb(f, float(ux), float(uy), f.shape[1])


def apply_zou_he_right_pressure(
    f: np.ndarray,
    rho_out: float,
    lattice: Lattice2D,
) -> None:
    """Apply Zou–He pressure outlet on the right column.

    Pass density directly, or convert pressure via ``rho = 3 * p``
    (cs2 = 1/3).
    """
    del lattice
    _zou_he_right_pressure_nb(f, float(rho_out), f.shape[1])


class ZouHeVelocity(BoundaryCondition):
    """Left inlet (x=0) prescribing velocity (ux, uy).

    kwargs:
        ux / u_inflow: streamwise velocity (default 0.05)
        uy: transverse velocity (default 0.0)
    """

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        ux = float(kwargs.get("ux", kwargs.get("u_inflow", 0.05)))
        uy = float(kwargs.get("uy", 0.0))
        apply_zou_he_left(f, ux, uy, lattice)


class ZouHePressure(BoundaryCondition):
    """Right outlet prescribing density (or pressure via rho=3p).

    kwargs:
        rho_out: outlet density (default 1.0)
        p_out: optional pressure; if given, rho_out = 3 * p_out
    """

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        if "p_out" in kwargs:
            rho_out = 3.0 * float(kwargs["p_out"])
        else:
            rho_out = float(kwargs.get("rho_out", 1.0))
        apply_zou_he_right_pressure(f, rho_out, lattice)


# ---------------------------------------------------------------------------
# Moving wall (halfway bounce-back with wall momentum)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, boundscheck=False)
def _moving_wall_top_nb(
    f: np.ndarray,
    u_wall: float,
    v_wall: float,
    w: np.ndarray,
    rho_row: np.ndarray,
    width: int,
    row: int,
) -> None:
    """Moving wall on a horizontal row (unknowns: f4, f7, f8).

    f_i = f_opp + 2 * w_i * rho * (c_i · u_wall) / cs2
        = f_opp + 6 * w_i * rho * (c_i · u_wall)   with cs2 = 1/3
    """
    for x in range(width):
        rho = rho_row[x]
        if rho <= 0.0:
            rho = 1.0
        # i=4 S (0,-1), opp=2; c·u = -v_wall
        f[4, row, x] = f[2, row, x] + 6.0 * w[4] * rho * (-v_wall)
        # i=7 SW (-1,-1), opp=5; c·u = -u_wall - v_wall
        f[7, row, x] = f[5, row, x] + 6.0 * w[7] * rho * (-u_wall - v_wall)
        # i=8 SE (1,-1), opp=6; c·u = u_wall - v_wall
        f[8, row, x] = f[6, row, x] + 6.0 * w[8] * rho * (u_wall - v_wall)


@njit(cache=True, fastmath=True, boundscheck=False)
def _moving_wall_bottom_nb(
    f: np.ndarray,
    u_wall: float,
    v_wall: float,
    w: np.ndarray,
    rho_row: np.ndarray,
    width: int,
    row: int,
) -> None:
    """Moving wall on a horizontal row (unknowns: f2, f5, f6)."""
    for x in range(width):
        rho = rho_row[x]
        if rho <= 0.0:
            rho = 1.0
        # i=2 N (0,1), opp=4; c·u = v_wall
        f[2, row, x] = f[4, row, x] + 6.0 * w[2] * rho * v_wall
        # i=5 NE (1,1), opp=7; c·u = u_wall + v_wall
        f[5, row, x] = f[7, row, x] + 6.0 * w[5] * rho * (u_wall + v_wall)
        # i=6 NW (-1,1), opp=8; c·u = -u_wall + v_wall
        f[6, row, x] = f[8, row, x] + 6.0 * w[6] * rho * (-u_wall + v_wall)


def _row_density(f: np.ndarray, row: int, rho_row: np.ndarray | None) -> np.ndarray:
    """Density along a wall row (sum of populations, or provided array)."""
    if rho_row is not None:
        return np.asarray(rho_row, dtype=np.float32)
    return np.sum(f[:, row, :], axis=0).astype(np.float32)


def apply_moving_wall_top(
    f: np.ndarray,
    u_wall: float,
    lattice: Lattice2D,
    rho_row: np.ndarray | None = None,
    *,
    v_wall: float = 0.0,
    row: int | None = None,
) -> None:
    """Moving lid on the top row (default y = height-1)."""
    if row is None:
        row = f.shape[1] - 1
    rho = _row_density(f, row, rho_row)
    _moving_wall_top_nb(
        f, float(u_wall), float(v_wall), lattice.w, rho, f.shape[2], int(row)
    )


def apply_moving_wall_bottom(
    f: np.ndarray,
    u_wall: float,
    lattice: Lattice2D,
    rho_row: np.ndarray | None = None,
    *,
    v_wall: float = 0.0,
    row: int | None = None,
) -> None:
    """Moving (or stationary) wall on the bottom row (default y = 0)."""
    if row is None:
        row = 0
    rho = _row_density(f, row, rho_row)
    _moving_wall_bottom_nb(
        f, float(u_wall), float(v_wall), lattice.w, rho, f.shape[2], int(row)
    )


class MovingWall(BoundaryCondition):
    """Horizontal moving wall with momentum-exchange bounce-back.

    kwargs:
        row: wall row index (optional if ``side`` is set)
        side: ``\"top\"`` or ``\"bottom\"`` (default ``\"top\"``)
        u_wall: streamwise wall velocity (default 0.0)
        v_wall: wall-normal velocity (default 0.0)
        rho_row: optional density along the wall (1-d array)
    """

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        u_wall = float(kwargs.get("u_wall", 0.0))
        v_wall = float(kwargs.get("v_wall", 0.0))
        rho_row = kwargs.get("rho_row")
        side = kwargs.get("side")
        row = kwargs.get("row")
        height = f.shape[1]

        if side is None and row is not None:
            side = "bottom" if int(row) <= 0 else "top"
        if side is None:
            side = "top"

        if side == "bottom":
            apply_moving_wall_bottom(
                f,
                u_wall,
                lattice,
                rho_row,
                v_wall=v_wall,
                row=0 if row is None else int(row),
            )
        else:
            apply_moving_wall_top(
                f,
                u_wall,
                lattice,
                rho_row,
                v_wall=v_wall,
                row=(height - 1) if row is None else int(row),
            )


# ---------------------------------------------------------------------------
# Periodic BC (streaming already wraps)
# ---------------------------------------------------------------------------


class PeriodicBC(BoundaryCondition):
    """Marker: domain is fully periodic.

    Streaming in LBM2D already uses wrapped neighbour indices, so this
    class is a no-op.  Engines that honour BC plugins should skip wall /
    inflow / outflow application when PeriodicBC is present.
    """

    skip_walls: bool = True
    skip_inflow: bool = True
    skip_outflow: bool = True

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        return


# ---------------------------------------------------------------------------
# Symmetry (specular reflection)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, boundscheck=False)
def _symmetry_row_nb(f: np.ndarray, row: int, width: int) -> None:
    """Specular reflection on a horizontal row: swap normal (y) components.

    N↔S (2↔4), NE↔SE (5↔8), NW↔SW (6↔7).
    """
    for x in range(width):
        tmp = f[2, row, x]
        f[2, row, x] = f[4, row, x]
        f[4, row, x] = tmp
        tmp = f[5, row, x]
        f[5, row, x] = f[8, row, x]
        f[8, row, x] = tmp
        tmp = f[6, row, x]
        f[6, row, x] = f[7, row, x]
        f[7, row, x] = tmp


@njit(cache=True, fastmath=True, boundscheck=False)
def _symmetry_col_nb(f: np.ndarray, col: int, height: int) -> None:
    """Specular reflection on a vertical column: swap normal (x) components.

    E↔W (1↔3), NE↔NW (5↔6), SE↔SW (8↔7).
    """
    for y in range(height):
        tmp = f[1, y, col]
        f[1, y, col] = f[3, y, col]
        f[3, y, col] = tmp
        tmp = f[5, y, col]
        f[5, y, col] = f[6, y, col]
        f[6, y, col] = tmp
        tmp = f[8, y, col]
        f[8, y, col] = f[7, y, col]
        f[7, y, col] = tmp


class SymmetryBC(BoundaryCondition):
    """Specular reflection on a row or column.

    kwargs:
        row: horizontal symmetry line (swaps cy components)
        col: vertical symmetry line (swaps cx components)
    Exactly one of ``row`` / ``col`` should be provided.
    """

    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        del lattice
        if "row" in kwargs and kwargs["row"] is not None:
            _symmetry_row_nb(f, int(kwargs["row"]), f.shape[2])
        if "col" in kwargs and kwargs["col"] is not None:
            _symmetry_col_nb(f, int(kwargs["col"]), f.shape[1])
