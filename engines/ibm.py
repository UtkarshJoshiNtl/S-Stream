"""Immersed Boundary Method (rigid) for moving/prescribed bodies — Phase D.

Lagrangian markers on the immersed surface; interpolate velocity from the
Eulerian grid, compute penalty force toward prescribed velocity, spread
force back to the Eulerian fluid (Peskin-style regularized delta).

This is a compact rigid-IBM suitable for translating/rotating cylinders.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _phi4(r: float) -> float:
    """4-point Peskin delta kernel in 1D."""
    ar = abs(r)
    if ar >= 2.0:
        return 0.0
    if ar >= 1.0:
        return (5.0 - 2.0 * ar - np.sqrt(-7.0 + 12.0 * ar - 4.0 * ar * ar)) / 8.0
    return (3.0 - 2.0 * ar + np.sqrt(1.0 + 4.0 * ar - 4.0 * ar * ar)) / 8.0


@njit(cache=True, fastmath=True)
def _delta2(dx: float, dy: float) -> float:
    return _phi4(dx) * _phi4(dy)


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _interpolate_velocity_nb(
    u: np.ndarray,
    v: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    out_u: np.ndarray,
    out_v: np.ndarray,
    height: int,
    width: int,
) -> None:
    n = px.shape[0]
    for k in prange(n):
        x = px[k]
        y = py[k]
        i0 = int(np.floor(x)) - 1
        j0 = int(np.floor(y)) - 1
        su = 0.0
        sv = 0.0
        for jj in range(4):
            j = j0 + jj
            if j < 0 or j >= height:
                continue
            for ii in range(4):
                i = i0 + ii
                if i < 0 or i >= width:
                    continue
                w = _delta2(x - i, y - j)
                su += w * u[j, i]
                sv += w * v[j, i]
        out_u[k] = su
        out_v[k] = sv


@njit(cache=True, fastmath=True, boundscheck=False)
def _spread_force_nb(
    fx: np.ndarray,
    fy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    f_lag_x: np.ndarray,
    f_lag_y: np.ndarray,
    height: int,
    width: int,
    ds: float,
) -> None:
    n = px.shape[0]
    fx[:] = 0.0
    fy[:] = 0.0
    # Serial accumulate into Eulerian field (avoid races)
    for k in range(n):
        x = px[k]
        y = py[k]
        i0 = int(np.floor(x)) - 1
        j0 = int(np.floor(y)) - 1
        qx = f_lag_x[k] * ds
        qy = f_lag_y[k] * ds
        for jj in range(4):
            j = j0 + jj
            if j < 0 or j >= height:
                continue
            for ii in range(4):
                i = i0 + ii
                if i < 0 or i >= width:
                    continue
                w = _delta2(x - i, y - j)
                fx[j, i] += w * qx
                fy[j, i] += w * qy


class RigidIBM:
    """Rigid immersed boundary driven by prescribed Lagrangian velocity."""

    def __init__(
        self,
        width: int,
        height: int,
        stiffness: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height
        self.stiffness = stiffness
        self.px = np.zeros(0, dtype=np.float32)
        self.py = np.zeros(0, dtype=np.float32)
        self.u_des = np.zeros(0, dtype=np.float32)
        self.v_des = np.zeros(0, dtype=np.float32)
        self.ds = 1.0
        self.fx = np.zeros((height, width), dtype=np.float32)
        self.fy = np.zeros((height, width), dtype=np.float32)
        self._iu = np.zeros(0, dtype=np.float32)
        self._iv = np.zeros(0, dtype=np.float32)

    def set_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        n_markers: int | None = None,
        u_body: float = 0.0,
        v_body: float = 0.0,
    ) -> None:
        if n_markers is None:
            n_markers = max(16, int(2.0 * np.pi * radius))
        theta = np.linspace(0.0, 2.0 * np.pi, n_markers, endpoint=False)
        self.px = (cx + radius * np.cos(theta)).astype(np.float32)
        self.py = (cy + radius * np.sin(theta)).astype(np.float32)
        self.u_des = np.full(n_markers, u_body, dtype=np.float32)
        self.v_des = np.full(n_markers, v_body, dtype=np.float32)
        self.ds = float(2.0 * np.pi * radius / n_markers)
        self._iu = np.zeros(n_markers, dtype=np.float32)
        self._iv = np.zeros(n_markers, dtype=np.float32)

    def compute_force(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Eulerian force density (fx, fy) from penalty IBM."""
        if self.px.size == 0:
            self.fx[:] = 0.0
            self.fy[:] = 0.0
            return self.fx, self.fy
        _interpolate_velocity_nb(
            u, v, self.px, self.py, self._iu, self._iv, self.height, self.width
        )
        f_lag_x = self.stiffness * (self.u_des - self._iu)
        f_lag_y = self.stiffness * (self.v_des - self._iv)
        _spread_force_nb(
            self.fx,
            self.fy,
            self.px,
            self.py,
            f_lag_x,
            f_lag_y,
            self.height,
            self.width,
            self.ds,
        )
        return self.fx, self.fy
