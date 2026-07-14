from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange

from engines.lbm_common import Lattice2D


class CollisionOperator(ABC):
    @abstractmethod
    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D,
        viscosity: float,
    ) -> None:
        ...


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _bgk_collide_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega: float,
    height: int,
    width: int,
) -> None:
    for y in prange(height):
        for x in range(width):
            r = 0.0
            u_vel = 0.0
            v_vel = 0.0
            for i in range(9):
                fiv = f[i, y, x]
                r += fiv
                u_vel += fiv * cx[i]
                v_vel += fiv * cy[i]
            rho_safe = r if r > 0 else 1.0
            u_vel /= rho_safe
            v_vel /= rho_safe
            u2 = u_vel * u_vel + v_vel * v_vel
            for i in range(9):
                cu = cx[i] * u_vel + cy[i] * v_vel
                feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f[i, y, x] = f[i, y, x] * (1.0 - omega) + feq * omega
            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel


class BGKCollision(CollisionOperator):
    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D,
        viscosity: float,
    ) -> None:
        omega = lattice.omega_from_viscosity(viscosity)
        _bgk_collide_nb(
            f,
            rho,
            u,
            v,
            lattice.w,
            lattice.cx,
            lattice.cy,
            omega,
            f.shape[1],
            f.shape[2],
        )


class TRTCollision(CollisionOperator):
    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D,
        viscosity: float,
    ) -> None:
        # Placeholder for TRT
        pass


class MRTCollision(CollisionOperator):
    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D,
        viscosity: float,
    ) -> None:
        # Placeholder for MRT
        pass
