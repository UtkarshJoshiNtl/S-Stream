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


@njit(parallel=True)
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
    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        _bounce_back_nb(f, obstacles, lattice.opp, f.shape[1], f.shape[2])


class EquilibriumInflow(BoundaryCondition):
    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        u_in = kwargs.get("u_inflow", 0.15)
        u2 = u_in * u_in
        cu = lattice.cx * u_in
        feq = lattice.w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
        f[:, :, 0] = feq[:, np.newaxis]


class OpenOutflow(BoundaryCondition):
    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        f[:, :, -1] = f[:, :, -2]


class MovingWall(BoundaryCondition):
    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        pass


class SymmetryBC(BoundaryCondition):
    def apply(
        self,
        f: np.ndarray,
        obstacles: np.ndarray,
        lattice: Lattice2D,
        **kwargs,
    ) -> None:
        pass
