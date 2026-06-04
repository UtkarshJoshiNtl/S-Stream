from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lattice2D:
    """D2Q9 lattice constants."""

    w: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    opp: np.ndarray

    def equilibrium(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        feq = np.zeros((9, *rho.shape))
        u2 = u**2 + v**2
        for i in range(9):
            cu = self.cx[i] * u + self.cy[i] * v
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    @staticmethod
    def omega_from_viscosity(nu: float) -> float:
        return 1.0 / (3.0 * nu + 0.5)

    def assert_stable(self, nu: float, omega: float) -> None:
        assert 0 < omega < 2, (
            f"omega={omega:.3f} outside stable range (0, 2) " f"for viscosity={nu}"
        )


@dataclass(frozen=True)
class Lattice3D:
    """D3Q19 lattice constants."""

    w: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    opp: np.ndarray

    def equilibrium(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w_vel: np.ndarray,
    ) -> np.ndarray:
        feq = np.zeros((19, *rho.shape))
        u2 = u**2 + v**2 + w_vel**2
        for i in range(19):
            cu = self.cx[i] * u + self.cy[i] * v + self.cz[i] * w_vel
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    @staticmethod
    def omega_from_viscosity(nu: float) -> float:
        return 1.0 / (3.0 * nu + 0.5)

    @staticmethod
    def assert_stable(nu: float, omega: float) -> None:
        assert 0 < omega < 2, (
            f"omega={omega:.3f} outside stable range (0, 2) " f"for viscosity={nu}"
        )


LATTICE_2D = Lattice2D(
    w=np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]),
    cx=np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]),
    cy=np.array([0, 0, 1, 0, -1, 1, 1, -1, -1]),
    opp=np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]),
)

LATTICE_3D = Lattice3D(
    w=np.array(
        [
            1 / 3,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
        ]
    ),
    cx=np.array(
        [
            0,
            1,
            -1,
            0,
            0,
            0,
            0,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            0,
            0,
            0,
            0,
        ]
    ),
    cy=np.array(
        [
            0,
            0,
            0,
            1,
            -1,
            0,
            0,
            1,
            -1,
            -1,
            1,
            0,
            0,
            0,
            0,
            1,
            -1,
            1,
            -1,
        ]
    ),
    cz=np.array(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            -1,
            0,
            0,
            0,
            0,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
        ]
    ),
    opp=np.array(
        [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
            18,
            17,
        ]
    ),
)
