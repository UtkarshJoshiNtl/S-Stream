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
        cu = self.cx[:, np.newaxis, np.newaxis] * u[np.newaxis, :, :] + \
             self.cy[:, np.newaxis, np.newaxis] * v[np.newaxis, :, :]
        u2 = u**2 + v**2
        return self.w[:, np.newaxis, np.newaxis] * rho[np.newaxis, :, :] * \
            (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2[np.newaxis, :, :])

    @staticmethod
    def omega_from_viscosity(nu: float) -> float:
        return 1.0 / (3.0 * nu + 0.5)

    def assert_stable(self, nu: float, omega: float) -> None:
        assert 0 < omega < 2, (
            f"omega={omega:.3f} outside stable range (0, 2) for viscosity={nu}"
        )


LATTICE_2D = Lattice2D(
    w=np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
        dtype=np.float32,
    ),
    cx=np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32),
    cy=np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32),
    opp=np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32),
)
