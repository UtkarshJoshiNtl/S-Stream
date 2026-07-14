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
        cu = (
            self.cx[:, np.newaxis, np.newaxis] * u[np.newaxis, :, :]
            + self.cy[:, np.newaxis, np.newaxis] * v[np.newaxis, :, :]
        )
        u2 = u**2 + v**2
        return (
            self.w[:, np.newaxis, np.newaxis]
            * rho[np.newaxis, :, :]
            * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2[np.newaxis, :, :])
        )

    @staticmethod
    def omega_from_viscosity(nu: float) -> float:
        return 1.0 / (3.0 * nu + 0.5)

    def assert_stable(self, nu: float, omega: float) -> None:
        assert (
            0 < omega < 2
        ), f"omega={omega:.3f} outside stable range (0, 2) for viscosity={nu}"


LATTICE_2D = Lattice2D(
    w=np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
        dtype=np.float32,
    ),
    cx=np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32),
    cy=np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32),
    opp=np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32),
)


@dataclass(frozen=True)
class Lattice3D:
    """3D lattice constants (D3Q19 or D3Q27)."""

    w: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    opp: np.ndarray
    n_velocities: int

    def equilibrium(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute equilibrium distribution for 3D.

        Args:
            rho: Density field (D, H, W).
            u: X-velocity field (D, H, W).
            v: Y-velocity field (D, H, W).
            w_vel: Z-velocity field (D, H, W).

        Returns:
            Equilibrium distribution (Q, D, H, W).
        """
        cu = (
            self.cx[:, np.newaxis, np.newaxis, np.newaxis] * u[np.newaxis, :, :, :]
            + self.cy[:, np.newaxis, np.newaxis, np.newaxis] * v[np.newaxis, :, :, :]
            + self.cz[:, np.newaxis, np.newaxis, np.newaxis] * w_vel[np.newaxis, :, :, :]
        )
        u2 = u**2 + v**2 + w_vel**2
        return (
            self.w[:, np.newaxis, np.newaxis, np.newaxis]
            * rho[np.newaxis, :, :, :]
            * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2[np.newaxis, :, :, :])
        )

    @staticmethod
    def omega_from_viscosity(nu: float) -> float:
        return 1.0 / (3.0 * nu + 0.5)

    def assert_stable(self, nu: float, omega: float) -> None:
        assert (
            0 < omega < 2
        ), f"omega={omega:.3f} outside stable range (0, 2) for viscosity={nu}"


def _build_d3q19() -> Lattice3D:
    """Build D3Q19 lattice constants.

    D3Q19 has 19 velocity directions:
    - 1 rest (0,0,0)
    - 6 face-center (±1,0,0), (0,±1,0), (0,0,±1)
    - 12 edge-center (±1,±1,0), (±1,0,±1), (0,±1,±1)

    Reference: d'Humières et al. (2002), "Multiple-relaxation-time Lattice
    Boltzmann models for 3D simulations."
    """
    # fmt: off
    cx = np.array([
        0,   # 0: rest
        1, -1, 0,  0, 0,  0,   # 1-6: face-center
        1, -1,  1, -1, 0, 0,  1, -1,  1, -1,  # 7-18: edge-center
    ], dtype=np.int32)

    cy = np.array([
        0,   # 0: rest
        0,  0, 1, -1, 0,  0,   # 1-6: face-center
        1, -1, -1,  1, 1, -1, 0,  0,  0,  0,  # 7-18: edge-center
    ], dtype=np.int32)

    cz = np.array([
        0,   # 0: rest
        0,  0, 0,  0, 1, -1,   # 1-6: face-center
        0,  0,  0,  0, 1, -1, 1, -1, -1,  1,  # 7-18: edge-center
    ], dtype=np.int32)

    # Weights: 1/3 for rest, 1/18 for face-center, 1/36 for edge-center
    w = np.array([
        1 / 3,                                          # 0: rest
        1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,  # 1-6: face-center
        1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,  # 7-12: edge-center
        1 / 36, 1 / 36, 1 / 36, 1 / 36,                  # 13-16: edge-center
        1 / 36, 1 / 36,                                   # 17-18: edge-center (wait, need to recount)
    ], dtype=np.float32)
    # fmt: on

    # Recount: we have 19 velocities. Let me rebuild properly.
    # fmt: off
    velocities = [
        ( 0,  0,  0),  # 0: rest
        ( 1,  0,  0),  # 1
        (-1,  0,  0),  # 2
        ( 0,  1,  0),  # 3
        ( 0, -1,  0),  # 4
        ( 0,  0,  1),  # 5
        ( 0,  0, -1),  # 6
        ( 1,  1,  0),  # 7
        (-1, -1,  0),  # 8
        ( 1, -1,  0),  # 9
        (-1,  1,  0),  # 10
        ( 1,  0,  1),  # 11
        (-1,  0, -1),  # 12
        ( 1,  0, -1),  # 13
        (-1,  0,  1),  # 14
        ( 0,  1,  1),  # 15
        ( 0, -1, -1),  # 16
        ( 0,  1, -1),  # 17
        ( 0, -1,  1),  # 18
    ]
    # fmt: on

    cx = np.array([v[0] for v in velocities], dtype=np.int32)
    cy = np.array([v[1] for v in velocities], dtype=np.int32)
    cz = np.array([v[2] for v in velocities], dtype=np.int32)

    # Weights
    w = np.full(19, 1 / 36, dtype=np.float32)
    w[0] = 1 / 3  # rest
    for i in range(1, 7):
        w[i] = 1 / 18  # face-center

    # Opposite direction indices
    opp = np.zeros(19, dtype=np.int32)
    opp[0] = 0  # rest
    opp[1], opp[2] = 2, 1  # ±x
    opp[3], opp[4] = 4, 3  # ±y
    opp[5], opp[6] = 6, 5  # ±z
    opp[7], opp[8] = 8, 7  # (1,1,0) ↔ (-1,-1,0)
    opp[9], opp[10] = 10, 9  # (1,-1,0) ↔ (-1,1,0)
    opp[11], opp[12] = 12, 11  # (1,0,1) ↔ (-1,0,-1)
    opp[13], opp[14] = 14, 13  # (1,0,-1) ↔ (-1,0,1)
    opp[15], opp[16] = 16, 15  # (0,1,1) ↔ (0,-1,-1)
    opp[17], opp[18] = 18, 17  # (0,1,-1) ↔ (0,-1,1)

    return Lattice3D(
        w=w, cx=cx, cy=cy, cz=cz, opp=opp, n_velocities=19
    )


def _build_d3q27() -> Lattice3D:
    """Build D3Q27 lattice constants.

    D3Q27 has 27 velocity directions: all combinations of {-1, 0, 1}^3.
    Higher accuracy than D3Q19 for complex flows.

    Reference: zhao & feng (2020), "D3Q27 lattice Boltzmann model for
    simulating turbulent flows."
    """
    # Generate all 27 velocities: {-1, 0, 1}^3
    velocities = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                velocities.append((dx, dy, dz))

    cx = np.array([v[0] for v in velocities], dtype=np.int32)
    cy = np.array([v[1] for v in velocities], dtype=np.int32)
    cz = np.array([v[2] for v in velocities], dtype=np.int32)

    # Weights based on number of non-zero components
    w = np.zeros(27, dtype=np.float32)
    for i, (dx, dy, dz) in enumerate(velocities):
        n_nonzero = (dx != 0) + (dy != 0) + (dz != 0)
        if n_nonzero == 0:
            w[i] = 8 / 27  # rest
        elif n_nonzero == 1:
            w[i] = 2 / 27  # face-center (6 velocities)
        elif n_nonzero == 2:
            w[i] = 1 / 54  # edge-center (12 velocities)
        else:
            w[i] = 1 / 216  # corner (8 velocities)

    # Opposite direction: negate all components
    opp = np.zeros(27, dtype=np.int32)
    for i, (dx, dy, dz) in enumerate(velocities):
        target = (-dx, -dy, -dz)
        j = velocities.index(target)
        opp[i] = j

    return Lattice3D(
        w=w, cx=cx, cy=cy, cz=cz, opp=opp, n_velocities=27
    )


LATTICE_3D_Q19 = _build_d3q19()
LATTICE_3D_Q27 = _build_d3q27()
