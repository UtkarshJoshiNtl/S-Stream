from __future__ import annotations

import numpy as np


class CPULBM3D:
    """
    CPU-based D3Q19 Lattice Boltzmann Method implementation
    with passive scalar smoke advection in 3D.
    """

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        depth: int = 64,
        viscosity: float = 0.02,
    ) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.viscosity = viscosity

        self.omega = 1.0 / (3.0 * viscosity + 0.5)
        assert 0 < self.omega < 2, (
            f"omega={self.omega:.3f} outside stable range (0, 2) "
            f"for viscosity={viscosity}"
        )

        self.w = np.array([
            1 / 3,
            1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
        ])

        self.cx = np.array([
            0,  1, -1,  0,  0,  0,  0,
            1, -1,  1, -1,  1, -1,  1, -1,
            0,  0,  0,  0,
        ])

        self.cy = np.array([
            0,  0,  0,  1, -1,  0,  0,
            1, -1, -1,  1,  0,  0,  0,  0,
            1, -1,  1, -1,
        ])

        self.cz = np.array([
            0,  0,  0,  0,  0,  1, -1,
            0,  0,  0,  0,  1, -1, -1,  1,
            1, -1, -1,  1,
        ])

        self.opp = np.array([
            0, 2, 1, 4, 3, 6, 5,
            8, 7, 10, 9, 12, 11, 14, 13,
            16, 15, 18, 17,
        ])

        self.f = np.zeros((19, depth, height, width))

        self.rho = np.ones((depth, height, width))
        self.u = np.zeros((depth, height, width))
        self.v = np.zeros((depth, height, width))
        self.w_vel = np.zeros((depth, height, width))

        self.obstacles = np.zeros((depth, height, width), dtype=bool)

        self.smoke = np.zeros((depth, height, width))
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, int, float]] = []

        self._grid_z, self._grid_y, self._grid_x = np.meshgrid(
            np.arange(depth), np.arange(height), np.arange(width), indexing='ij'
        )

        self.initialize(rho=1.0, u=0.1, v=0.0, w=0.0)

    def equilibrium(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w_vel: np.ndarray,
    ) -> np.ndarray:
        feq = np.zeros((19, self.depth, self.height, self.width))
        u2 = u**2 + v**2 + w_vel**2
        for i in range(19):
            cu = self.cx[i] * u + self.cy[i] * v + self.cz[i] * w_vel
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    def initialize(
        self,
        rho: float = 1.0,
        u: float = 0.1,
        v: float = 0.0,
        w: float = 0.0,
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.w_vel[:] = w
        self.f = self.equilibrium(self.rho, self.u, self.v, self.w_vel)
        self.smoke[:] = 0.0
        self.emitters.clear()

    def collision(self) -> None:
        self.rho = np.sum(self.f, axis=0)
        rho_safe = np.where(self.rho > 0, self.rho, 1.0)
        c = self.cx[:, np.newaxis, np.newaxis, np.newaxis]
        self.u = np.sum(self.f * c, axis=0) / rho_safe
        c = self.cy[:, np.newaxis, np.newaxis, np.newaxis]
        self.v = np.sum(self.f * c, axis=0) / rho_safe
        c = self.cz[:, np.newaxis, np.newaxis, np.newaxis]
        self.w_vel = np.sum(self.f * c, axis=0) / rho_safe

        feq = self.equilibrium(self.rho, self.u, self.v, self.w_vel)
        self.f = self.f * (1 - self.omega) + feq * self.omega

    def streaming(self) -> None:
        for i in range(19):
            self.f[i] = np.roll(
                self.f[i],
                shift=(self.cz[i], self.cy[i], self.cx[i]),
                axis=(0, 1, 2),
            )
