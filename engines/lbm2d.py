from __future__ import annotations

import numpy as np

from engines.base import SimEngine
from engines.lbm_common import LATTICE_2D


class LBM2D(SimEngine):
    """D2Q9 Lattice Boltzmann fluid simulation with passive scalar smoke."""

    def __init__(
        self, width: int = 128, height: int = 128, viscosity: float = 0.02
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.15

        self.lattice = LATTICE_2D
        self.omega = self.lattice.omega_from_viscosity(viscosity)
        self.lattice.assert_stable(viscosity, self.omega)

        self.f = np.zeros((9, height, width))

        self.rho = np.ones((height, width))
        self.u = np.zeros((height, width))
        self.v = np.zeros((height, width))

        self.obstacles = np.zeros((height, width), dtype=bool)

        self.smoke = np.zeros((height, width))
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, float]] = []

        self._grid_x, self._grid_y = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
        )

        self.initialize(rho=1.0, u=0.1, v=0.0)

    # --- SimEngine interface ---

    @property
    def ndim(self) -> int:
        return 2

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return (self.height, self.width)

    def initialize(
        self, rho: float = 1.0, u: float = 0.1, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.f = self.lattice.equilibrium(self.rho, self.u, self.v)
        self.smoke[:] = 0.0
        self.emitters.clear()

    def step(self) -> None:
        self.streaming()
        self.apply_obstacles()
        self.apply_inflow()
        self.apply_outflow()
        self.apply_walls()
        self.collision()
        self.apply_emitters()
        self.advect_smoke()
        self.diffuse_smoke()
        self.smoke[self.obstacles] = 0.0
        self.decay_smoke()

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def get_density(self) -> np.ndarray:
        return self.rho.copy()

    def get_velocity(self) -> np.ndarray:
        return np.stack([self.u, self.v], axis=2)

    def get_smoke(self) -> np.ndarray:
        return self.smoke.copy()

    def get_obstacles(self) -> np.ndarray:
        return self.obstacles.copy()

    def get_emitter_count(self) -> int:
        return len(self.emitters)

    def add_obstacle(self, x: int, y: int, radius: int = 5) -> None:
        y_grid, x_grid = np.ogrid[: self.height, : self.width]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius**2
        self.obstacles[mask] = True

    def clear_obstacles(self) -> None:
        self.obstacles[:] = False

    def add_emitter(self, x: int, y: int, strength: float = 0.05) -> None:
        self.emitters.append((x, y, strength))

    def clear_emitters(self) -> None:
        self.emitters.clear()

    # --- LBM internals ---

    def collision(self) -> None:
        self.rho = np.sum(self.f, axis=0)
        rho_safe = np.where(self.rho > 0, self.rho, 1.0)
        c = self.lattice.cx[:, np.newaxis, np.newaxis]
        self.u = np.sum(self.f * c, axis=0) / rho_safe
        c = self.lattice.cy[:, np.newaxis, np.newaxis]
        self.v = np.sum(self.f * c, axis=0) / rho_safe

        feq = self.lattice.equilibrium(self.rho, self.u, self.v)
        self.f = self.f * (1 - self.omega) + feq * self.omega

    def streaming(self) -> None:
        for i in range(9):
            self.f[i] = np.roll(
                self.f[i], shift=(self.lattice.cx[i], self.lattice.cy[i]), axis=(1, 0)
            )

    def apply_obstacles(self) -> None:
        for i in range(9):
            self.f[i][self.obstacles] = self.f[self.lattice.opp[i]][self.obstacles]

    def apply_inflow(self) -> None:
        rho_inlet = 1.0
        u_inlet = np.full(self.height, self.u_inflow)
        v_inlet = np.zeros(self.height)
        for i in range(9):
            cu = self.lattice.cx[i] * u_inlet + self.lattice.cy[i] * v_inlet
            u2 = u_inlet**2 + v_inlet**2
            feq = self.lattice.w[i] * rho_inlet * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
            self.f[i, :, 0] = feq

    def apply_outflow(self) -> None:
        for i in range(9):
            self.f[i, :, -1] = self.f[i, :, -2]

    def apply_walls(self) -> None:
        for i in range(9):
            self.f[i, 0, :] = self.f[self.lattice.opp[i], 0, :]
        for i in range(9):
            self.f[i, -1, :] = self.f[self.lattice.opp[i], -1, :]

    def apply_emitters(self) -> None:
        for x, y, strength in self.emitters:
            self.smoke[y, x] += strength
        np.clip(self.smoke, 0, 1, out=self.smoke)

    def advect_smoke(self) -> None:
        u_adv = np.where(self.obstacles, 0.0, self.u)
        v_adv = np.where(self.obstacles, 0.0, self.v)

        x_orig = self._grid_x - u_adv
        y_orig = self._grid_y - v_adv
        x_orig = np.clip(x_orig, 0, self.width - 1)
        y_orig = np.clip(y_orig, 0, self.height - 1)

        x0 = np.floor(x_orig).astype(np.int32)
        y0 = np.floor(y_orig).astype(np.int32)
        x1 = np.minimum(x0 + 1, self.width - 1)
        y1 = np.minimum(y0 + 1, self.height - 1)

        fx = x_orig - x0
        fy = y_orig - y0

        c00 = self.smoke[y0, x0]
        c10 = self.smoke[y0, x1]
        c01 = self.smoke[y1, x0]
        c11 = self.smoke[y1, x1]

        self.smoke = (
            c00 * (1 - fx) * (1 - fy)
            + c10 * fx * (1 - fy)
            + c01 * (1 - fx) * fy
            + c11 * fx * fy
        )

    def diffuse_smoke(self) -> None:
        s = self.smoke
        d = self.smoke_diffusion
        lap = np.zeros_like(s)
        lap[1:] += s[:-1] - s[1:]
        lap[:-1] += s[1:] - s[:-1]
        lap[:, 1:] += s[:, :-1] - s[:, 1:]
        lap[:, :-1] += s[:, 1:] - s[:, :-1]
        s += d * lap

    def decay_smoke(self) -> None:
        self.smoke *= self.smoke_decay
