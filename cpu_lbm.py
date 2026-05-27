from __future__ import annotations

import numpy as np


class CPULBM2D:
    """
    CPU-based D2Q9 Lattice Boltzmann Method implementation
    with passive scalar smoke advection.
    """

    def __init__(
        self, width: int = 128, height: int = 128, viscosity: float = 0.02
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity

        # Relaxation rate from viscosity (BGK): omega = 1 / (3*nu + 0.5)
        self.omega = 1.0 / (3.0 * viscosity + 0.5)

        # D2Q9 lattice: weights w[i], velocity vectors (cx[i], cy[i]),
        # and opposite-direction index opp[i] for bounce-back.
        self.w = np.array(
            [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
        )
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        self.f = np.zeros((9, height, width))

        self.rho = np.ones((height, width))
        self.u = np.zeros((height, width))
        self.v = np.zeros((height, width))

        self.obstacles = np.zeros((height, width), dtype=bool)

        self.smoke = np.zeros((height, width))
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, float]] = []

        self._grid_x, self._grid_y = np.meshgrid(np.arange(width), np.arange(height))

        self.initialize(rho=1.0, u=0.1, v=0.0)

    def equilibrium(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        feq = np.zeros((9, self.height, self.width))
        u2 = u**2 + v**2
        for i in range(9):
            cu = self.cx[i] * u + self.cy[i] * v
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    def initialize(self, rho: float = 1.0, u: float = 0.1, v: float = 0.0) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.f = self.equilibrium(self.rho, self.u, self.v)
        self.smoke[:] = 0.0
        self.emitters.clear()

    def collision(self) -> None:
        self.rho = np.sum(self.f, axis=0)
        # Prevent division-by-zero at walls/obstacles where rho may be zero
        rho_safe = np.where(self.rho > 0, self.rho, 1.0)
        self.u = np.sum(self.f * self.cx[:, np.newaxis, np.newaxis], axis=0) / rho_safe
        self.v = np.sum(self.f * self.cy[:, np.newaxis, np.newaxis], axis=0) / rho_safe

        # BGK relaxation toward equilibrium
        feq = self.equilibrium(self.rho, self.u, self.v)
        self.f = self.f * (1 - self.omega) + feq * self.omega

    def streaming(self) -> None:
        for i in range(9):
            self.f[i] = np.roll(self.f[i], shift=(self.cx[i], self.cy[i]), axis=(1, 0))

    def apply_obstacles(self) -> None:
        self.smoke[self.obstacles] = 0.0
        for i in range(9):
            self.f[i][self.obstacles] = self.f[self.opp[i]][self.obstacles]

    def apply_inflow(self, u_inflow: float = 0.15) -> None:
        rho_inlet = 1.0
        u_inlet = np.full(self.height, u_inflow)
        v_inlet = np.zeros(self.height)
        for i in range(9):
            cu = self.cx[i] * u_inlet + self.cy[i] * v_inlet
            u2 = u_inlet**2 + v_inlet**2
            feq = self.w[i] * rho_inlet * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
            self.f[i, :, 0] = feq

    def apply_outflow(self) -> None:
        for i in range(9):
            self.f[i, :, -1] = self.f[i, :, -2]

    def apply_walls(self) -> None:
        for i in range(9):
            self.f[i, 0, :] = self.f[self.opp[i], 0, :]
        for i in range(9):
            self.f[i, -1, :] = self.f[self.opp[i], -1, :]

    def apply_emitters(self) -> None:
        for x, y, strength in self.emitters:
            self.smoke[y, x] += strength
            self.smoke = np.minimum(self.smoke, 1.0)

    def advect_smoke(self) -> None:
        xs = self._grid_x
        ys = self._grid_y
        x_orig = xs - self.u
        y_orig = ys - self.v
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
        self.smoke += self.smoke_diffusion * (
            np.roll(self.smoke, 1, axis=0)
            + np.roll(self.smoke, -1, axis=0)
            + np.roll(self.smoke, 1, axis=1)
            + np.roll(self.smoke, -1, axis=1)
            - 4 * self.smoke
        )

    def decay_smoke(self) -> None:
        self.smoke *= self.smoke_decay

    def step(self) -> None:
        # Order matters: obstacles bounce-back before boundary conditions,
        # collision after boundaries, smoke advection after emitters.
        self.streaming()
        self.apply_obstacles()
        self.apply_inflow(u_inflow=0.15)
        self.apply_outflow()
        self.apply_walls()
        self.collision()
        self.apply_emitters()
        self.advect_smoke()
        self.diffuse_smoke()
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
