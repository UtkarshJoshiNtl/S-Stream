from __future__ import annotations

import numpy as np

from engines.base import SimEngine
from engines.lbm_common import LATTICE_3D


class LBM3DCPU(SimEngine):
    """D3Q19 Lattice Boltzmann fluid simulation with passive scalar smoke (CPU)."""

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
        self.u_inflow = 0.15

        self.lattice = LATTICE_3D
        self.omega = self.lattice.omega_from_viscosity(viscosity)
        self.lattice.assert_stable(viscosity, self.omega)

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

        z, y, x = np.meshgrid(
            np.arange(depth, dtype=np.float64),
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        self._grid_x = x
        self._grid_y = y
        self._grid_z = z

        self.initialize(rho=1.0, u=0.1, v=0.0, w=0.0)

    # --- SimEngine interface ---

    @property
    def ndim(self) -> int:
        return 3

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return (self.depth, self.height, self.width)

    def initialize(
        self, rho: float = 1.0, u: float = 0.1, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.w_vel[:] = w
        self.f = self.lattice.equilibrium(self.rho, self.u, self.v, self.w_vel)
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
        return np.stack([self.u, self.v, self.w_vel], axis=3)

    def get_smoke(self) -> np.ndarray:
        return self.smoke.copy()

    def get_obstacles(self) -> np.ndarray:
        return self.obstacles.copy()

    def get_emitter_count(self) -> int:
        return len(self.emitters)

    def add_obstacle(self, x: int, y: int, z: int, radius: int = 5) -> None:
        z_grid, y_grid, x_grid = np.ogrid[: self.depth, : self.height, : self.width]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 + (z_grid - z) ** 2 <= radius**2
        self.obstacles[mask] = True

    def clear_obstacles(self) -> None:
        self.obstacles[:] = False

    def add_emitter(self, x: int, y: int, z: int, strength: float = 0.05) -> None:
        self.emitters.append((x, y, z, strength))

    def clear_emitters(self) -> None:
        self.emitters.clear()

    # --- LBM internals ---

    def collision(self) -> None:
        self.rho = np.sum(self.f, axis=0)
        rho_safe = np.where(self.rho > 0, self.rho, 1.0)
        c = self.lattice.cx[:, np.newaxis, np.newaxis, np.newaxis]
        self.u = np.sum(self.f * c, axis=0) / rho_safe
        c = self.lattice.cy[:, np.newaxis, np.newaxis, np.newaxis]
        self.v = np.sum(self.f * c, axis=0) / rho_safe
        c = self.lattice.cz[:, np.newaxis, np.newaxis, np.newaxis]
        self.w_vel = np.sum(self.f * c, axis=0) / rho_safe

        feq = self.lattice.equilibrium(self.rho, self.u, self.v, self.w_vel)
        self.f = self.f * (1 - self.omega) + feq * self.omega

    def streaming(self) -> None:
        for i in range(19):
            self.f[i] = np.roll(
                self.f[i],
                shift=(self.lattice.cz[i], self.lattice.cy[i], self.lattice.cx[i]),
                axis=(0, 1, 2),
            )

    def apply_obstacles(self) -> None:
        for i in range(19):
            self.f[i][self.obstacles] = self.f[self.lattice.opp[i]][self.obstacles]

    def apply_inflow(self) -> None:
        rho_inlet = 1.0
        u_inlet = np.full((self.depth, self.height), self.u_inflow)
        v_inlet = np.zeros((self.depth, self.height))
        w_inlet = np.zeros((self.depth, self.height))
        for i in range(19):
            cu = (
                self.lattice.cx[i] * u_inlet
                + self.lattice.cy[i] * v_inlet
                + self.lattice.cz[i] * w_inlet
            )
            u2 = u_inlet**2 + v_inlet**2 + w_inlet**2
            feq = self.lattice.w[i] * rho_inlet * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
            self.f[i, :, :, 0] = feq

    def apply_outflow(self) -> None:
        for i in range(19):
            self.f[i, :, :, -1] = self.f[i, :, :, -2]

    def apply_walls(self) -> None:
        for i in range(19):
            self.f[i, :, 0, :] = self.f[self.lattice.opp[i], :, 0, :]
            self.f[i, :, -1, :] = self.f[self.lattice.opp[i], :, -1, :]
            self.f[i, 0, :, :] = self.f[self.lattice.opp[i], 0, :, :]
            self.f[i, -1, :, :] = self.f[self.lattice.opp[i], -1, :, :]

    def apply_emitters(self) -> None:
        for x, y, z, strength in self.emitters:
            self.smoke[z, y, x] += strength
        np.clip(self.smoke, 0, 1, out=self.smoke)

    def advect_smoke(self) -> None:
        u_adv = np.where(self.obstacles, 0.0, self.u)
        v_adv = np.where(self.obstacles, 0.0, self.v)
        w_adv = np.where(self.obstacles, 0.0, self.w_vel)

        x_orig = self._grid_x - u_adv
        y_orig = self._grid_y - v_adv
        z_orig = self._grid_z - w_adv

        x_orig = np.clip(x_orig, 0, self.width - 1)
        y_orig = np.clip(y_orig, 0, self.height - 1)
        z_orig = np.clip(z_orig, 0, self.depth - 1)

        x0 = np.floor(x_orig).astype(np.int32)
        y0 = np.floor(y_orig).astype(np.int32)
        z0 = np.floor(z_orig).astype(np.int32)
        x1 = np.minimum(x0 + 1, self.width - 1)
        y1 = np.minimum(y0 + 1, self.height - 1)
        z1 = np.minimum(z0 + 1, self.depth - 1)

        fx = x_orig - x0
        fy = y_orig - y0
        fz = z_orig - z0

        c000 = self.smoke[z0, y0, x0]
        c100 = self.smoke[z0, y0, x1]
        c010 = self.smoke[z0, y1, x0]
        c110 = self.smoke[z0, y1, x1]
        c001 = self.smoke[z1, y0, x0]
        c101 = self.smoke[z1, y0, x1]
        c011 = self.smoke[z1, y1, x0]
        c111 = self.smoke[z1, y1, x1]

        self.smoke = (
            c000 * (1 - fx) * (1 - fy) * (1 - fz)
            + c100 * fx * (1 - fy) * (1 - fz)
            + c010 * (1 - fx) * fy * (1 - fz)
            + c110 * fx * fy * (1 - fz)
            + c001 * (1 - fx) * (1 - fy) * fz
            + c101 * fx * (1 - fy) * fz
            + c011 * (1 - fx) * fy * fz
            + c111 * fx * fy * fz
        )

    def diffuse_smoke(self) -> None:
        s = self.smoke
        d = self.smoke_diffusion
        lap = np.zeros_like(s)
        lap[1:] += s[:-1] - s[1:]
        lap[:-1] += s[1:] - s[:-1]
        lap[:, 1:] += s[:, :-1] - s[:, 1:]
        lap[:, :-1] += s[:, 1:] - s[:, :-1]
        lap[:, :, 1:] += s[:, :, :-1] - s[:, :, 1:]
        lap[:, :, :-1] += s[:, :, 1:] - s[:, :, :-1]
        s += d * lap

    def decay_smoke(self) -> None:
        self.smoke *= self.smoke_decay
