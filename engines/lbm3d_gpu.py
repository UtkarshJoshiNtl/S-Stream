from __future__ import annotations

import cupy as cp
import numpy as np

from engines.base import SimEngine
from engines.lbm_common import LATTICE_3D


class LBM3DGPU(SimEngine):
    """D3Q19 Lattice Boltzmann fluid simulation with passive scalar smoke (GPU/CuPy)."""

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        depth: int = 128,
        viscosity: float = 0.02,
    ) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.viscosity = viscosity
        self.u_inflow = 0.15

        # Copy lattice data to CuPy float32 arrays for GPU usage
        self._lattice = LATTICE_3D
        self._lattice.assert_stable(
            viscosity, self._lattice.omega_from_viscosity(viscosity)
        )

        self.w = cp.array(self._lattice.w, dtype=cp.float32)
        self.cx = cp.array(self._lattice.cx, dtype=cp.float32)
        self.cy = cp.array(self._lattice.cy, dtype=cp.float32)
        self.cz = cp.array(self._lattice.cz, dtype=cp.float32)
        self.opp = cp.array(self._lattice.opp, dtype=cp.int32)

        self.f = cp.zeros((19, depth, height, width), dtype=cp.float32)

        self.rho = cp.ones((depth, height, width), dtype=cp.float32)
        self.u = cp.zeros((depth, height, width), dtype=cp.float32)
        self.v = cp.zeros((depth, height, width), dtype=cp.float32)
        self.w_vel = cp.zeros((depth, height, width), dtype=cp.float32)

        self.obstacles = cp.zeros((depth, height, width), dtype=cp.bool_)

        self.smoke = cp.zeros((depth, height, width), dtype=cp.float32)
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, int, float]] = []

        z, y, x = cp.meshgrid(
            cp.arange(depth, dtype=cp.float32),
            cp.arange(height, dtype=cp.float32),
            cp.arange(width, dtype=cp.float32),
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

    @property
    def omega(self) -> float:
        return self._lattice.omega_from_viscosity(self.viscosity)

    def initialize(
        self, rho: float = 1.0, u: float = 0.1, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.w_vel[:] = w
        self.f = self._equilibrium(self.rho, self.u, self.v, self.w_vel)
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

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
        return cp.asnumpy(self.rho)

    def get_velocity(self) -> np.ndarray:
        return cp.asnumpy(cp.stack([self.u, self.v, self.w_vel], axis=3))

    def get_smoke(self) -> np.ndarray:
        return cp.asnumpy(self.smoke)

    def get_obstacles(self) -> np.ndarray:
        return cp.asnumpy(self.obstacles)

    def get_emitter_count(self) -> int:
        return len(self.emitters)

    def add_obstacle(self, x: int, y: int, z: int, radius: int = 5) -> None:
        z_grid, y_grid, x_grid = cp.ogrid[: self.depth, : self.height, : self.width]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 + (z_grid - z) ** 2 <= radius**2
        self.obstacles[mask] = True

    def clear_obstacles(self) -> None:
        self.obstacles[:] = False

    def add_emitter(self, x: int, y: int, z: int, strength: float = 0.05) -> None:
        self.emitters.append((x, y, z, strength))

    def clear_emitters(self) -> None:
        self.emitters.clear()

    # --- LBM internals ---

    def _equilibrium(
        self,
        rho: cp.ndarray,
        u: cp.ndarray,
        v: cp.ndarray,
        w_vel: cp.ndarray,
    ) -> cp.ndarray:
        feq = cp.zeros((19, *rho.shape), dtype=cp.float32)
        u2 = u**2 + v**2 + w_vel**2
        for i in range(19):
            cu = self.cx[i] * u + self.cy[i] * v + self.cz[i] * w_vel
            feq[i] = self.w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    def _shift(self, i: int) -> tuple[int, int, int]:
        cz_i = int(self.cz[i].item()) if hasattr(self.cz[i], "item") else self.cz[i]
        cy_i = int(self.cy[i].item()) if hasattr(self.cy[i], "item") else self.cy[i]
        cx_i = int(self.cx[i].item()) if hasattr(self.cx[i], "item") else self.cx[i]
        return (cz_i, cy_i, cx_i)

    def collision(self) -> None:
        self.rho = cp.sum(self.f, axis=0)
        rho_safe = cp.where(self.rho > 0, self.rho, 1.0)
        c = self.cx[:, cp.newaxis, cp.newaxis, cp.newaxis]
        self.u = cp.sum(self.f * c, axis=0) / rho_safe
        c = self.cy[:, cp.newaxis, cp.newaxis, cp.newaxis]
        self.v = cp.sum(self.f * c, axis=0) / rho_safe
        c = self.cz[:, cp.newaxis, cp.newaxis, cp.newaxis]
        self.w_vel = cp.sum(self.f * c, axis=0) / rho_safe

        feq = self._equilibrium(self.rho, self.u, self.v, self.w_vel)
        self.f = self.f * (1 - self.omega) + feq * self.omega

    def streaming(self) -> None:
        for i in range(19):
            shift = self._shift(i)
            self.f[i] = cp.roll(self.f[i], shift=shift, axis=(0, 1, 2))

    def apply_obstacles(self) -> None:
        for i in range(19):
            opp_i = int(self.opp[i])
            if i < opp_i:
                tmp = self.f[i][self.obstacles].copy()
                self.f[i][self.obstacles] = self.f[opp_i][self.obstacles]
                self.f[opp_i][self.obstacles] = tmp

    def apply_inflow(self) -> None:
        rho_inlet = 1.0
        u_inlet = cp.full((self.depth, self.height), self.u_inflow, dtype=cp.float32)
        v_inlet = cp.zeros((self.depth, self.height), dtype=cp.float32)
        w_inlet = cp.zeros((self.depth, self.height), dtype=cp.float32)
        for i in range(19):
            cu = self.cx[i] * u_inlet + self.cy[i] * v_inlet + self.cz[i] * w_inlet
            u2 = u_inlet**2 + v_inlet**2 + w_inlet**2
            feq = self.w[i] * rho_inlet * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
            self.f[i, :, :, 0] = feq

    def apply_outflow(self) -> None:
        for i in range(19):
            self.f[i, :, :, -1] = self.f[i, :, :, -2]

    def apply_walls(self) -> None:
        f_copy = self.f.copy()
        for i in range(19):
            opp_i = int(self.opp[i])
            self.f[i, :, 0, :] = f_copy[opp_i, :, 0, :]
            self.f[i, :, -1, :] = f_copy[opp_i, :, -1, :]
            self.f[i, 0, :, :] = f_copy[opp_i, 0, :, :]
            self.f[i, -1, :, :] = f_copy[opp_i, -1, :, :]

    def apply_emitters(self) -> None:
        for x, y, z, strength in self.emitters:
            self.smoke[z, y, x] += strength
        cp.clip(self.smoke, 0, 1, out=self.smoke)

    def advect_smoke(self) -> None:
        u_adv = cp.where(self.obstacles, 0.0, self.u)
        v_adv = cp.where(self.obstacles, 0.0, self.v)
        w_adv = cp.where(self.obstacles, 0.0, self.w_vel)

        x_orig = self._grid_x - u_adv
        y_orig = self._grid_y - v_adv
        z_orig = self._grid_z - w_adv

        x_orig = cp.clip(x_orig, 0, self.width - 1)
        y_orig = cp.clip(y_orig, 0, self.height - 1)
        z_orig = cp.clip(z_orig, 0, self.depth - 1)

        x0 = cp.floor(x_orig).astype(cp.int32)
        y0 = cp.floor(y_orig).astype(cp.int32)
        z0 = cp.floor(z_orig).astype(cp.int32)
        x1 = cp.minimum(x0 + 1, self.width - 1)
        y1 = cp.minimum(y0 + 1, self.height - 1)
        z1 = cp.minimum(z0 + 1, self.depth - 1)

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
        lap = cp.zeros_like(s)
        lap[1:] += s[:-1] - s[1:]
        lap[:-1] += s[1:] - s[:-1]
        lap[:, 1:] += s[:, :-1] - s[:, 1:]
        lap[:, :-1] += s[:, 1:] - s[:, :-1]
        lap[:, :, 1:] += s[:, :, :-1] - s[:, :, 1:]
        lap[:, :, :-1] += s[:, :, 1:] - s[:, :, :-1]
        s += d * lap

    def decay_smoke(self) -> None:
        self.smoke *= self.smoke_decay
