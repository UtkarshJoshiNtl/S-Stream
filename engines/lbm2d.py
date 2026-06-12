from __future__ import annotations

import numpy as np
from numba import njit, prange

from engines.base import SimEngine
from engines.lbm_common import LATTICE_2D


@njit(parallel=True)
def _fused_step_nb(
    f, rho, u, v, obstacles, opp, w, cx, cy, omega, u_inflow, height, width,
):
    f_new = np.empty_like(f)
    for y in prange(height):
        for x in range(width):
            fi = np.empty(9, dtype=np.float32)
            for i in range(9):
                sx = x - cx[i]
                if sx < 0:
                    sx += width
                elif sx >= width:
                    sx -= width
                sy = y - cy[i]
                if sy < 0:
                    sy += height
                elif sy >= height:
                    sy -= height
                fi[i] = f[i, sy, sx]

            if obstacles[y, x]:
                for i in range(9):
                    opp_i = opp[i]
                    if i < opp_i:
                        tmp = fi[i]
                        fi[i] = fi[opp_i]
                        fi[opp_i] = tmp

            if x == 0:
                u2 = u_inflow * u_inflow
                for i in range(9):
                    cu = cx[i] * u_inflow
                    fi[i] = w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

            if y == 0 or y == height - 1:
                for i in range(9):
                    opp_i = opp[i]
                    if i < opp_i:
                        tmp = fi[i]
                        fi[i] = fi[opp_i]
                        fi[opp_i] = tmp

            r = 0.0
            u_vel = 0.0
            v_vel = 0.0
            for i in range(9):
                fiv = fi[i]
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
                f_new[i, y, x] = fi[i] * (1.0 - omega) + feq * omega

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel

    f[:] = f_new


@njit(parallel=True)
def _collide_nb(f, rho, u, v, w, cx, cy, omega, height, width):
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


@njit(parallel=True)
def _stream_nb(f, cx, cy, height, width):
    f_new = np.empty_like(f)
    for i in range(9):
        for y in prange(height):
            for x in range(width):
                sx = x - cx[i]
                if sx < 0:
                    sx += width
                elif sx >= width:
                    sx -= width
                sy = y - cy[i]
                if sy < 0:
                    sy += height
                elif sy >= height:
                    sy -= height
                f_new[i, y, x] = f[i, sy, sx]
    f[:] = f_new


@njit(parallel=True)
def _bounce_back_nb(f, mask, opp, height, width):
    for i in range(9):
        opp_i = opp[i]
        if i < opp_i:
            for y in prange(height):
                for x in range(width):
                    if mask[y, x]:
                        tmp = f[i, y, x]
                        f[i, y, x] = f[opp_i, y, x]
                        f[opp_i, y, x] = tmp


class LBM2D(SimEngine):
    """D2Q9 Lattice Boltzmann fluid simulation with Numba-accelerated solver."""

    def __init__(
        self, width: int = 128, height: int = 128, viscosity: float = 0.02
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.15

        self.lattice = LATTICE_2D
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )

        self.f = np.zeros((9, height, width), dtype=np.float32)

        self.rho = np.ones((height, width), dtype=np.float32)
        self.u = np.zeros((height, width), dtype=np.float32)
        self.v = np.zeros((height, width), dtype=np.float32)

        self.obstacles = np.zeros((height, width), dtype=np.bool_)

        self.smoke = np.zeros((height, width), dtype=np.float32)
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, float]] = []

        self._lap_buffer = np.empty_like(self.smoke)
        self._x_coords = np.arange(width, dtype=np.float32)
        self._y_coords = np.arange(height, dtype=np.float32)

        self.initialize(rho=1.0, u=0.1, v=0.0)

    # --- SimEngine interface ---

    @property
    def ndim(self) -> int:
        return 2

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return (self.height, self.width)

    @property
    def omega(self) -> float:
        return self.lattice.omega_from_viscosity(self.viscosity)

    def initialize(
        self, rho: float = 1.0, u: float = 0.1, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.f = self.lattice.equilibrium(self.rho, self.u, self.v)
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

    def step(self) -> None:
        _fused_step_nb(
            self.f, self.rho, self.u, self.v,
            self.obstacles,
            self.lattice.opp, self.lattice.w,
            self.lattice.cx, self.lattice.cy,
            self.omega, self.u_inflow,
            self.height, self.width,
        )
        self.f[:, :, -1] = self.f[:, :, -2]
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

    # --- LBM internals (kept for testing) ---

    def collision(self) -> None:
        _collide_nb(
            self.f, self.rho, self.u, self.v,
            self.lattice.w, self.lattice.cx, self.lattice.cy,
            self.omega, self.height, self.width,
        )

    def streaming(self) -> None:
        _stream_nb(self.f, self.lattice.cx, self.lattice.cy, self.height, self.width)

    def apply_obstacles(self) -> None:
        _bounce_back_nb(
            self.f, self.obstacles, self.lattice.opp, self.height, self.width
        )

    def apply_inflow(self) -> None:
        u_in = self.u_inflow
        u2 = u_in * u_in
        cu = self.lattice.cx * u_in
        feq = self.lattice.w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
        self.f[:, :, 0] = feq[:, np.newaxis]

    def apply_outflow(self) -> None:
        self.f[:, :, -1] = self.f[:, :, -2]

    def apply_walls(self) -> None:
        for i in range(9):
            opp_i = self.lattice.opp[i]
            if i < opp_i:
                tmp_top = self.f[i, 0, :].copy()
                self.f[i, 0, :] = self.f[opp_i, 0, :]
                self.f[opp_i, 0, :] = tmp_top
                tmp_bot = self.f[i, -1, :].copy()
                self.f[i, -1, :] = self.f[opp_i, -1, :]
                self.f[opp_i, -1, :] = tmp_bot

    def apply_emitters(self) -> None:
        for x, y, strength in self.emitters:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.smoke[y, x] += strength
        np.clip(self.smoke, 0, 1, out=self.smoke)

    def advect_smoke(self) -> None:
        u_adv = np.where(self.obstacles, 0.0, self.u)
        v_adv = np.where(self.obstacles, 0.0, self.v)

        x_orig = self._x_coords[np.newaxis, :] - u_adv
        y_orig = self._y_coords[:, np.newaxis] - v_adv
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
        lap = self._lap_buffer
        lap[:] = 0.0
        lap[1:] += s[:-1] - s[1:]
        lap[:-1] += s[1:] - s[:-1]
        lap[:, 1:] += s[:, :-1] - s[:, 1:]
        lap[:, :-1] += s[:, 1:] - s[:, :-1]
        s += d * lap

    def decay_smoke(self) -> None:
        self.smoke *= self.smoke_decay
