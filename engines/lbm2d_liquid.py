from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from engines.base import SimEngine
from engines.lbm_common import LATTICE_2D

_FORCE_CLIP = 0.3
_VEL_CLIP = 0.3


@njit
def _psi(r: float) -> float:
    return 1.0 - math.exp(-r)


@njit(parallel=True)
def _compute_force_nb(
    rho, obstacles, w, cx, cy,
    g: float, g_adhesion: float,
    height: int, width: int,
    fx_out, fy_out,
):
    for y in prange(height):
        for x in range(width):
            if obstacles[y, x]:
                fx_out[y, x] = 0.0
                fy_out[y, x] = 0.0
                continue

            psix = _psi(rho[y, x])
            sfx = 0.0
            sfy = 0.0
            afx = 0.0
            afy = 0.0

            for i in range(9):
                sx = x + cx[i]
                sy = y + cy[i]
                if 0 <= sx < width and 0 <= sy < height:
                    if obstacles[sy, sx]:
                        afx += w[i] * cx[i]
                        afy += w[i] * cy[i]
                    else:
                        psij = _psi(rho[sy, sx])
                        sfx += w[i] * psij * cx[i]
                        sfy += w[i] * psij * cy[i]

            fx = -psix * (g * sfx + g_adhesion * afx)
            fy = -psix * (g * sfy + g_adhesion * afy)
            fx_out[y, x] = max(min(fx, _FORCE_CLIP), -_FORCE_CLIP)
            fy_out[y, x] = max(min(fy, _FORCE_CLIP), -_FORCE_CLIP)


@njit(parallel=True)
def _fused_step_liquid_nb(
    f, rho, u, v,
    fx, fy, obstacles,
    opp, w, cx, cy,
    omega: float,
    height: int, width: int,
):
    tau = 1.0 / omega
    f_new = np.empty_like(f)

    for y in prange(height):
        for x in range(width):
            fi = np.empty(9, dtype=np.float32)
            is_obstacle = obstacles[y, x]

            for i in range(9):
                sx = x - cx[i]
                sy = y - cy[i]
                if 0 <= sx < width and 0 <= sy < height:
                    fi[i] = f[i, sy, sx]
                else:
                    fi[i] = f[opp[i], y, x]

            if is_obstacle:
                for i in range(9):
                    oi = opp[i]
                    if i < oi:
                        tmp = fi[i]
                        fi[i] = fi[oi]
                        fi[oi] = tmp

            r = 0.0
            mom_x = 0.0
            mom_y = 0.0
            for i in range(9):
                fiv = fi[i]
                r += fiv
                mom_x += fiv * cx[i]
                mom_y += fiv * cy[i]

            rho_safe = r if r > 0 else 1e-6
            u_vel = mom_x / rho_safe
            v_vel = mom_y / rho_safe

            if not is_obstacle:
                u_eq = u_vel + tau * fx[y, x] / rho_safe
                v_eq = v_vel + tau * fy[y, x] / rho_safe
            else:
                u_eq = 0.0
                v_eq = 0.0

            mag = math.sqrt(u_eq * u_eq + v_eq * v_eq)
            if mag > _VEL_CLIP:
                u_eq *= _VEL_CLIP / mag
                v_eq *= _VEL_CLIP / mag

            u2 = u_eq * u_eq + v_eq * v_eq
            for i in range(9):
                cu = cx[i] * u_eq + cy[i] * v_eq
                feq = w[i] * r * (
                    1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2
                )
                f_new[i, y, x] = fi[i] * (1.0 - omega) + feq * omega

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel

    f[:] = f_new


class LBM2DLiquid(SimEngine):
    """D2Q9 Shan-Chen multiphase liquid simulation."""

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        viscosity: float = 0.02,
        g: float = -5.0,
        g_adhesion: float = -5.0,
        droplet_radius: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.0

        self.g = g
        self.g_adhesion = g_adhesion
        self.droplet_radius = droplet_radius

        self.lattice = LATTICE_2D
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )

        self.f = np.zeros((9, height, width), dtype=np.float32)

        self.rho = np.ones((height, width), dtype=np.float32)
        self.u = np.zeros((height, width), dtype=np.float32)
        self.v = np.zeros((height, width), dtype=np.float32)

        self.obstacles = np.zeros((height, width), dtype=np.bool_)

        self.fx = np.empty((height, width), dtype=np.float32)
        self.fy = np.empty((height, width), dtype=np.float32)

        self.smoke = np.zeros((height, width), dtype=np.float32)
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, float]] = []

        self._x_coords = np.arange(width, dtype=np.float32)
        self._y_coords = np.arange(height, dtype=np.float32)
        self._lap_buffer = np.empty_like(self.smoke)

        self.initialize()
        self.step()
        self.initialize()

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
        self, rho: float = 0.01, u: float = 0.0, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v

        radius = self.droplet_radius
        if radius is None:
            radius = min(self.width, self.height) // 6
        cx = self.width // 2
        cy = self.height // 2
        y_grid, x_grid = np.ogrid[: self.height, : self.width]
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
        self.rho[mask] = 2.0

        self.f = self.lattice.equilibrium(self.rho, self.u, self.v)
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

    def step(self) -> None:
        _compute_force_nb(
            self.rho, self.obstacles,
            self.lattice.w, self.lattice.cx, self.lattice.cy,
            self.g, self.g_adhesion,
            self.height, self.width,
            self.fx, self.fy,
        )
        _fused_step_liquid_nb(
            self.f, self.rho, self.u, self.v,
            self.fx, self.fy, self.obstacles,
            self.lattice.opp, self.lattice.w,
            self.lattice.cx, self.lattice.cy,
            self.omega, self.height, self.width,
        )
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
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2
        self.obstacles[mask] = True

    def clear_obstacles(self) -> None:
        self.obstacles[:] = False

    def add_emitter(self, x: int, y: int, strength: float = 0.05) -> None:
        self.emitters.append((x, y, strength))

    def clear_emitters(self) -> None:
        self.emitters.clear()

    # --- Smoke helpers ---

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
        x0 = np.floor(x_orig).astype(np.int32)
        y0 = np.floor(y_orig).astype(np.int32)
        x0 = np.clip(x0, 0, self.width - 2)
        y0 = np.clip(y0, 0, self.height - 2)
        x1 = x0 + 1
        y1 = y0 + 1

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
