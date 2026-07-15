from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from engines.base import SimEngine
from engines.lbm_common import LATTICE_2D
from engines.particle_tracer import ParticleTracer
from engines.smoke_mixin import SmokeMixin

_FORCE_CLIP = 0.3
_VEL_CLIP = 0.3


@njit(cache=True, fastmath=True, boundscheck=False)
def _psi(r: float) -> float:
    return 1.0 - math.exp(-r)


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _compute_force_nb(
    rho,
    obstacles,
    w,
    cx,
    cy,
    g: float,
    g_adhesion: float,
    height: int,
    width: int,
    fx_out,
    fy_out,
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


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _fused_step_liquid_nb(
    f,
    rho,
    u,
    v,
    fx,
    fy,
    obstacles,
    opp,
    w,
    cx,
    cy,
    omega: float,
    height: int,
    width: int,
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
                feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f_new[i, y, x] = fi[i] * (1.0 - omega) + feq * omega

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel

    f[:] = f_new


class LBM2DLiquid(SimEngine, SmokeMixin):
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
        self.xp = np

        self._particle_tracer = ParticleTracer(width, height, trail_length=20)
        self._vel_buf = np.empty((height, width, 2), dtype=np.float32)

        # EMA normalization caches
        self._ema_smoke_max = 0.001
        self._ema_speed_max = 0.001
        self._ema_vort_max = 0.001
        self._ema_pres_max = 0.001
        self._ema_alpha = 0.05

        self.initialize()
        self._warmup_jit()

    def _warmup_jit(self) -> None:
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
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2
        self.rho[mask] = 2.0

        self.f = self.lattice.equilibrium(self.rho, self.u, self.v)
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

    def step(self) -> None:
        _compute_force_nb(
            self.rho,
            self.obstacles,
            self.lattice.w,
            self.lattice.cx,
            self.lattice.cy,
            self.g,
            self.g_adhesion,
            self.height,
            self.width,
            self.fx,
            self.fy,
        )
        _fused_step_liquid_nb(
            self.f,
            self.rho,
            self.u,
            self.v,
            self.fx,
            self.fy,
            self.obstacles,
            self.lattice.opp,
            self.lattice.w,
            self.lattice.cx,
            self.lattice.cy,
            self.omega,
            self.height,
            self.width,
        )
        self.apply_emitters()
        self.advect_smoke()
        self.diffuse_smoke()
        self.smoke[self.obstacles] = 0.0
        self.decay_smoke()
        vel = self.get_velocity_view()
        self._particle_tracer.step(vel)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def get_density(self) -> np.ndarray:
        return self.rho.copy()

    def get_velocity(self) -> np.ndarray:
        np.stack([self.u, self.v], axis=2, out=self._vel_buf)
        return self._vel_buf.copy()

    def get_velocity_view(self) -> np.ndarray:
        np.stack([self.u, self.v], axis=2, out=self._vel_buf)
        return self._vel_buf

    def get_velocity_at(self, x: int, y: int) -> tuple[float, float]:
        return float(self.u[y, x]), float(self.v[y, x])

    def get_smoke(self) -> np.ndarray:
        return self.smoke.copy()

    def get_obstacles(self) -> np.ndarray:
        return self.obstacles.copy()

    def get_obstacles_mut(self) -> np.ndarray:
        return self.obstacles

    def get_f(self) -> np.ndarray:
        return self.f

    def get_pressure(self) -> np.ndarray:
        return self.rho / 3.0

    def get_field_names(self) -> list[str]:
        return ["smoke", "speed", "vorticity", "pressure", "density", "phase"]

    def get_field(self, name: str) -> np.ndarray:
        a = self._ema_alpha
        if name == "smoke":
            cur_max = max(float(np.max(self.smoke)), 0.001)
            self._ema_smoke_max = (1 - a) * self._ema_smoke_max + a * cur_max
            return np.clip(self.smoke / self._ema_smoke_max, 0, 1).astype(np.float32)
        if name == "speed":
            speed = np.sqrt(
                self.u.astype(np.float32) ** 2 + self.v.astype(np.float32) ** 2
            )
            cur_max = max(float(np.max(speed)), 0.001)
            self._ema_speed_max = (1 - a) * self._ema_speed_max + a * cur_max
            return np.clip(speed / self._ema_speed_max, 0, 1).astype(np.float32)
        if name == "vorticity":
            dvdx = np.zeros_like(self.u, dtype=np.float32)
            dudy = np.zeros_like(self.u, dtype=np.float32)
            dvdx[:, 1:-1] = (self.v[:, 2:] - self.v[:, :-2]) * 0.5
            dudy[1:-1, :] = (self.u[2:, :] - self.u[:-2, :]) * 0.5
            vort = dvdx - dudy
            cur_max = max(float(np.max(np.abs(vort))), 0.001)
            self._ema_vort_max = (1 - a) * self._ema_vort_max + a * cur_max
            return np.clip(vort / self._ema_vort_max * 0.5 + 0.5, 0, 1).astype(
                np.float32
            )
        if name == "pressure":
            p = (self.rho - 1.0).astype(np.float32)
            cur_max = max(float(np.max(np.abs(p))), 0.001)
            self._ema_pres_max = (1 - a) * self._ema_pres_max + a * cur_max
            return np.clip(p / self._ema_pres_max * 0.5 + 0.5, 0, 1).astype(np.float32)
        if name == "density":
            lo, hi = float(np.min(self.rho)), float(np.max(self.rho))
            if hi - lo < 0.001:
                return np.full_like(self.rho, 0.5, dtype=np.float32)
            return np.clip((self.rho - lo) / (hi - lo), 0, 1).astype(np.float32)
        if name == "phase":
            field = 1.0 / (1.0 + np.exp(-15 * (self.rho - 0.5)))
            return np.clip(field, 0, 1).astype(np.float32)
        raise ValueError(
            f"Unknown field: {name!r}. Available: {self.get_field_names()}"
        )

    def get_emitter_count(self) -> int:
        return len(self.emitters)

    def get_particle_tracer(self) -> ParticleTracer:
        return self._particle_tracer

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
