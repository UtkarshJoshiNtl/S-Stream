from __future__ import annotations

import numpy as np
from numba import njit, prange

from engines.base import SimEngine
from engines.collision import BGKCollision, CollisionOperator
from engines.lbm_common import LATTICE_2D
from engines.particle_tracer import ParticleTracer
from engines.smoke_mixin import SmokeMixin
from engines.thermal_mixin import ThermalMixin


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _fused_step_nb(
    f,
    rho,
    u,
    v,
    obstacles,
    opp,
    w,
    cx,
    cy,
    omega,
    u_inflow,
    height,
    width,
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


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
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


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
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


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
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


class LBM2D(SimEngine, SmokeMixin, ThermalMixin):
    """D2Q9 Lattice Boltzmann fluid simulation with Numba-accelerated solver."""

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        viscosity: float = 0.02,
        collision: CollisionOperator | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.15

        self.lattice = LATTICE_2D
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )
        self.collision_op = collision or BGKCollision()

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
        self.xp = np

        self._particle_tracer = ParticleTracer(width, height, trail_length=20)
        self._vel_buf = np.empty((height, width, 2), dtype=np.float32)

        # EMA normalization caches (replaces per-frame percentile sort)
        self._ema_smoke_max = 0.001
        self._ema_speed_max = 0.001
        self._ema_vort_max = 0.001
        self._ema_pres_max = 0.001
        self._ema_alpha = 0.05  # smoothing factor

        self.initialize(rho=1.0, u=0.1, v=0.0)
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        self.step()
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
        self.streaming()
        self.apply_boundary_conditions()
        self.collision()
        self.apply_outflow()
        self.apply_emitters()
        self.advect_smoke()
        self.diffuse_smoke()
        self.smoke[self.obstacles] = 0.0
        self.decay_smoke()
        vel = self.get_velocity_view()
        self._particle_tracer.step(vel)

    def apply_boundary_conditions(self) -> None:
        self.apply_obstacles()
        self.apply_inflow()
        self.apply_walls()

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
        return self.rho - 1.0

    def get_field_names(self) -> list[str]:
        names = ["smoke", "speed", "vorticity", "pressure", "density"]
        if getattr(self, "thermal_enabled", False):
            names.append("temperature")
        return names

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
            mx = max(self.u_inflow * 1.5, self._ema_speed_max, 0.001)
            return np.clip(speed / mx, 0, 1).astype(np.float32)
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
        if name == "temperature":
            if getattr(self, "thermal_enabled", False):
                T = self.temperature.copy()
                lo, hi = float(np.min(T)), float(np.max(T))
                if hi - lo < 0.001:
                    return np.full_like(T, 0.5, dtype=np.float32)
                return np.clip((T - lo) / (hi - lo), 0, 1).astype(np.float32)
            return np.full(self.grid_shape, 0.5, dtype=np.float32)
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

    def streaming(self) -> None:
        _stream_nb(self.f, self.lattice.cx, self.lattice.cy, self.height, self.width)

    def collision(self) -> None:
        self.collision_op.collide(
            self.f, self.rho, self.u, self.v, self.lattice, self.viscosity
        )

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
