from __future__ import annotations

import numpy as np
from numba import njit, prange

from engines.base import SimEngine
from engines.boundary_conditions import (
    apply_moving_wall_bottom,
    apply_moving_wall_top,
    apply_zou_he_left,
    apply_zou_he_right_pressure,
)
from engines.collision import BGKCollision, CollisionOperator
from engines.forcing import _bgk_collide_guo_const_nb
from engines.lbm_common import LATTICE_2D
from engines.particle_tracer import ParticleTracer
from engines.smoke_mixin import SmokeMixin
from engines.thermal_mixin import ThermalMixin

# Domain modes for Phase A validation setups.
DOMAIN_CHANNEL = "channel"  # Zou-He / EQ inlet + walls + outflow
DOMAIN_CAVITY = "cavity"  # closed; MovingWall lid
DOMAIN_PERIODIC = "periodic"  # fully periodic (TGV)
DOMAIN_FORCE = "force"  # periodic-x + walls + Guo body force (Poiseuille)


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _fused_step_nb(
    f,
    f_new,
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
    """Fused pull-stream + obstacle BB + EQ inflow + wall BB + BGK."""
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


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _stream_nb(f, f_new, cx, cy, height, width):
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


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _bounce_back_nb(f, mask, opp, height, width):
    """Full-way bounce-back on solid mask cells."""
    for i in range(9):
        opp_i = opp[i]
        if i < opp_i:
            for y in prange(height):
                for x in range(width):
                    if mask[y, x]:
                        tmp = f[i, y, x]
                        f[i, y, x] = f[opp_i, y, x]
                        f[opp_i, y, x] = tmp


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _halfway_bb_2d_nb(f, mask, cx, cy, opp, height, width):
    """Halfway bounce-back after streaming (fluid nodes only).

    For each fluid→solid link along c_i, the missing population at the
    fluid node (incoming from the solid) is the opposite direction opp[i].
    Set f[opp] = f[i] (stationary wall).
    """
    for y in prange(height):
        for x in range(width):
            if mask[y, x]:
                continue
            for i in range(1, 9):
                nx = x + cx[i]
                ny = y + cy[i]
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if mask[ny, nx]:
                    oi = opp[i]
                    f[oi, y, x] = f[i, y, x]


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _update_macros_nb(f, rho, u, v, cx, cy, height, width):
    for y in prange(height):
        for x in range(width):
            r = 0.0
            mu = 0.0
            mv = 0.0
            for i in range(9):
                fiv = f[i, y, x]
                r += fiv
                mu += fiv * cx[i]
                mv += fiv * cy[i]
            rho_safe = r if r > 0 else 1.0
            rho[y, x] = r
            u[y, x] = mu / rho_safe
            v[y, x] = mv / rho_safe


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _wall_bb_nb(f, opp, height, width):
    """Stationary bounce-back on top and bottom rows."""
    for i in range(9):
        opp_i = opp[i]
        if i < opp_i:
            for x in prange(width):
                tmp = f[i, 0, x]
                f[i, 0, x] = f[opp_i, 0, x]
                f[opp_i, 0, x] = tmp
                tmp = f[i, height - 1, x]
                f[i, height - 1, x] = f[opp_i, height - 1, x]
                f[opp_i, height - 1, x] = tmp


class LBM2D(SimEngine, SmokeMixin, ThermalMixin):
    """D2Q9 Lattice Boltzmann fluid simulation with Numba-accelerated solver."""

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        viscosity: float = 0.02,
        collision: CollisionOperator | None = None,
        domain_mode: str = DOMAIN_CHANNEL,
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.05  # low-Mach default (Ma ≈ 0.09)
        self.domain_mode = domain_mode
        self.lid_velocity = 0.1
        self.body_force = (0.0, 0.0)
        self.rho_outlet = 1.0
        self.use_zou_he = False  # EQ inflow default (stable); Zou-He opt-in
        self.use_fused = True  # BGK + channel fast path
        self.obstacle_bc = "halfway"  # "halfway" | "fullway"
        self.ibm: object | None = None  # optional RigidIBM

        self.lattice = LATTICE_2D
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )
        self.collision_op = collision or BGKCollision()

        self.f = np.zeros((9, height, width), dtype=np.float32)
        self._f_swap = np.zeros((9, height, width), dtype=np.float32)

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

        self._ema_smoke_max = 0.001
        self._ema_speed_max = 0.001
        self._ema_vort_max = 0.001
        self._ema_pres_max = 0.001
        self._ema_alpha = 0.05

        # Thermal opt-in (Hidden from product UI until Phase A5 complete)
        self.thermal_enabled = False

        self.initialize(rho=1.0, u=0.0, v=0.0)
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        self.step(physics_only=True)
        self.initialize(rho=1.0, u=0.0, v=0.0)

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
        self, rho: float = 1.0, u: float = 0.0, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.f[:] = self.lattice.equilibrium(self.rho, self.u, self.v)
        self._f_swap[:] = self.f
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

    def _can_use_fused(self) -> bool:
        return (
            self.use_fused
            and isinstance(self.collision_op, BGKCollision)
            and self.domain_mode == DOMAIN_CHANNEL
            and self.body_force == (0.0, 0.0)
            and not self.use_zou_he
        )

    def step(self, physics_only: bool = False) -> None:
        if self._can_use_fused():
            _fused_step_nb(
                self.f,
                self._f_swap,
                self.rho,
                self.u,
                self.v,
                self.obstacles,
                self.lattice.opp,
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                self.omega,
                self.u_inflow,
                self.height,
                self.width,
            )
            self.f, self._f_swap = self._f_swap, self.f
            # Open right column after fused step
            self.f[:, :, -1] = self.f[:, :, -2]
        else:
            # Collide → stream → BC (standard Zou-He / moving-wall order)
            self.collision()
            self.streaming()
            self.apply_boundary_conditions()
            _update_macros_nb(
                self.f,
                self.rho,
                self.u,
                self.v,
                self.lattice.cx,
                self.lattice.cy,
                self.height,
                self.width,
            )

        # Thermal physics: collide, stream temperature, then apply buoyancy
        if hasattr(self, "thermal_enabled") and self.thermal_enabled:
            self.collision_temperature()
            self.streaming_temperature()
            self.apply_buoyancy()

        if not physics_only:
            self.apply_emitters()
            self.advect_smoke()
            self.diffuse_smoke()
            self.smoke[self.obstacles] = 0.0
            self.decay_smoke()
            vel = self.get_velocity_view()
            self._particle_tracer.step(vel)

    def apply_boundary_conditions(self) -> None:
        mode = self.domain_mode
        if mode == DOMAIN_PERIODIC:
            return

        self.apply_obstacles()

        if mode == DOMAIN_CAVITY:
            apply_moving_wall_bottom(self.f, 0.0, self.lattice)
            self._bounce_side_walls()
            apply_moving_wall_top(self.f, self.lid_velocity, self.lattice)
            return

        if mode == DOMAIN_FORCE:
            self.apply_walls()
            return

        # CHANNEL: walls, then inlet / outlet
        self.apply_walls()
        if self.use_zou_he:
            apply_zou_he_left(self.f, self.u_inflow, 0.0, self.lattice)
            apply_zou_he_right_pressure(self.f, self.rho_outlet, self.lattice)
        else:
            self.apply_inflow()
            self.f[:, :, -1] = self.f[:, :, -2]

    def _bounce_side_walls(self) -> None:
        """Full-way bounce-back on left and right columns (cavity sides)."""
        opp = self.lattice.opp
        for i in range(9):
            oi = int(opp[i])
            if i < oi:
                for col in (0, self.width - 1):
                    tmp = self.f[i, :, col].copy()
                    self.f[i, :, col] = self.f[oi, :, col]
                    self.f[oi, :, col] = tmp

    def run(self, steps: int, physics_only: bool = False) -> None:
        for _ in range(steps):
            self.step(physics_only=physics_only)

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
        """Lattice pressure p = ρ / 3 (c_s² = 1/3)."""
        return self.rho / 3.0

    def get_pressure_gauge(self) -> np.ndarray:
        """Gauge pressure p − 1/3 (relative to ρ₀ = 1)."""
        return self.rho / 3.0 - (1.0 / 3.0)

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
            mx = max(self.u_inflow * 1.5, self.lid_velocity * 1.5, self._ema_speed_max)
            mx = max(mx, 0.001)
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
            p = self.get_pressure_gauge().astype(np.float32)
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
        _stream_nb(
            self.f,
            self._f_swap,
            self.lattice.cx,
            self.lattice.cy,
            self.height,
            self.width,
        )
        self.f, self._f_swap = self._f_swap, self.f

    def collision(self) -> None:
        fx, fy = self.body_force
        if self.ibm is not None:
            ibx, iby = self.ibm.compute_force(self.u, self.v)
            # Combine uniform body force + IBM force density via Guo
            from engines.forcing import _bgk_collide_guo_nb

            fx_field = ibx.astype(np.float32, copy=True)
            fy_field = iby.astype(np.float32, copy=True)
            if fx != 0.0:
                fx_field += np.float32(fx)
            if fy != 0.0:
                fy_field += np.float32(fy)
            _bgk_collide_guo_nb(
                self.f,
                self.rho,
                self.u,
                self.v,
                fx_field,
                fy_field,
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                self.omega,
                self.height,
                self.width,
            )
            return
        if (fx != 0.0 or fy != 0.0) and isinstance(self.collision_op, BGKCollision):
            _bgk_collide_guo_const_nb(
                self.f,
                self.rho,
                self.u,
                self.v,
                float(fx),
                float(fy),
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                self.omega,
                self.height,
                self.width,
            )
            return
        self.collision_op.collide(
            self.f, self.rho, self.u, self.v, self.lattice, self.viscosity
        )

    def apply_obstacles(self) -> None:
        if self.obstacle_bc == "halfway":
            _halfway_bb_2d_nb(
                self.f,
                self.obstacles,
                self.lattice.cx,
                self.lattice.cy,
                self.lattice.opp,
                self.height,
                self.width,
            )
        else:
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
        if self.use_zou_he and self.domain_mode == DOMAIN_CHANNEL:
            apply_zou_he_right_pressure(self.f, self.rho_outlet, self.lattice)
        else:
            self.f[:, :, -1] = self.f[:, :, -2]

    def apply_walls(self) -> None:
        _wall_bb_nb(self.f, self.lattice.opp, self.height, self.width)
