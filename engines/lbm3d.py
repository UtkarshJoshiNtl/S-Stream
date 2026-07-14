from __future__ import annotations

import numpy as np
from numba import njit, prange

from engines.base import SimEngine
from engines.collision import BGKCollision, CollisionOperator
from engines.lbm_common import LATTICE_3D_Q19, Lattice3D
from engines.particle_tracer import ParticleTracer
from engines.smoke_mixin import SmokeMixin
from engines.thermal_mixin import ThermalMixin


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _stream_3d_nb(f, cx, cy, cz, depth, height, width):
    """Stream distribution functions in 3D with periodic boundaries."""
    f_new = np.empty_like(f)
    n_vel = f.shape[0]
    for i in range(n_vel):
        for z in prange(depth):
            for y in range(height):
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
                    sz = z - cz[i]
                    if sz < 0:
                        sz += depth
                    elif sz >= depth:
                        sz -= depth
                    f_new[i, z, y, x] = f[i, sz, sy, sx]
    f[:] = f_new


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _bounce_back_3d_nb(f, mask, opp, depth, height, width):
    """Apply bounce-back boundary conditions in 3D."""
    n_vel = f.shape[0]
    for i in range(n_vel):
        opp_i = opp[i]
        if i < opp_i:
            for z in prange(depth):
                for y in range(height):
                    for x in range(width):
                        if mask[z, y, x]:
                            tmp = f[i, z, y, x]
                            f[i, z, y, x] = f[opp_i, z, y, x]
                            f[opp_i, z, y, x] = tmp


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _collide_3d_nb(
    f, rho, u, v, w_vel, w, cx, cy, cz, omega, depth, height, width
):
    """BGK collision kernel for 3D."""
    n_vel = f.shape[0]
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                r = 0.0
                u_val = 0.0
                v_val = 0.0
                w_val = 0.0
                for i in range(n_vel):
                    fiv = f[i, z, y, x]
                    r += fiv
                    u_val += fiv * cx[i]
                    v_val += fiv * cy[i]
                    w_val += fiv * cz[i]
                rho_safe = r if r > 0 else 1.0
                u_val /= rho_safe
                v_val /= rho_safe
                w_val /= rho_safe
                u2 = u_val * u_val + v_val * v_val + w_val * w_val
                for i in range(n_vel):
                    cu = cx[i] * u_val + cy[i] * v_val + cz[i] * w_val
                    feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                    f[i, z, y, x] = f[i, z, y, x] * (1.0 - omega) + feq * omega
                rho[z, y, x] = r
                u[z, y, x] = u_val
                v[z, y, x] = v_val
                w_vel[z, y, x] = w_val


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _advect_smoke_3d_nb(smoke, u, v, w_vel, x_coords, y_coords, z_coords, dt):
    """Advect smoke using trilinear interpolation in 3D."""
    depth, height, width = smoke.shape
    smoke_new = np.empty_like(smoke)
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                # Backtracing
                px = x - u[z, y, x] * dt
                py = y - v[z, y, x] * dt
                pz = z - w_vel[z, y, x] * dt

                # Clamp to grid
                px = max(0.0, min(float(width - 1), px))
                py = max(0.0, min(float(height - 1), py))
                pz = max(0.0, min(float(depth - 1), pz))

                # Trilinear interpolation
                x0 = int(px)
                y0 = int(py)
                z0 = int(pz)
                x1 = min(x0 + 1, width - 1)
                y1 = min(y0 + 1, height - 1)
                z1 = min(z0 + 1, depth - 1)

                fx = px - x0
                fy = py - py
                fz = pz - z0

                smoke_new[z, y, x] = (
                    smoke[z0, y0, x0] * (1 - fx) * (1 - fy) * (1 - fz)
                    + smoke[z0, y0, x1] * fx * (1 - fy) * (1 - fz)
                    + smoke[z0, y1, x0] * (1 - fx) * fy * (1 - fz)
                    + smoke[z0, y1, x1] * fx * fy * (1 - fz)
                    + smoke[z1, y0, x0] * (1 - fx) * (1 - fy) * fz
                    + smoke[z1, y0, x1] * fx * (1 - fy) * fz
                    + smoke[z1, y1, x0] * (1 - fx) * fy * fz
                    + smoke[z1, y1, x1] * fx * fy * fz
                )
    smoke[:] = smoke_new


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _diffuse_smoke_3d_nb(smoke, out, diffusion, depth, height, width):
    """Diffuse smoke using 3D Laplacian."""
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                center = smoke[z, y, x]
                neighbors = 0.0
                count = 0.0
                if x > 0:
                    neighbors += smoke[z, y, x - 1]
                    count += 1
                if x < width - 1:
                    neighbors += smoke[z, y, x + 1]
                    count += 1
                if y > 0:
                    neighbors += smoke[z, y - 1, x]
                    count += 1
                if y < height - 1:
                    neighbors += smoke[z, y + 1, x]
                    count += 1
                if z > 0:
                    neighbors += smoke[z - 1, y, x]
                    count += 1
                if z < depth - 1:
                    neighbors += smoke[z + 1, y, x]
                    count += 1
                laplacian = neighbors - count * center
                out[z, y, x] = center + diffusion * laplacian


class LBM3D(SimEngine, SmokeMixin, ThermalMixin):
    """D3Q19 Lattice Boltzmann fluid simulation with Numba-accelerated solver.

    3D extension of LBM2D with the same pluggable collision operator interface.
    Memory: f array is (19, D, H, W) float32 — a 128^3 grid uses ~1GB.
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        depth: int = 64,
        viscosity: float = 0.02,
        collision: CollisionOperator | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.viscosity = viscosity
        self.u_inflow = 0.15

        self.lattice = LATTICE_3D_Q19
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )
        self.collision_op = collision or BGKCollision()

        # Distribution functions
        self.f = np.zeros((19, depth, height, width), dtype=np.float32)

        # Macroscopic fields
        self.rho = np.ones((depth, height, width), dtype=np.float32)
        self.u = np.zeros((depth, height, width), dtype=np.float32)
        self.v = np.zeros((depth, height, width), dtype=np.float32)
        self.w_vel = np.zeros((depth, height, width), dtype=np.float32)

        # Obstacles
        self.obstacles = np.zeros((depth, height, width), dtype=np.bool_)

        # Smoke (passive scalar)
        self.smoke = np.zeros((depth, height, width), dtype=np.float32)
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, int, float]] = []

        # Work arrays for smoke operations
        self._lap_buffer = np.empty_like(self.smoke)
        self._x_coords = np.arange(width, dtype=np.float32)
        self._y_coords = np.arange(height, dtype=np.float32)
        self._z_coords = np.arange(depth, dtype=np.float32)
        self.xp = np

        self._particle_tracer = ParticleTracer(width, height, depth, trail_length=20)
        self._vel_buf = np.empty((depth, height, width, 3), dtype=np.float32)

        # EMA normalization caches
        self._ema_smoke_max = 0.001
        self._ema_speed_max = 0.001
        self._ema_vort_max = 0.001
        self._ema_pres_max = 0.001
        self._ema_alpha = 0.05

        self.initialize(rho=1.0, u=0.1, v=0.0, w=0.0)
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        self.step()
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
        return self.lattice.omega_from_viscosity(self.viscosity)

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
        np.stack([self.u, self.v, self.w_vel], axis=3, out=self._vel_buf)
        return self._vel_buf.copy()

    def get_velocity_view(self) -> np.ndarray:
        np.stack([self.u, self.v, self.w_vel], axis=3, out=self._vel_buf)
        return self._vel_buf

    def get_velocity_at(self, x: int, y: int) -> tuple[float, float]:
        return float(self.u[0, y, x]), float(self.v[0, y, x])

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
        vel3d = self.get_velocity_view()
        u, v, w = vel3d[..., 0], vel3d[..., 1], vel3d[..., 2]
        a = self._ema_alpha
        if name == "smoke":
            cur_max = max(float(np.max(self.smoke)), 0.001)
            self._ema_smoke_max = (1 - a) * self._ema_smoke_max + a * cur_max
            return np.clip(self.smoke / self._ema_smoke_max, 0, 1).astype(np.float32)
        if name == "speed":
            speed = np.sqrt(
                u.astype(np.float32) ** 2
                + v.astype(np.float32) ** 2
                + w.astype(np.float32) ** 2
            )
            cur_max = max(float(np.max(speed)), 0.001)
            self._ema_speed_max = (1 - a) * self._ema_speed_max + a * cur_max
            mx = max(self.u_inflow * 1.5, self._ema_speed_max, 0.001)
            return np.clip(speed / mx, 0, 1).astype(np.float32)
        if name == "vorticity":
            dvdx = np.zeros_like(u, dtype=np.float32)
            dudy = np.zeros_like(u, dtype=np.float32)
            dvdx[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) * 0.5
            dudy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) * 0.5
            vort = dvdx - dudy
            cur_max = max(float(np.max(np.abs(vort))), 0.001)
            self._ema_vort_max = (1 - a) * self._ema_vort_max + a * cur_max
            return np.clip(
                vort / self._ema_vort_max * 0.5 + 0.5, 0, 1
            ).astype(np.float32)
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

    def streaming(self) -> None:
        _stream_3d_nb(
            self.f, self.lattice.cx, self.lattice.cy, self.lattice.cz,
            self.depth, self.height, self.width
        )

    def collision(self) -> None:
        self.collision_op.collide(
            self.f, self.rho, self.u, self.v, self.lattice, self.viscosity,
            w_vel=self.w_vel
        )

    def apply_obstacles(self) -> None:
        _bounce_back_3d_nb(
            self.f, self.obstacles, self.lattice.opp,
            self.depth, self.height, self.width
        )

    def apply_inflow(self) -> None:
        u_in = self.u_inflow
        u2 = u_in * u_in
        cu = self.lattice.cx * u_in
        feq = self.lattice.w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
        self.f[:, :, :, 0] = feq[:, np.newaxis, np.newaxis]

    def apply_outflow(self) -> None:
        self.f[:, :, :, -1] = self.f[:, :, :, -2]

    def apply_walls(self) -> None:
        n_vel = self.lattice.n_velocities
        for i in range(n_vel):
            opp_i = self.lattice.opp[i]
            if i < opp_i:
                # Top wall (z=0)
                tmp = self.f[i, 0, :, :].copy()
                self.f[i, 0, :, :] = self.f[opp_i, 0, :, :]
                self.f[opp_i, 0, :, :] = tmp
                # Bottom wall (z=-1)
                tmp = self.f[i, -1, :, :].copy()
                self.f[i, -1, :, :] = self.f[opp_i, -1, :, :]
                self.f[opp_i, -1, :, :] = tmp
                # Front wall (y=0)
                tmp = self.f[i, :, 0, :].copy()
                self.f[i, :, 0, :] = self.f[opp_i, :, 0, :]
                self.f[opp_i, :, 0, :] = tmp
                # Back wall (y=-1)
                tmp = self.f[i, :, -1, :].copy()
                self.f[i, :, -1, :] = self.f[opp_i, :, -1, :]
                self.f[opp_i, :, -1, :] = tmp
