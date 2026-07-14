"""Multi-component Shan-Chen LBM engine (two immiscible fluids).

Implements the Shan-Chen pseudopotential model for two fluid species
with inter-component repulsion. Each component has its own distribution
function, density field, and pseudopotential force.

The interaction potential is:
  F_k = -g_kk · ψ(ρ_k) · Σ wᵢ · ψ(ρ_k(x+eᵢ)) · eᵢ        (intra)
        -g_kl · ψ(ρ_k) · Σ wᵢ · ψ(ρ_l(x+eᵢ)) · eᵢ        (inter)

Where k and l are the two components, g_kk controls cohesion within
component k, and g_kl controls repulsion between components.

A color field perturbation force sharpens the interface between
components (reduces spurious currents at the interface).

References:
- Shan & Chen (1993), "Lattice Boltzmann model for simulating flows
  with multiple phases and components"
- Grunau et al. (1993), "A lattice Boltzmann model for multiphase
  flow phenomena"
"""

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
def _compute_force_multicomponent_nb(
    rho1,
    rho2,
    obstacles,
    w,
    cx,
    cy,
    g11: float,
    g22: float,
    g12: float,
    g_adhesion: float,
    height: int,
    width: int,
    fx1,
    fy1,
    fx2,
    fy2,
):
    """Compute Shan-Chen forces for two components."""
    for y in prange(height):
        for x in range(width):
            if obstacles[y, x]:
                fx1[y, x] = 0.0
                fy1[y, x] = 0.0
                fx2[y, x] = 0.0
                fy2[y, x] = 0.0
                continue

            psi1_i = _psi(rho1[y, x])
            psi2_i = _psi(rho2[y, x])

            # Intra-component forces for component 1
            s11x = 0.0
            s11y = 0.0
            # Inter-component forces on component 1 from component 2
            s12x = 0.0
            s12y = 0.0
            # Intra-component forces for component 2
            s22x = 0.0
            s22y = 0.0
            # Inter-component forces on component 2 from component 1
            s21x = 0.0
            s21y = 0.0
            # Adhesion
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
                        psi1_j = _psi(rho1[sy, sx])
                        psi2_j = _psi(rho2[sy, sx])
                        # Intra: 1-1
                        s11x += w[i] * psi1_j * cx[i]
                        s11y += w[i] * psi1_j * cy[i]
                        # Inter: 1<-2
                        s12x += w[i] * psi2_j * cx[i]
                        s12y += w[i] * psi2_j * cy[i]
                        # Intra: 2-2
                        s22x += w[i] * psi2_j * cx[i]
                        s22y += w[i] * psi2_j * cy[i]
                        # Inter: 2<-1
                        s21x += w[i] * psi1_j * cx[i]
                        s21y += w[i] * psi1_j * cy[i]

            f1x = -psi1_i * (g11 * s11x + g12 * s12x + g_adhesion * afx)
            f1y = -psi1_i * (g11 * s11y + g12 * s12y + g_adhesion * afy)
            f2x = -psi2_i * (g22 * s22x + g12 * s21x + g_adhesion * afx)
            f2y = -psi2_i * (g22 * s22y + g12 * s21y + g_adhesion * afy)

            fx1[y, x] = max(min(f1x, _FORCE_CLIP), -_FORCE_CLIP)
            fy1[y, x] = max(min(f1y, _FORCE_CLIP), -_FORCE_CLIP)
            fx2[y, x] = max(min(f2x, _FORCE_CLIP), -_FORCE_CLIP)
            fy2[y, x] = max(min(f2y, _FORCE_CLIP), -_FORCE_CLIP)


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _compute_color_perturbation_nb(
    rho1,
    rho2,
    obstacles,
    w,
    cx,
    cy,
    sigma: float,
    height: int,
    width: int,
    fx1,
    fy1,
    fx2,
    fy2,
):
    """Color gradient perturbation force to sharpen the interface.

    The color field C = (rho1 - rho2) / (rho1 + rho2) is computed
    and a perturbation force is applied perpendicular to the interface
    to reduce spurious currents.
    """
    for y in prange(height):
        for x in range(width):
            if obstacles[y, x]:
                continue

            total = rho1[y, x] + rho2[y, x]
            if total < 1e-6:
                continue

            # Compute color gradient via finite differences
            # dC/dx and dC/dy
            dcx = 0.0
            dcy = 0.0

            # X direction
            x_in = x > 0 and x < width - 1
            x_free = x_in and not obstacles[y, x - 1] and not obstacles[y, x + 1]
            if x_free:
                t_l = rho1[y, x - 1] + rho2[y, x - 1]
                t_r = rho1[y, x + 1] + rho2[y, x + 1]
                if t_l > 1e-6 and t_r > 1e-6:
                    c_l = (rho1[y, x - 1] - rho2[y, x - 1]) / t_l
                    c_r = (rho1[y, x + 1] - rho2[y, x + 1]) / t_r
                    dcx = (c_r - c_l) * 0.5

            # Y direction
            y_in = y > 0 and y < height - 1
            y_free = y_in and not obstacles[y - 1, x] and not obstacles[y + 1, x]
            if y_free:
                t_b = rho1[y - 1, x] + rho2[y - 1, x]
                t_t = rho1[y + 1, x] + rho2[y + 1, x]
                if t_b > 1e-6 and t_t > 1e-6:
                    c_b = (rho1[y - 1, x] - rho2[y - 1, x]) / t_b
                    c_t = (rho1[y + 1, x] - rho2[y + 1, x]) / t_t
                    dcy = (c_t - c_b) * 0.5

            mag = math.sqrt(dcx * dcx + dcy * dcy)
            if mag < 1e-8:
                continue

            # Perturbation force: push component 1 toward its own color,
            # push component 2 the opposite way
            nx = dcx / mag
            ny = dcy / mag
            f_pert = sigma * mag

            fx1[y, x] = max(min(fx1[y, x] + f_pert * nx, _FORCE_CLIP), -_FORCE_CLIP)
            fy1[y, x] = max(min(fy1[y, x] + f_pert * ny, _FORCE_CLIP), -_FORCE_CLIP)
            fx2[y, x] = max(min(fx2[y, x] - f_pert * nx, _FORCE_CLIP), -_FORCE_CLIP)
            fy2[y, x] = max(min(fy2[y, x] - f_pert * ny, _FORCE_CLIP), -_FORCE_CLIP)


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _fused_step_component_nb(
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
    """Streaming + collision for a single component distribution."""
    tau = 1.0 / omega
    f_new = np.empty_like(f)

    for y in prange(height):
        for x in range(width):
            fi = np.empty(9, dtype=np.float32)
            is_obs = obstacles[y, x]

            for i in range(9):
                sx = x - cx[i]
                sy = y - cy[i]
                if 0 <= sx < width and 0 <= sy < height:
                    fi[i] = f[i, sy, sx]
                else:
                    fi[i] = f[opp[i], y, x]

            if is_obs:
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

            rho_safe = r if r > 1e-6 else 1e-6
            u_vel = mom_x / rho_safe
            v_vel = mom_y / rho_safe

            # Clip stored velocity to prevent instability
            vmag = math.sqrt(u_vel * u_vel + v_vel * v_vel)
            if vmag > _VEL_CLIP:
                u_vel *= _VEL_CLIP / vmag
                v_vel *= _VEL_CLIP / vmag

            if not is_obs:
                u_eq = u_vel + tau * fx[y, x] / rho_safe
                v_eq = v_vel + tau * fy[y, x] / rho_safe
            else:
                u_eq = 0.0
                v_eq = 0.0

            mag = math.sqrt(u_eq * u_eq + v_eq * v_eq)
            if mag > _VEL_CLIP:
                u_eq *= _VEL_CLIP / mag
                v_eq *= _VEL_CLIP / mag

            # Use rho_safe for equilibrium to prevent negative distributions
            r_eq = rho_safe
            u2 = u_eq * u_eq + v_eq * v_eq
            for i in range(9):
                cu = cx[i] * u_eq + cy[i] * v_eq
                feq = w[i] * r_eq * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f_new[i, y, x] = fi[i] * (1.0 - omega) + feq * omega

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel

    f[:] = f_new


class LBM2DMultiComponent(SimEngine, SmokeMixin):
    """D2Q9 Shan-Chen multiphase simulation with two immiscible fluid components.

    Each component (e.g. oil and water) has its own distribution function
    and density field. Inter-component repulsion (g12) drives phase separation.

    Attributes:
        g11: Intra-component cohesion for component 1 (default -5.0)
        g22: Intra-component cohesion for component 2 (default -5.0)
        g12: Inter-component repulsion (default +5.0, positive = repel)
        sigma: Color gradient perturbation strength (default 0.05)
        g_adhesion: Wall adhesion strength (default -5.0)
    """

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        viscosity: float = 0.02,
        g11: float = -5.0,
        g22: float = -5.0,
        g12: float = 5.0,
        sigma: float = 0.05,
        g_adhesion: float = -5.0,
        droplet_radius: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.0

        self.g11 = g11
        self.g22 = g22
        self.g12 = g12
        self.sigma = sigma
        self.g_adhesion = g_adhesion
        self.droplet_radius = droplet_radius

        self.lattice = LATTICE_2D
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )

        # Component 1 distribution and density
        self.f1 = np.zeros((9, height, width), dtype=np.float32)
        self.rho1 = np.ones((height, width), dtype=np.float32)

        # Component 2 distribution and density
        self.f2 = np.zeros((9, height, width), dtype=np.float32)
        self.rho2 = np.ones((height, width), dtype=np.float32)

        # Total velocity (for display and smoke advection)
        self.u = np.zeros((height, width), dtype=np.float32)
        self.v = np.zeros((height, width), dtype=np.float32)

        self.obstacles = np.zeros((height, width), dtype=np.bool_)

        # Force arrays
        self.fx1 = np.empty((height, width), dtype=np.float32)
        self.fy1 = np.empty((height, width), dtype=np.float32)
        self.fx2 = np.empty((height, width), dtype=np.float32)
        self.fy2 = np.empty((height, width), dtype=np.float32)

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
        self.rho1[:] = rho
        self.rho2[:] = rho
        self.u[:] = u
        self.v[:] = v

        radius = self.droplet_radius
        if radius is None:
            radius = min(self.width, self.height) // 6
        cx = self.width // 2
        cy = self.height // 2
        y_grid, x_grid = np.ogrid[: self.height, : self.width]
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2

        # Component 1: high density droplet
        self.rho1[mask] = 2.0
        # Component 2: high density background (inverted)
        self.rho2[~mask] = 2.0

        self.f1 = self.lattice.equilibrium(self.rho1, self.u, self.v)
        self.f2 = self.lattice.equilibrium(self.rho2, self.u, self.v)
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

    def step(self) -> None:
        # Compute Shan-Chen forces for both components
        _compute_force_multicomponent_nb(
            self.rho1,
            self.rho2,
            self.obstacles,
            self.lattice.w,
            self.lattice.cx,
            self.lattice.cy,
            self.g11,
            self.g22,
            self.g12,
            self.g_adhesion,
            self.height,
            self.width,
            self.fx1,
            self.fy1,
            self.fx2,
            self.fy2,
        )

        # Color gradient perturbation to sharpen interface
        if self.sigma > 0:
            _compute_color_perturbation_nb(
                self.rho1,
                self.rho2,
                self.obstacles,
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                self.sigma,
                self.height,
                self.width,
                self.fx1,
                self.fy1,
                self.fx2,
                self.fy2,
            )

        # Collision + streaming for component 1
        _fused_step_component_nb(
            self.f1,
            self.rho1,
            self.u,
            self.v,
            self.fx1,
            self.fy1,
            self.obstacles,
            self.lattice.opp,
            self.lattice.w,
            self.lattice.cx,
            self.lattice.cy,
            self.omega,
            self.height,
            self.width,
        )

        # Save component 1 velocity, compute component 2
        u1 = self.u.copy()
        v1 = self.v.copy()

        # Collision + streaming for component 2
        _fused_step_component_nb(
            self.f2,
            self.rho2,
            self.u,
            self.v,
            self.fx2,
            self.fy2,
            self.obstacles,
            self.lattice.opp,
            self.lattice.w,
            self.lattice.cx,
            self.lattice.cy,
            self.omega,
            self.height,
            self.width,
        )

        # Total velocity = average of both components
        self.u = 0.5 * (u1 + self.u)
        self.v = 0.5 * (v1 + self.v)

        # Safety clamp to prevent NaN propagation into smoke advection
        np.clip(self.u, -_VEL_CLIP, _VEL_CLIP, out=self.u)
        np.clip(self.v, -_VEL_CLIP, _VEL_CLIP, out=self.v)
        np.nan_to_num(self.u, copy=False, nan=0.0)
        np.nan_to_num(self.v, copy=False, nan=0.0)
        np.clip(self.rho1, 0.0, None, out=self.rho1)
        np.clip(self.rho2, 0.0, None, out=self.rho2)

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
        return (self.rho1 + self.rho2).copy()

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
        return np.concatenate([self.f1, self.f2], axis=0)

    def get_pressure(self) -> np.ndarray:
        return self.rho1 + self.rho2 - 1.0

    def get_field_names(self) -> list[str]:
        return [
            "smoke",
            "speed",
            "vorticity",
            "pressure",
            "density",
            "phase",
            "component1",
            "component2",
            "color",
        ]

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
            p = (self.rho1 + self.rho2 - 1.0).astype(np.float32)
            cur_max = max(float(np.max(np.abs(p))), 0.001)
            self._ema_pres_max = (1 - a) * self._ema_pres_max + a * cur_max
            return np.clip(p / self._ema_pres_max * 0.5 + 0.5, 0, 1).astype(np.float32)
        if name == "density":
            total = (self.rho1 + self.rho2).astype(np.float32)
            lo, hi = float(np.min(total)), float(np.max(total))
            if hi - lo < 0.001:
                return np.full_like(total, 0.5, dtype=np.float32)
            return np.clip((total - lo) / (hi - lo), 0, 1).astype(np.float32)
        if name == "phase":
            field = 1.0 / (1.0 + np.exp(-15 * (self.rho1 - self.rho2)))
            return np.clip(field, 0, 1).astype(np.float32)
        if name == "component1":
            lo = float(np.min(self.rho1))
            hi = float(np.max(self.rho1))
            if hi - lo < 0.001:
                return np.full_like(self.rho1, 0.5, dtype=np.float32)
            return np.clip((self.rho1 - lo) / (hi - lo), 0, 1).astype(np.float32)
        if name == "component2":
            lo = float(np.min(self.rho2))
            hi = float(np.max(self.rho2))
            if hi - lo < 0.001:
                return np.full_like(self.rho2, 0.5, dtype=np.float32)
            return np.clip((self.rho2 - lo) / (hi - lo), 0, 1).astype(np.float32)
        if name == "color":
            total = self.rho1 + self.rho2
            safe_total = np.where(total > 1e-6, total, 1.0)
            color = (self.rho1 - self.rho2) / safe_total
            return np.clip(color * 0.5 + 0.5, 0, 1).astype(np.float32)
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
