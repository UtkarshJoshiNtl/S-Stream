"""Thermal LBM mixin for temperature field simulation.

Adds a temperature distribution function that is streamed and collided
alongside the velocity distribution, with Boussinesq buoyancy coupling.

Reference: Lettuce thermal LBM (MIT license, can reference directly).
Boussinesq approximation: F_buoyancy = -beta * (T - T_0) * g_hat

Reference: Batchelor (1954), "Heat transfer by free convection across a
closed cell or a heated vertical plate."
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _stream_temperature_2d_nb(
    f_T: np.ndarray,
    f_T_new: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    height: int,
    width: int,
) -> None:
    """Stream temperature distribution in 2D (D2Q9)."""
    for y in prange(height):
        for x in range(width):
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
                f_T_new[i, y, x] = f_T[i, sy, sx]


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _stream_temperature_3d_nb(
    f_T: np.ndarray,
    f_T_new: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
) -> None:
    """Stream temperature distribution in 3D (D3Q19)."""
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                for i in range(n_vel):
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
                    f_T_new[i, z, y, x] = f_T[i, sz, sy, sx]


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _collide_temperature_2d_nb(
    f_T: np.ndarray,
    rho: np.ndarray,
    temperature: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega_T: float,
    height: int,
    width: int,
) -> None:
    """BGK collision for temperature distribution in 2D (D2Q9)."""
    for y in prange(height):
        for x in range(width):
            # Compute temperature moment
            T = 0.0
            for i in range(9):
                T += f_T[i, y, x]

            # Compute equilibrium for temperature
            r = rho[y, x]
            T_val = T if r > 0 else 0.0
            T_norm = T_val

            for i in range(9):
                feq_T = w[i] * T_norm
                f_T[i, y, x] = f_T[i, y, x] * (1.0 - omega_T) + feq_T * omega_T

            temperature[y, x] = T_norm


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _collide_temperature_3d_nb(
    f_T: np.ndarray,
    rho: np.ndarray,
    temperature: np.ndarray,
    fw: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    omega_T: float,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
) -> None:
    """BGK collision for temperature distribution in 3D (D3Q19)."""
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                T = 0.0
                for i in range(n_vel):
                    T += f_T[i, z, y, x]

                T_val = T

                for i in range(n_vel):
                    feq_T = fw[i] * T_val
                    f_T[i, z, y, x] = (
                        f_T[i, z, y, x] * (1.0 - omega_T) + feq_T * omega_T
                    )

                temperature[z, y, x] = T_val


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _apply_buoyancy_2d_nb(
    f: np.ndarray,
    temperature: np.ndarray,
    rho: np.ndarray,
    beta: float,
    T_ref: float,
    g_x: float,
    g_y: float,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    height: int,
    width: int,
    mask: np.ndarray,
) -> None:
    """Apply a Boussinesq body force to the velocity distribution in 2D.

    F = -beta * (T - T_ref) * g

    The force enters as the discrete body-force term ``w_i * (F . c_i) / cs2``
    with cs2 = 1/3. Summing over i, the added momentum is exactly F, so the
    scheme carries no spurious directional component.

    ``mask`` is the obstacle mask; the force is not applied inside solids.
    """
    inv_cs2 = 3.0
    for y in prange(height):
        for x in range(width):
            if mask[y, x]:
                continue
            T = temperature[y, x]
            if rho[y, x] <= 0.0:
                continue
            magnitude = -beta * (T - T_ref)
            fx = magnitude * g_x
            fy = magnitude * g_y
            for i in range(9):
                f[i, y, x] += w[i] * inv_cs2 * (fx * cx[i] + fy * cy[i])


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _apply_buoyancy_3d_nb(
    f: np.ndarray,
    temperature: np.ndarray,
    rho: np.ndarray,
    beta: float,
    T_ref: float,
    g_x: float,
    g_y: float,
    g_z: float,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
    mask: np.ndarray,
) -> None:
    """Apply a Boussinesq body force to the velocity distribution in 3D.

    F = -beta * (T - T_ref) * g, applied as ``w_i * (F . c_i) / cs2`` over all
    n_vel directions. The previous implementation added the force only to the
    six axis-aligned populations, which applied a strongly anisotropic body
    force to D3Q19 and no force along the twelve diagonals at all.

    ``mask`` is the obstacle mask; the force is not applied inside solids.
    """
    inv_cs2 = 3.0
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                if mask[z, y, x]:
                    continue
                T = temperature[z, y, x]
                if rho[z, y, x] <= 0.0:
                    continue
                magnitude = -beta * (T - T_ref)
                fx = magnitude * g_x
                fy = magnitude * g_y
                fz = magnitude * g_z
                for i in range(n_vel):
                    f[i, z, y, x] += (
                        w[i] * inv_cs2 * (fx * cx[i] + fy * cy[i] + fz * cz[i])
                    )


class ThermalMixin:
    """Mixin adding temperature field simulation via LBM.

    Adds a temperature distribution function f_T that is streamed and collided
    alongside the velocity distribution f, with Boussinesq buoyancy coupling.

    Usage:
        class MyEngine(SimEngine, ThermalMixin):
            def step(self):
                self.streaming()
                self.apply_boundary_conditions()
                self.apply_buoyancy()  # New: couple temperature to velocity
                self.collision()
                self.collision_temperature()  # New: collide temperature
                ...
    """

    def init_thermal(
        self,
        thermal_diffusivity: float = 0.02,
        beta: float = 0.001,
        T_ref: float = 0.0,
        g_x: float = 0.0,
        g_y: float = -1.0,
        g_z: float = 0.0,
    ) -> None:
        """Initialize thermal fields.

        Args:
            thermal_diffusivity: Thermal diffusivity (α). Controls temperature
                                diffusion rate. Similar to kinematic viscosity.
            beta: Thermal expansion coefficient for Boussinesq approximation.
            T_ref: Reference temperature for buoyancy force calculation.
            g_x: Gravity direction x-component.
            g_y: Gravity direction y-component.
            g_z: Gravity direction z-component (3D only).
        """
        self.thermal_diffusivity = thermal_diffusivity
        self.beta = beta
        self.T_ref = T_ref
        self.g_x = g_x
        self.g_y = g_y
        self.g_z = g_z

        # Temperature distribution and macroscopic field
        if self.ndim == 2:
            self.f_T = np.zeros((9, self.height, self.width), dtype=np.float32)
            self._f_T_swap = np.zeros((9, self.height, self.width), dtype=np.float32)
            self.temperature = np.zeros((self.height, self.width), dtype=np.float32)
        else:
            self.f_T = np.zeros(
                (19, self.depth, self.height, self.width), dtype=np.float32
            )
            self._f_T_swap = np.zeros(
                (19, self.depth, self.height, self.width), dtype=np.float32
            )
            self.temperature = np.zeros(
                (self.depth, self.height, self.width), dtype=np.float32
            )

        self.thermal_enabled = True

    def collision_temperature(self) -> None:
        """Collide temperature distribution using BGK."""
        if not hasattr(self, "thermal_enabled") or not self.thermal_enabled:
            return

        omega_T = self.lattice.omega_from_viscosity(self.thermal_diffusivity)

        if self.ndim == 2:
            _collide_temperature_2d_nb(
                self.f_T,
                self.rho,
                self.temperature,
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                omega_T,
                self.height,
                self.width,
            )
        else:
            _collide_temperature_3d_nb(
                self.f_T,
                self.rho,
                self.temperature,
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                self.lattice.cz,
                omega_T,
                self.lattice.n_velocities,
                self.depth,
                self.height,
                self.width,
            )

    def streaming_temperature(self) -> None:
        """Stream temperature distribution."""
        if not hasattr(self, "thermal_enabled") or not self.thermal_enabled:
            return

        if self.ndim == 2:
            _stream_temperature_2d_nb(
                self.f_T,
                self._f_T_swap,
                self.lattice.cx,
                self.lattice.cy,
                self.height,
                self.width,
            )
            self.f_T, self._f_T_swap = self._f_T_swap, self.f_T
        else:
            _stream_temperature_3d_nb(
                self.f_T,
                self._f_T_swap,
                self.lattice.cx,
                self.lattice.cy,
                self.lattice.cz,
                self.lattice.n_velocities,
                self.depth,
                self.height,
                self.width,
            )
            self.f_T, self._f_T_swap = self._f_T_swap, self.f_T

    def apply_buoyancy(self) -> None:
        """Apply the Boussinesq buoyancy force to the velocity distributions."""
        if not hasattr(self, "thermal_enabled") or not self.thermal_enabled:
            return

        mask = self.get_obstacles()
        if self.ndim == 2:
            _apply_buoyancy_2d_nb(
                self.f,
                self.temperature,
                self.rho,
                self.beta,
                self.T_ref,
                self.g_x,
                self.g_y,
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                self.height,
                self.width,
                mask,
            )
        else:
            _apply_buoyancy_3d_nb(
                self.f,
                self.temperature,
                self.rho,
                self.beta,
                self.T_ref,
                self.g_x,
                self.g_y,
                self.g_z,
                self.lattice.w,
                self.lattice.cx,
                self.lattice.cy,
                self.lattice.cz,
                self.lattice.n_velocities,
                self.depth,
                self.height,
                self.width,
                mask,
            )

    def set_temperature_boundary(self, value: float, region: str = "left") -> None:
        """Set temperature boundary condition.

        Args:
            value: Temperature value to impose.
            region: Which boundary to set ('left', 'right', 'top', 'bottom').

        'top' is the last row and 'bottom' is row 0, matching
        ``apply_moving_wall_top`` / ``apply_moving_wall_bottom``. The two were
        previously swapped, so asking for a heated floor heated the ceiling and
        every Rayleigh-Benard setup was stratified the wrong way up.
        """
        if not hasattr(self, "thermal_enabled") or not self.thermal_enabled:
            return

        if self.ndim == 2:
            if region == "left":
                self.f_T[:, :, 0] = self.lattice.w[:, np.newaxis] * value
            elif region == "right":
                self.f_T[:, :, -1] = self.lattice.w[:, np.newaxis] * value
            elif region == "top":
                self.f_T[:, -1, :] = self.lattice.w[:, np.newaxis] * value
            elif region == "bottom":
                self.f_T[:, 0, :] = self.lattice.w[:, np.newaxis] * value
        else:
            if region == "left":
                self.f_T[:, :, :, 0] = self.lattice.w[:, np.newaxis, np.newaxis] * value
            elif region == "right":
                self.f_T[:, :, :, -1] = (
                    self.lattice.w[:, np.newaxis, np.newaxis] * value
                )
            elif region == "top":
                self.f_T[:, :, -1, :] = (
                    self.lattice.w[:, np.newaxis, np.newaxis] * value
                )
            elif region == "bottom":
                self.f_T[:, :, 0, :] = self.lattice.w[:, np.newaxis, np.newaxis] * value

    def get_temperature(self) -> np.ndarray:
        """Get temperature field."""
        if hasattr(self, "thermal_enabled") and self.thermal_enabled:
            return self.temperature.copy()
        return np.zeros_like(self.rho)
