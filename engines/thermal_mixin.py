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
            T_safe = T_val if T_val > 0 else 1.0
            T_norm = T_val  # Temperature is a passive scalar, not normalized by rho

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
                    f_T[i, z, y, x] = f_T[i, z, y, x] * (1.0 - omega_T) + feq_T * omega_T

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
    height: int,
    width: int,
) -> None:
    """Apply Boussinesq buoyancy force to velocity distributions in 2D.

    Force: F = -beta * (T - T_ref) * g_hat
    Applied as a forcing term to the equilibrium distribution.
    """
    for y in prange(height):
        for x in range(width):
            T = temperature[y, x]
            r = rho[y, x]
            if r > 0:
                # Buoyancy force magnitude
                force = -beta * (T - T_ref)
                # Apply as momentum forcing
                f[3, y, x] += force * g_y * 0.5  # +y direction
                f[4, y, x] -= force * g_y * 0.5  # -y direction
                f[1, y, x] += force * g_x * 0.5  # +x direction
                f[2, y, x] -= force * g_x * 0.5  # -x direction


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
    depth: int,
    height: int,
    width: int,
) -> None:
    """Apply Boussinesq buoyancy force to velocity distributions in 3D."""
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                T = temperature[z, y, x]
                r = rho[z, y, x]
                if r > 0:
                    force = -beta * (T - T_ref)
                    # D3Q19 directions: 1-6 are face centers
                    f[5, z, y, x] += force * g_z * 0.5  # +z
                    f[6, z, y, x] -= force * g_z * 0.5  # -z
                    f[3, z, y, x] += force * g_y * 0.5  # +y
                    f[4, z, y, x] -= force * g_y * 0.5  # -y
                    f[1, z, y, x] += force * g_x * 0.5  # +x
                    f[2, z, y, x] -= force * g_x * 0.5  # -x


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
            self.temperature = np.zeros((self.height, self.width), dtype=np.float32)
        else:
            self.f_T = np.zeros((19, self.depth, self.height, self.width), dtype=np.float32)
            self.temperature = np.zeros((self.depth, self.height, self.width), dtype=np.float32)

        self.thermal_enabled = True

    def collision_temperature(self) -> None:
        """Collide temperature distribution using BGK."""
        if not hasattr(self, 'thermal_enabled') or not self.thermal_enabled:
            return

        omega_T = self.lattice.omega_from_viscosity(self.thermal_diffusivity)

        if self.ndim == 2:
            _collide_temperature_2d_nb(
                self.f_T, self.rho, self.temperature,
                self.lattice.w, self.lattice.cx, self.lattice.cy,
                omega_T, self.height, self.width,
            )
        else:
            _collide_temperature_3d_nb(
                self.f_T, self.rho, self.temperature,
                self.lattice.w, self.lattice.cx, self.lattice.cy, self.lattice.cz,
                omega_T, self.lattice.n_velocities,
                self.depth, self.height, self.width,
            )

    def apply_buoyancy(self) -> None:
        """Apply Boussinesq buoyancy force to velocity distributions."""
        if not hasattr(self, 'thermal_enabled') or not self.thermal_enabled:
            return

        if self.ndim == 2:
            _apply_buoyancy_2d_nb(
                self.f, self.temperature, self.rho,
                self.beta, self.T_ref, self.g_x, self.g_y,
                self.height, self.width,
            )
        else:
            _apply_buoyancy_3d_nb(
                self.f, self.temperature, self.rho,
                self.beta, self.T_ref, self.g_x, self.g_y, self.g_z,
                self.depth, self.height, self.width,
            )

    def set_temperature_boundary(self, value: float, region: str = "left") -> None:
        """Set temperature boundary condition.

        Args:
            value: Temperature value to impose.
            region: Which boundary to set ('left', 'right', 'top', 'bottom').
        """
        if not hasattr(self, 'thermal_enabled') or not self.thermal_enabled:
            return

        if self.ndim == 2:
            if region == "left":
                self.f_T[:, :, 0] = self.lattice.w[:, np.newaxis] * value
            elif region == "right":
                self.f_T[:, :, -1] = self.lattice.w[:, np.newaxis] * value
            elif region == "top":
                self.f_T[:, 0, :] = self.lattice.w[:, np.newaxis] * value
            elif region == "bottom":
                self.f_T[:, -1, :] = self.lattice.w[:, np.newaxis] * value
        else:
            if region == "left":
                self.f_T[:, :, :, 0] = self.lattice.w[:, np.newaxis, np.newaxis] * value
            elif region == "right":
                self.f_T[:, :, :, -1] = self.lattice.w[:, np.newaxis, np.newaxis] * value
            elif region == "top":
                self.f_T[:, :, 0, :] = self.lattice.w[:, np.newaxis, np.newaxis] * value
            elif region == "bottom":
                self.f_T[:, :, -1, :] = self.lattice.w[:, np.newaxis, np.newaxis] * value

    def get_temperature(self) -> np.ndarray:
        """Get temperature field."""
        if hasattr(self, 'thermal_enabled') and self.thermal_enabled:
            return self.temperature.copy()
        return np.zeros_like(self.rho)
