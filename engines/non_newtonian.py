"""Non-Newtonian viscosity models for LBM.

Adds shear-rate dependent viscosity for simulating:
- Power-law fluids (shear-thinning, shear-thickening)
- Carreau model (polymer-like behavior)
- Bingham plastics (yield stress fluids)

These models compute local viscosity from the strain rate tensor,
reusing the same non-equilibrium stress computation as Smagorinsky.

Reference:bird, Armstrong, & Hassager (1987), "Dynamics of Polymeric
Liquids, Volume 1: Fluid Mechanics."
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange


class NonNewtonianModel(ABC):
    """Abstract base class for non-Newtonian viscosity models."""

    @abstractmethod
    def compute_viscosity(
        self,
        strain_rate_magnitude: np.ndarray,
        base_viscosity: float,
    ) -> np.ndarray:
        """Compute local viscosity from strain rate magnitude.

        Args:
            strain_rate_magnitude: |S| at each grid point.
            base_viscosity: Reference viscosity at zero shear rate.

        Returns:
            Local viscosity field.
        """
        ...


class PowerLawModel(NonNewtonianModel):
    """Power-law (Ostwald-de Waele) viscosity model.

    nu = nu_0 * gamma_dot^(n-1)

    where:
    - nu_0: Reference viscosity
    - gamma_dot: Shear rate magnitude
    - n: Flow behavior index
        - n < 1: Shear-thinning (pseudoplastic) - e.g., blood, paint
        - n = 1: Newtonian (reduces to constant viscosity)
        - n > 1: Shear-thickening (dilatant) - e.g., cornstarch, quicksand

    Reference: Bird et al. (1987).
    """

    def __init__(self, n: float = 0.5) -> None:
        """
        Args:
            n: Flow behavior index. Default 0.5 (shear-thinning).
               Range: (0, 2). Typical values: 0.2-0.8 for polymers.
        """
        self.n = n

    def compute_viscosity(
        self,
        strain_rate_magnitude: np.ndarray,
        base_viscosity: float,
    ) -> np.ndarray:
        # Avoid division by zero for zero shear rate
        safe_shear = np.maximum(strain_rate_magnitude, 1e-10)
        return base_viscosity * np.power(safe_shear, self.n - 1.0)


class CarreauModel(NonNewtonianModel):
    """Carreau viscosity model.

    More realistic than power-law for polymers. Captures:
    - Newtonian plateau at low shear rates
    - Power-law region at moderate shear rates
    - Newtonian plateau at high shear rates

    nu = nu_inf + (nu_0 - nu_inf) * (1 + (lambda * gamma_dot)^2)^((n-1)/2)

    Reference: Carreau (1972), "Rheological equations from molecular
    network theories."
    """

    def __init__(
        self,
        n: float = 0.5,
        lambda_val: float = 1.0,
        nu_inf_ratio: float = 0.01,
    ) -> None:
        """
        Args:
            n: Power-law index (same as PowerLawModel).
            lambda_val: Relaxation time. Controls shear-thinning onset.
            nu_inf_ratio: Ratio of infinite-shear viscosity to base viscosity.
                         Typically 0.01-0.1.
        """
        self.n = n
        self.lambda_val = lambda_val
        self.nu_inf_ratio = nu_inf_ratio

    def compute_viscosity(
        self,
        strain_rate_magnitude: np.ndarray,
        base_viscosity: float,
    ) -> np.ndarray:
        nu_inf = base_viscosity * self.nu_inf_ratio
        nu_0 = base_viscosity

        # Carreau model
        lambda_shear = self.lambda_val * strain_rate_magnitude
        viscosity = nu_inf + (nu_0 - nu_inf) * np.power(
            1.0 + lambda_shear * lambda_shear, (self.n - 1.0) / 2.0
        )
        return viscosity


class BinghamModel(NonNewtonianModel):
    """Bingham plastic viscosity model.

    Fluid behaves as solid below yield stress, flows like liquid above.

    mu = mu_p + tau_y / |S|   when |S| > 0
    mu = infinity              when |S| = 0

    Reference: Bingham (1922), "Fluidity and Plasticity."
    """

    def __init__(self, tau_y: float = 0.1, mu_p_ratio: float = 0.1) -> None:
        """
        Args:
            tau_y: Yield stress. Fluid won't flow below this stress.
            mu_p_ratio: Ratio of plastic viscosity to base viscosity.
        """
        self.tau_y = tau_y
        self.mu_p_ratio = mu_p_ratio

    def compute_viscosity(
        self,
        strain_rate_magnitude: np.ndarray,
        base_viscosity: float,
    ) -> np.ndarray:
        mu_p = base_viscosity * self.mu_p_ratio
        safe_shear = np.maximum(strain_rate_magnitude, 1e-10)
        viscosity = mu_p + self.tau_y / safe_shear
        return viscosity


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _compute_strain_rate_2d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    strain_rate: np.ndarray,
    height: int,
    width: int,
) -> None:
    """Compute strain rate magnitude from non-equilibrium distributions in 2D."""
    for y in prange(height):
        for x in range(width):
            r = rho[y, x]
            if r <= 0:
                strain_rate[y, x] = 0.0
                continue

            u_vel = u[y, x]
            v_vel = v[y, x]
            u2 = u_vel * u_vel + v_vel * v_vel

            # Compute non-equilibrium stress tensor
            pi_neq_xx = 0.0
            pi_neq_yy = 0.0
            pi_neq_xy = 0.0
            for i in range(9):
                cu = cx[i] * u_vel + cy[i] * v_vel
                feq_i = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                fneq_i = f[i, y, x] - feq_i
                pi_neq_xx += cx[i] * cx[i] * fneq_i
                pi_neq_yy += cy[i] * cy[i] * fneq_i
                pi_neq_xy += cx[i] * cy[i] * fneq_i

            # Strain rate magnitude
            s_mag_sq = pi_neq_xx * pi_neq_xx + pi_neq_yy * pi_neq_yy + 2.0 * pi_neq_xy * pi_neq_xy
            strain_rate[y, x] = np.sqrt(2.0 * s_mag_sq) if s_mag_sq > 0 else 0.0


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _compute_strain_rate_3d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w_vel: np.ndarray,
    fw: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    strain_rate: np.ndarray,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
) -> None:
    """Compute strain rate magnitude from non-equilibrium distributions in 3D."""
    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                r = rho[z, y, x]
                if r <= 0:
                    strain_rate[z, y, x] = 0.0
                    continue

                u_val = u[z, y, x]
                v_val = v[z, y, x]
                w_val = w_vel[z, y, x]
                u2 = u_val * u_val + v_val * v_val + w_val * w_val

                pi_neq_xx = 0.0
                pi_neq_yy = 0.0
                pi_neq_zz = 0.0
                pi_neq_xy = 0.0
                pi_neq_xz = 0.0
                pi_neq_yz = 0.0
                for i in range(n_vel):
                    cu = cx[i] * u_val + cy[i] * v_val + cz[i] * w_val
                    feq_i = fw[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                    fneq_i = f[i, z, y, x] - feq_i
                    pi_neq_xx += cx[i] * cx[i] * fneq_i
                    pi_neq_yy += cy[i] * cy[i] * fneq_i
                    pi_neq_zz += cz[i] * cz[i] * fneq_i
                    pi_neq_xy += cx[i] * cy[i] * fneq_i
                    pi_neq_xz += cx[i] * cz[i] * fneq_i
                    pi_neq_yz += cy[i] * cz[i] * fneq_i

                s_mag_sq = (
                    pi_neq_xx * pi_neq_xx + pi_neq_yy * pi_neq_yy + pi_neq_zz * pi_neq_zz
                    + 2.0 * (pi_neq_xy * pi_neq_xy + pi_neq_xz * pi_neq_xz + pi_neq_yz * pi_neq_yz)
                )
                strain_rate[z, y, x] = np.sqrt(2.0 * s_mag_sq) if s_mag_sq > 0 else 0.0


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _apply_non_newtonian_2d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    viscosity_field: np.ndarray,
    height: int,
    width: int,
) -> None:
    """Apply local viscosity to collision in 2D (variable omega per cell)."""
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

            # Local omega from local viscosity
            nu_local = viscosity_field[y, x]
            omega_local = 1.0 / (3.0 * nu_local + 0.5)
            omega_local = min(max(omega_local, 0.01), 1.99)

            u2 = u_vel * u_vel + v_vel * v_vel
            for i in range(9):
                cu = cx[i] * u_vel + cy[i] * v_vel
                feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f[i, y, x] = f[i, y, x] * (1.0 - omega_local) + feq * omega_local

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _apply_non_newtonian_3d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w_vel: np.ndarray,
    fw: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    viscosity_field: np.ndarray,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
) -> None:
    """Apply local viscosity to collision in 3D."""
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

                nu_local = viscosity_field[z, y, x]
                omega_local = 1.0 / (3.0 * nu_local + 0.5)
                omega_local = min(max(omega_local, 0.01), 1.99)

                u2 = u_val * u_val + v_val * v_val + w_val * w_val
                for i in range(n_vel):
                    cu = cx[i] * u_val + cy[i] * v_val + cz[i] * w_val
                    feq = fw[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                    f[i, z, y, x] = f[i, z, y, x] * (1.0 - omega_local) + feq * omega_local

                rho[z, y, x] = r
                u[z, y, x] = u_val
                v[z, y, x] = v_val
                w_vel[z, y, x] = w_val


class NonNewtonianCollision:
    """Non-Newtonian collision operator wrapper.

    Wraps an existing collision operator and applies local viscosity
    based on strain rate and a non-Newtonian model.

    Usage:
        power_law = PowerLawModel(n=0.5)
        collision = NonNewtonianCollision(
            BGKCollision(), power_law, base_viscosity=0.02
        )
        sim = LBM2D(width=128, height=128, collision=collision)
    """

    def __init__(
        self,
        base_collision: object,
        model: NonNewtonianModel,
        base_viscosity: float = 0.02,
    ) -> None:
        """
        Args:
            base_collision: Underlying collision operator (BGK, TRT, etc.)
            model: Non-Newtonian viscosity model.
            base_viscosity: Reference viscosity for the model.
        """
        self.base_collision = base_collision
        self.model = model
        self.base_viscosity = base_viscosity
        self._strain_rate: np.ndarray | None = None

    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: object,
        viscosity: float,
        w_vel: np.ndarray | None = None,
    ) -> None:
        """Apply non-Newtonian collision.

        1. Compute strain rate magnitude from non-equilibrium distributions
        2. Compute local viscosity from strain rate using the model
        3. Apply variable-omega collision with local viscosity
        """
        from engines.lbm_common import Lattice2D, Lattice3D

        if f.ndim == 4:
            depth, height, width = f.shape[1], f.shape[2], f.shape[3]
            if self._strain_rate is None or self._strain_rate.shape != (depth, height, width):
                self._strain_rate = np.zeros((depth, height, width), dtype=np.float32)
            _compute_strain_rate_3d_nb(
                f, rho, u, v, w_vel,
                lattice.w, lattice.cx, lattice.cy, lattice.cz,
                self._strain_rate, lattice.n_velocities,
                depth, height, width,
            )
            viscosity_field = self.model.compute_viscosity(
                self._strain_rate, self.base_viscosity
            )
            _apply_non_newtonian_3d_nb(
                f, rho, u, v, w_vel,
                lattice.w, lattice.cx, lattice.cy, lattice.cz,
                viscosity_field, lattice.n_velocities,
                depth, height, width,
            )
        else:
            height, width = f.shape[1], f.shape[2]
            if self._strain_rate is None or self._strain_rate.shape != (height, width):
                self._strain_rate = np.zeros((height, width), dtype=np.float32)
            _compute_strain_rate_2d_nb(
                f, rho, u, v,
                lattice.w, lattice.cx, lattice.cy,
                self._strain_rate, height, width,
            )
            viscosity_field = self.model.compute_viscosity(
                self._strain_rate, self.base_viscosity
            )
            _apply_non_newtonian_2d_nb(
                f, rho, u, v,
                lattice.w, lattice.cx, lattice.cy,
                viscosity_field, height, width,
            )
