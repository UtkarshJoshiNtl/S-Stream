from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange

from engines.lbm_common import Lattice2D, Lattice3D


class CollisionOperator(ABC):
    @abstractmethod
    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D,
        viscosity: float,
    ) -> None: ...


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _bgk_collide_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega: float,
    height: int,
    width: int,
) -> None:
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
def _bgk_collide_3d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w_vel: np.ndarray,
    fw: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    omega: float,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
) -> None:
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
                    feq = fw[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                    f[i, z, y, x] = f[i, z, y, x] * (1.0 - omega) + feq * omega
                rho[z, y, x] = r
                u[z, y, x] = u_val
                v[z, y, x] = v_val
                w_vel[z, y, x] = w_val


class BGKCollision(CollisionOperator):
    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D | Lattice3D,
        viscosity: float,
        w_vel: np.ndarray | None = None,
    ) -> None:
        omega = lattice.omega_from_viscosity(viscosity)
        if f.ndim == 4:
            # 3D case
            _bgk_collide_3d_nb(
                f,
                rho,
                u,
                v,
                w_vel,
                lattice.w,
                lattice.cx,
                lattice.cy,
                lattice.cz,
                omega,
                lattice.n_velocities,
                f.shape[1],
                f.shape[2],
                f.shape[3],
            )
        else:
            # 2D case
            _bgk_collide_nb(
                f,
                rho,
                u,
                v,
                lattice.w,
                lattice.cx,
                lattice.cy,
                omega,
                f.shape[1],
                f.shape[2],
            )


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _trt_collide_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    opp: np.ndarray,
    s_plus: float,
    s_minus: float,
    height: int,
    width: int,
) -> None:
    """TRT (Two-Relaxation-Time) collision kernel.

    Decomposes f into symmetric (m+) and antisymmetric (m-) modes,
    then relaxes each with its own rate.

    Reference: Geller et al. (2013), "A simple and accurate scheme for the
    lattice Boltzmann method."
    """
    for y in prange(height):
        for x in range(width):
            # Compute density and velocity
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

            # Compute equilibrium
            u2 = u_vel * u_vel + v_vel * v_vel
            feq = np.empty(9, dtype=np.float32)
            for i in range(9):
                cu = cx[i] * u_vel + cy[i] * v_vel
                feq[i] = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

            # TRT decomposition: symmetric (m+) and antisymmetric (m-)
            for i in range(9):
                opp_i = opp[i]
                if i <= opp_i:
                    # Symmetric mode: average of f and its opposite
                    m_plus = 0.5 * (f[i, y, x] + f[opp_i, y, x])
                    m_minus = 0.5 * (f[i, y, x] - f[opp_i, y, x])
                    feq_plus = 0.5 * (feq[i] + feq[opp_i])
                    feq_minus = 0.5 * (feq[i] - feq[opp_i])

                    # Relax each mode
                    new_plus = m_plus - s_plus * (m_plus - feq_plus)
                    new_minus = m_minus - s_minus * (m_minus - feq_minus)

                    # Recompose
                    f[i, y, x] = new_plus + new_minus
                    f[opp_i, y, x] = new_plus - new_minus

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel


class TRTCollision(CollisionOperator):
    """Two-relaxation-time collision operator.

    Uses two relaxation rates:
    - s_plus: controls viscosity (related to omega)
    - s_minus: controls boundary stability (typically ~1/4 for optimal stability)

    Reference: Geller et al. (2013), "A simple and accurate scheme for the
    lattice Boltzmann method."
    """

    def __init__(self, s_minus: float = 0.25) -> None:
        """
        Args:
            s_minus: Antisymmetric relaxation rate. Default 0.25 gives optimal
                     stability near boundaries. Range: (0, 1].
        """
        self.s_minus = s_minus

    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D | Lattice3D,
        viscosity: float,
        w_vel: np.ndarray | None = None,
    ) -> None:
        if f.ndim == 4:
            raise NotImplementedError("TRT collision not yet implemented for 3D")
        omega = lattice.omega_from_viscosity(viscosity)
        s_plus = omega
        _trt_collide_nb(
            f,
            rho,
            u,
            v,
            lattice.w,
            lattice.cx,
            lattice.cy,
            lattice.opp,
            s_plus,
            self.s_minus,
            f.shape[1],
            f.shape[2],
        )


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _mrt_collide_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    s: np.ndarray,
    height: int,
    width: int,
) -> None:
    """MRT (Multiple-Relaxation-Time) collision kernel.

    Transforms f to moment space, relaxes each moment independently,
    then transforms back.

    Reference: d'Humières et al. (2002), "Multiple-relaxation-time Lattice
    Boltzmann models for 3D simulations."
    """
    # D2Q9 moment transformation matrix M and its inverse M_inv
    # Moments: [rho, e, eps, jx, jy, qx, qy, pxx, pyy, pxy]
    # We use the standard D2Q9 MRT matrices.

    for y in prange(height):
        for x in range(width):
            # Compute density and velocity
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

            # Transform to moment space: m = M * f
            # Using the standard D2Q9 MRT transformation
            m0 = (
                f[0, y, x]
                + f[1, y, x]
                + f[2, y, x]
                + f[3, y, x]
                + f[4, y, x]
                + f[5, y, x]
                + f[6, y, x]
                + f[7, y, x]
                + f[8, y, x]
            )
            m1 = (
                -4 * f[0, y, x]
                - f[1, y, x]
                - f[2, y, x]
                - f[3, y, x]
                - f[4, y, x]
                + 2 * f[5, y, x]
                + 2 * f[6, y, x]
                + 2 * f[7, y, x]
                + 2 * f[8, y, x]
            )
            m2 = (
                4 * f[0, y, x]
                - 2 * f[1, y, x]
                - 2 * f[2, y, x]
                - 2 * f[3, y, x]
                - 2 * f[4, y, x]
                + f[5, y, x]
                + f[6, y, x]
                + f[7, y, x]
                + f[8, y, x]
            )
            m3 = (
                f[1, y, x]
                - f[3, y, x]
                + f[5, y, x]
                - f[6, y, x]
                + f[7, y, x]
                - f[8, y, x]
            )
            m4 = (
                f[2, y, x]
                - f[4, y, x]
                + f[5, y, x]
                + f[6, y, x]
                - f[7, y, x]
                - f[8, y, x]
            )
            m5 = (
                -f[1, y, x]
                + f[3, y, x]
                + f[5, y, x]
                - f[6, y, x]
                + f[7, y, x]
                - f[8, y, x]
            )
            m6 = (
                -f[2, y, x]
                + f[4, y, x]
                + f[5, y, x]
                + f[6, y, x]
                - f[7, y, x]
                - f[8, y, x]
            )
            m7 = (
                f[1, y, x]
                + f[3, y, x]
                - f[5, y, x]
                - f[6, y, x]
                - f[7, y, x]
                - f[8, y, x]
            )
            m8 = (
                -f[1, y, x]
                - f[3, y, x]
                + f[5, y, x]
                + f[6, y, x]
                - f[7, y, x]
                - f[8, y, x]
            )
            m9 = (
                f[1, y, x]
                - f[2, y, x]
                + f[3, y, x]
                - f[4, y, x]
                - f[5, y, x]
                - f[6, y, x]
                + f[7, y, x]
                + f[8, y, x]
            )

            # Compute equilibrium moments
            e1 = r * (-2 + 3 * u_vel * u_vel + 3 * v_vel * v_vel)
            e2 = r * (1 - 3 * u_vel * u_vel - 3 * v_vel * v_vel)
            e3 = r * u_vel
            e4 = r * v_vel
            e5 = -r * u_vel
            e6 = -r * v_vel
            e7 = r * (u_vel * u_vel - v_vel * v_vel)
            e8 = -r * (u_vel * u_vel - v_vel * v_vel)
            e9 = r * u_vel * v_vel

            # Relax each moment independently
            # s[0]=1.0 (conserved), s[4]=s[8]=omega, rest=1.0
            m_new = np.empty(10, dtype=np.float32)
            m_new[0] = m0  # density conserved
            m_new[1] = m1 - s[1] * (m1 - e1)
            m_new[2] = m2 - s[2] * (m2 - e2)
            m_new[3] = m3 - s[3] * (m3 - e3)
            m_new[4] = m4 - s[4] * (m4 - e4)
            m_new[5] = m5 - s[5] * (m5 - e5)
            m_new[6] = m6 - s[6] * (m6 - e6)
            m_new[7] = m7 - s[7] * (m7 - e7)
            m_new[8] = m8 - s[8] * (m8 - e8)
            m_new[9] = m9 - s[9] * (m9 - e9)

            # Transform back to distribution space: f = M_inv * m
            # Using the standard D2Q9 MRT inverse transformation
            f[0, y, x] = (m_new[0] - m_new[1] + m_new[2]) / 9.0
            f[1, y, x] = (
                m_new[0]
                - m_new[1] / 2.0
                + m_new[2] / 4.0
                + m_new[3] / 2.0
                + m_new[5] / 2.0
                + m_new[7] / 4.0
                + m_new[9] / 2.0
            ) / 9.0
            f[2, y, x] = (
                m_new[0]
                - m_new[1] / 2.0
                + m_new[2] / 4.0
                + m_new[4] / 2.0
                + m_new[6] / 2.0
                + m_new[8] / 4.0
                - m_new[9] / 2.0
            ) / 9.0
            f[3, y, x] = (
                m_new[0]
                - m_new[1] / 2.0
                + m_new[2] / 4.0
                - m_new[3] / 2.0
                - m_new[5] / 2.0
                + m_new[7] / 4.0
                + m_new[9] / 2.0
            ) / 9.0
            f[4, y, x] = (
                m_new[0]
                - m_new[1] / 2.0
                + m_new[2] / 4.0
                - m_new[4] / 2.0
                - m_new[6] / 2.0
                + m_new[8] / 4.0
                - m_new[9] / 2.0
            ) / 9.0
            f[5, y, x] = (
                m_new[0]
                + m_new[1] / 4.0
                + m_new[2] / 8.0
                + m_new[3] / 2.0
                + m_new[4] / 2.0
                + m_new[5] / 2.0
                + m_new[6] / 2.0
                + m_new[7] / 4.0
                + m_new[8] / 4.0
                + m_new[9] / 2.0
            ) / 9.0
            f[6, y, x] = (
                m_new[0]
                + m_new[1] / 4.0
                + m_new[2] / 8.0
                - m_new[3] / 2.0
                + m_new[4] / 2.0
                - m_new[5] / 2.0
                + m_new[6] / 2.0
                + m_new[7] / 4.0
                + m_new[8] / 4.0
                - m_new[9] / 2.0
            ) / 9.0
            f[7, y, x] = (
                m_new[0]
                + m_new[1] / 4.0
                + m_new[2] / 8.0
                - m_new[3] / 2.0
                - m_new[4] / 2.0
                - m_new[5] / 2.0
                - m_new[6] / 2.0
                + m_new[7] / 4.0
                + m_new[8] / 4.0
                + m_new[9] / 2.0
            ) / 9.0
            f[8, y, x] = (
                m_new[0]
                + m_new[1] / 4.0
                + m_new[2] / 8.0
                + m_new[3] / 2.0
                - m_new[4] / 2.0
                + m_new[5] / 2.0
                - m_new[6] / 2.0
                + m_new[7] / 4.0
                + m_new[8] / 4.0
                - m_new[9] / 2.0
            ) / 9.0

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel


class MRTCollision(CollisionOperator):
    """Multiple-relaxation-time collision operator.

    Decouples all moment relaxation rates for maximum accuracy.
    Requires transformation matrices M and M_inv for D2Q9.

    Reference: d'Humières et al. (2002), "Multiple-relaxation-time Lattice
    Boltzmann models for 3D simulations."
    """

    def __init__(self) -> None:
        # Default MRT relaxation rates (s[0] is unused, kept for indexing)
        # s[1]-s[9] control relaxation of non-conserved moments
        self._s = np.array(
            [
                1.0,  # s0: unused (conserved)
                1.19,  # s1: e (kinetic energy)
                1.4,  # s2: eps (energy square)
                1.0,  # s3: jx (x-momentum)
                1.0,  # s4: jy (y-momentum)
                1.2,  # s5: qx (x-energy flux)
                1.2,  # s6: qy (y-energy flux)
                1.0,  # s7: pxx (xx stress)
                1.0,  # s8: pyy (yy stress)
                1.0,  # s9: pxy (xy stress)
            ],
            dtype=np.float32,
        )

    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D | Lattice3D,
        viscosity: float,
        w_vel: np.ndarray | None = None,
    ) -> None:
        if f.ndim == 4:
            raise NotImplementedError("MRT collision not yet implemented for 3D")
        omega = lattice.omega_from_viscosity(viscosity)
        s = self._s.copy()
        s[1] = omega  # Set viscosity-related moment
        s[7] = omega  # Set stress-related moment
        s[8] = omega  # Set stress-related moment
        _mrt_collide_nb(
            f,
            rho,
            u,
            v,
            lattice.w,
            lattice.cx,
            lattice.cy,
            s,
            f.shape[1],
            f.shape[2],
        )


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _smagorinsky_collide_2d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega_base: float,
    cs: float,
    height: int,
    width: int,
) -> None:
    """Smagorinsky SGS collision kernel for 2D (D2Q9).

    Computes turbulent viscosity from strain rate tensor and adds it to
    molecular viscosity for effective relaxation.

    Algorithm (clean-room from Smagorinsky (1963), Lilly (1967)):
    1. Compute non-equilibrium stress: Π_neq = Σ c_iα c_iβ (f_i - f_eq_i)
    2. Strain rate magnitude: |S| = sqrt(2 * S_ij * S_ij)
    3. Turbulent viscosity: nu_t = (C_s * Δ)^2 * |S|
    4. Effective omega: omega_eff = 1 / (3 * (nu + nu_t) + 0.5)

    Reference: Smagorinsky (1963), "General circulation experiments with the
    primitive equations." Lilly (1967), "The representation of small-scale
    turbulence in numerical simulation experiments."
    """
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

            # Compute equilibrium and non-equilibrium stress tensor
            u2 = u_vel * u_vel + v_vel * v_vel
            # Π_neq components (off-diagonal only needed for |S|)
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

            # Strain rate magnitude: |S| = sqrt(2 * (S_xx^2 + S_yy^2 + 2*S_xy^2))
            # S_ij = -(1/2τ) * Π_neq_ij, but we only need the magnitude
            # |S| = sqrt(2 * (S_xx^2 + S_yy^2 + 2*S_xy^2))
            # Since S_xx + S_yy = 0 (incompressible), S_xx^2 + S_yy^2 = 2*S_xx^2
            # |S| = sqrt(4*S_xx^2 + 4*S_xy^2) = 2*sqrt(S_xx^2 + S_xy^2)
            # But we use the non-equilibrium moments directly for efficiency
            s_mag_sq = (
                pi_neq_xx * pi_neq_xx
                + pi_neq_yy * pi_neq_yy
                + 2.0 * pi_neq_xy * pi_neq_xy
            )
            s_mag = np.sqrt(2.0 * s_mag_sq) if s_mag_sq > 0 else 0.0

            # Turbulent viscosity: nu_t = (C_s * Δ)^2 * |S|  (Δ = 1 in lattice units)
            nu_t = cs * cs * s_mag

            # Effective omega with turbulent viscosity
            omega_eff = 1.0 / (3.0 * (1.0 / (3.0 * omega_base - 1.5) + nu_t) + 0.5)
            omega_eff = min(omega_eff, 1.99)  # Stability clamp

            for i in range(9):
                cu = cx[i] * u_vel + cy[i] * v_vel
                feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f[i, y, x] = f[i, y, x] * (1.0 - omega_eff) + feq * omega_eff

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _smagorinsky_collide_3d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w_vel: np.ndarray,
    fw: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    omega_base: float,
    cs: float,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
) -> None:
    """Smagorinsky SGS collision kernel for 3D (D3Q19/D3Q27)."""
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

                # Compute non-equilibrium stress tensor
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

                # |S| = sqrt(2 * sum(S_ij^2))
                s_mag_sq = (
                    pi_neq_xx * pi_neq_xx
                    + pi_neq_yy * pi_neq_yy
                    + pi_neq_zz * pi_neq_zz
                    + 2.0
                    * (
                        pi_neq_xy * pi_neq_xy
                        + pi_neq_xz * pi_neq_xz
                        + pi_neq_yz * pi_neq_yz
                    )
                )
                s_mag = np.sqrt(2.0 * s_mag_sq) if s_mag_sq > 0 else 0.0

                # Turbulent viscosity and effective omega
                nu_t = cs * cs * s_mag
                omega_eff = 1.0 / (3.0 * (1.0 / (3.0 * omega_base - 1.5) + nu_t) + 0.5)
                omega_eff = min(omega_eff, 1.99)

                for i in range(n_vel):
                    cu = cx[i] * u_val + cy[i] * v_val + cz[i] * w_val
                    feq = fw[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                    f[i, z, y, x] = f[i, z, y, x] * (1.0 - omega_eff) + feq * omega_eff

                rho[z, y, x] = r
                u[z, y, x] = u_val
                v[z, y, x] = v_val
                w_vel[z, y, x] = w_val


class SmagorinskyCollision(CollisionOperator):
    """Smagorinsky subgrid-scale (SGS) turbulence model.

    Adds turbulent viscosity based on local strain rate to enable high-Re
    flows without prohibitive grid resolution.

    Reference: Smagorinsky (1963), "General circulation experiments with the
    primitive equations." Lilly (1967).
    """

    def __init__(self, cs: float = 0.1) -> None:
        """
        Args:
            cs: Smagorinsky constant. Typical range: 0.1-0.2.
                 Lower values reduce dissipation, higher values increase stability.
        """
        self.cs = cs

    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D | Lattice3D,
        viscosity: float,
        w_vel: np.ndarray | None = None,
    ) -> None:
        omega_base = lattice.omega_from_viscosity(viscosity)
        if f.ndim == 4:
            _smagorinsky_collide_3d_nb(
                f,
                rho,
                u,
                v,
                w_vel,
                lattice.w,
                lattice.cx,
                lattice.cy,
                lattice.cz,
                omega_base,
                self.cs,
                lattice.n_velocities,
                f.shape[1],
                f.shape[2],
                f.shape[3],
            )
        else:
            _smagorinsky_collide_2d_nb(
                f,
                rho,
                u,
                v,
                lattice.w,
                lattice.cx,
                lattice.cy,
                omega_base,
                self.cs,
                f.shape[1],
                f.shape[2],
            )


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _wale_collide_2d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega_base: float,
    cs_w: float,
    height: int,
    width: int,
) -> None:
    """WALE (Wall-Adapting Local Eddy-viscosity) collision kernel for 2D.

    WALE naturally handles near-wall behavior without explicit damping functions.
    Better than Smagorinsky for wall-bounded flows.

    Reference: Nicoud & Ducros (1999), "Stress tensor and subgrid-scale
    scalar dissipation in LES of turbulent flows."
    """
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

            # Compute velocity gradient tensor S_ij via non-equilibrium stress
            u2 = u_vel * u_vel + v_vel * v_vel
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

            # WALE uses S_ij^d = 0.5 * (g_ik * g_kj + g_jk * g_ki) - (1/3) * g_kk * δ_ij
            # For 2D incompressible: |S_d|^2 = S_d_xx^2 + S_d_yy^2 + 2*S_d_xy^2
            # Using non-equilibrium moments as proxy for velocity gradients
            s_d_xx = pi_neq_xx
            s_d_yy = pi_neq_yy
            s_d_xy = pi_neq_xy
            s_d_mag_sq = s_d_xx * s_d_xx + s_d_yy * s_d_yy + 2.0 * s_d_xy * s_d_xy
            s_d_mag = np.sqrt(s_d_mag_sq) if s_d_mag_sq > 0 else 0.0

            # WALE turbulent viscosity
            nu_t = (cs_w * cs_w * s_d_mag * s_d_mag * s_d_mag) if s_d_mag > 0 else 0.0

            omega_eff = 1.0 / (3.0 * (1.0 / (3.0 * omega_base - 1.5) + nu_t) + 0.5)
            omega_eff = min(omega_eff, 1.99)

            for i in range(9):
                cu = cx[i] * u_vel + cy[i] * v_vel
                feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f[i, y, x] = f[i, y, x] * (1.0 - omega_eff) + feq * omega_eff

            rho[y, x] = r
            u[y, x] = u_vel
            v[y, x] = v_vel


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _wale_collide_3d_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w_vel: np.ndarray,
    fw: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    omega_base: float,
    cs_w: float,
    n_vel: int,
    depth: int,
    height: int,
    width: int,
) -> None:
    """WALE collision kernel for 3D."""
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

                s_d_mag_sq = (
                    pi_neq_xx * pi_neq_xx
                    + pi_neq_yy * pi_neq_yy
                    + pi_neq_zz * pi_neq_zz
                    + 2.0
                    * (
                        pi_neq_xy * pi_neq_xy
                        + pi_neq_xz * pi_neq_xz
                        + pi_neq_yz * pi_neq_yz
                    )
                )
                s_d_mag = np.sqrt(s_d_mag_sq) if s_d_mag_sq > 0 else 0.0

                nu_t = (
                    (cs_w * cs_w * s_d_mag * s_d_mag * s_d_mag) if s_d_mag > 0 else 0.0
                )
                omega_eff = 1.0 / (3.0 * (1.0 / (3.0 * omega_base - 1.5) + nu_t) + 0.5)
                omega_eff = min(omega_eff, 1.99)

                for i in range(n_vel):
                    cu = cx[i] * u_val + cy[i] * v_val + cz[i] * w_val
                    feq = fw[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                    f[i, z, y, x] = f[i, z, y, x] * (1.0 - omega_eff) + feq * omega_eff

                rho[z, y, x] = r
                u[z, y, x] = u_val
                v[z, y, x] = v_val
                w_vel[z, y, x] = w_val


class WaleCollision(CollisionOperator):
    """WALE (Wall-Adapting Local Eddy-viscosity) turbulence model.

    Improved near-wall behavior compared to Smagorinsky. No explicit
    damping function required.

    Reference: Nicoud & Ducros (1999).
    """

    def __init__(self, cs: float = 0.1) -> None:
        """
        Args:
            cs: WALE constant. Typical range: 0.1-0.2.
        """
        self.cs = cs

    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lattice: Lattice2D | Lattice3D,
        viscosity: float,
        w_vel: np.ndarray | None = None,
    ) -> None:
        omega_base = lattice.omega_from_viscosity(viscosity)
        if f.ndim == 4:
            _wale_collide_3d_nb(
                f,
                rho,
                u,
                v,
                w_vel,
                lattice.w,
                lattice.cx,
                lattice.cy,
                lattice.cz,
                omega_base,
                self.cs,
                lattice.n_velocities,
                f.shape[1],
                f.shape[2],
                f.shape[3],
            )
        else:
            _wale_collide_2d_nb(
                f,
                rho,
                u,
                v,
                lattice.w,
                lattice.cx,
                lattice.cy,
                omega_base,
                self.cs,
                f.shape[1],
                f.shape[2],
            )
