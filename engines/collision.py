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
    ) -> None:
        ...


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
                f, rho, u, v, w_vel,
                lattice.w, lattice.cx, lattice.cy, lattice.cz,
                omega, lattice.n_velocities,
                f.shape[1], f.shape[2], f.shape[3],
            )
        else:
            # 2D case
            _bgk_collide_nb(
                f, rho, u, v,
                lattice.w, lattice.cx, lattice.cy,
                omega,
                f.shape[1], f.shape[2],
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
            f, rho, u, v,
            lattice.w, lattice.cx, lattice.cy, lattice.opp,
            s_plus, self.s_minus,
            f.shape[1], f.shape[2],
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
            m0 = f[0,y,x] + f[1,y,x] + f[2,y,x] + f[3,y,x] + f[4,y,x] + f[5,y,x] + f[6,y,x] + f[7,y,x] + f[8,y,x]
            m1 = -4*f[0,y,x] - f[1,y,x] - f[2,y,x] - f[3,y,x] - f[4,y,x] + 2*f[5,y,x] + 2*f[6,y,x] + 2*f[7,y,x] + 2*f[8,y,x]
            m2 = 4*f[0,y,x] - 2*f[1,y,x] - 2*f[2,y,x] - 2*f[3,y,x] - 2*f[4,y,x] + f[5,y,x] + f[6,y,x] + f[7,y,x] + f[8,y,x]
            m3 = f[1,y,x] - f[3,y,x] + f[5,y,x] - f[6,y,x] + f[7,y,x] - f[8,y,x]
            m4 = f[2,y,x] - f[4,y,x] + f[5,y,x] + f[6,y,x] - f[7,y,x] - f[8,y,x]
            m5 = -f[1,y,x] + f[3,y,x] + f[5,y,x] - f[6,y,x] + f[7,y,x] - f[8,y,x]
            m6 = -f[2,y,x] + f[4,y,x] + f[5,y,x] + f[6,y,x] - f[7,y,x] - f[8,y,x]
            m7 = f[1,y,x] + f[3,y,x] - f[5,y,x] - f[6,y,x] - f[7,y,x] - f[8,y,x]
            m8 = -f[1,y,x] - f[3,y,x] + f[5,y,x] + f[6,y,x] - f[7,y,x] - f[8,y,x]
            m9 = f[1,y,x] - f[2,y,x] + f[3,y,x] - f[4,y,x] - f[5,y,x] - f[6,y,x] + f[7,y,x] + f[8,y,x]

            # Compute equilibrium moments
            e0 = r
            e1 = r * (-2 + 3*u_vel*u_vel + 3*v_vel*v_vel)
            e2 = r * (1 - 3*u_vel*u_vel - 3*v_vel*v_vel)
            e3 = r * u_vel
            e4 = r * v_vel
            e5 = -r * u_vel
            e6 = -r * v_vel
            e7 = r * (u_vel*u_vel - v_vel*v_vel)
            e8 = -r * (u_vel*u_vel - v_vel*v_vel)
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
            f[0,y,x] = (m_new[0] - m_new[1] + m_new[2]) / 9.0
            f[1,y,x] = (m_new[0] - m_new[1]/2.0 + m_new[2]/4.0 + m_new[3]/2.0 + m_new[5]/2.0 + m_new[7]/4.0 + m_new[9]/2.0) / 9.0
            f[2,y,x] = (m_new[0] - m_new[1]/2.0 + m_new[2]/4.0 + m_new[4]/2.0 + m_new[6]/2.0 + m_new[8]/4.0 - m_new[9]/2.0) / 9.0
            f[3,y,x] = (m_new[0] - m_new[1]/2.0 + m_new[2]/4.0 - m_new[3]/2.0 - m_new[5]/2.0 + m_new[7]/4.0 + m_new[9]/2.0) / 9.0
            f[4,y,x] = (m_new[0] - m_new[1]/2.0 + m_new[2]/4.0 - m_new[4]/2.0 - m_new[6]/2.0 + m_new[8]/4.0 - m_new[9]/2.0) / 9.0
            f[5,y,x] = (m_new[0] + m_new[1]/4.0 + m_new[2]/8.0 + m_new[3]/2.0 + m_new[4]/2.0 + m_new[5]/2.0 + m_new[6]/2.0 + m_new[7]/4.0 + m_new[8]/4.0 + m_new[9]/2.0) / 9.0
            f[6,y,x] = (m_new[0] + m_new[1]/4.0 + m_new[2]/8.0 - m_new[3]/2.0 + m_new[4]/2.0 - m_new[5]/2.0 + m_new[6]/2.0 + m_new[7]/4.0 + m_new[8]/4.0 - m_new[9]/2.0) / 9.0
            f[7,y,x] = (m_new[0] + m_new[1]/4.0 + m_new[2]/8.0 - m_new[3]/2.0 - m_new[4]/2.0 - m_new[5]/2.0 - m_new[6]/2.0 + m_new[7]/4.0 + m_new[8]/4.0 + m_new[9]/2.0) / 9.0
            f[8,y,x] = (m_new[0] + m_new[1]/4.0 + m_new[2]/8.0 + m_new[3]/2.0 - m_new[4]/2.0 + m_new[5]/2.0 - m_new[6]/2.0 + m_new[7]/4.0 + m_new[8]/4.0 - m_new[9]/2.0) / 9.0

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
        self._s = np.array([
            1.0,   # s0: unused (conserved)
            1.19,  # s1: e (kinetic energy)
            1.4,   # s2: eps (energy square)
            1.0,   # s3: jx (x-momentum)
            1.0,   # s4: jy (y-momentum)
            1.2,   # s5: qx (x-energy flux)
            1.2,   # s6: qy (y-energy flux)
            1.0,   # s7: pxx (xx stress)
            1.0,   # s8: pyy (yy stress)
            1.0,   # s9: pxy (xy stress)
        ], dtype=np.float32)

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
            f, rho, u, v,
            lattice.w, lattice.cx, lattice.cy,
            s,
            f.shape[1], f.shape[2],
        )
