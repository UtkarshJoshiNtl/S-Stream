"""Guo forcing for D2Q9 LBM (Guo, Zheng & Shi, 2002).

Lattice units assumed throughout:
    dt = 1
    cs2 = 1/3   (hence 1/cs2 = 3, 1/cs4 = 9)

Guo force recovers the Navier–Stokes equations at the discrete level:

    u' = u + F * dt / (2 * rho)          # equilibrium / force velocity
    F_i = (1 - omega/2) * w_i *
          (3*(c_i - u')·F + 9*(c_i·u')(c_i·F))

BGK collision uses the shifted velocity u'; the force term F_i is then
added to the post-collision populations.  Macroscopic velocity reported
to the caller should be the same shifted u' (already includes F dt / 2).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Documented lattice constants (dt=1, cs2=1/3).
DT: float = 1.0
CS2: float = 1.0 / 3.0


def guo_velocity_shift(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fx: np.ndarray | float,
    fy: np.ndarray | float,
    *,
    out_u: np.ndarray | None = None,
    out_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equilibrium velocity u' = u + F*dt/(2*rho).

    Accepts scalar or array force components.  If ``out_u`` / ``out_v`` are
    provided the shift is written in-place there; otherwise new arrays are
    allocated.  Inputs ``u``/``v`` are never mutated unless they are the
    ``out_*`` arrays.
    """
    half = np.float32(0.5 * DT)
    rho_safe = np.where(rho > 0.0, rho, np.float32(1.0))
    if out_u is None:
        out_u = np.empty_like(u)
    if out_v is None:
        out_v = np.empty_like(v)
    out_u[:] = u + half * np.asarray(fx, dtype=u.dtype) / rho_safe
    out_v[:] = v + half * np.asarray(fy, dtype=v.dtype) / rho_safe
    return out_u, out_v


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _guo_force_term_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u_eq: np.ndarray,
    v_eq: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega: float,
    height: int,
    width: int,
) -> None:
    """Add Guo force contribution to populations after BGK collision.

    ``u_eq``, ``v_eq`` must already be the shifted velocities.
    ``fx``, ``fy`` are force density fields (same shape as rho).
    Uses dt=1, cs2=1/3.
    """
    coef = 1.0 - 0.5 * omega
    for y in prange(height):
        for x in range(width):
            r = rho[y, x]
            if r <= 0.0:
                continue
            ux = u_eq[y, x]
            uy = v_eq[y, x]
            force_x = fx[y, x]
            force_y = fy[y, x]
            if force_x == 0.0 and force_y == 0.0:
                continue
            for i in range(9):
                cix = cx[i]
                ciy = cy[i]
                cu = cix * ux + ciy * uy
                cf = cix * force_x + ciy * force_y
                # 3*(c - u')·F + 9*(c·u')(c·F)
                term = (
                    3.0 * ((cix - ux) * force_x + (ciy - uy) * force_y) + 9.0 * cu * cf
                )
                f[i, y, x] += coef * w[i] * term


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _guo_force_term_const_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u_eq: np.ndarray,
    v_eq: np.ndarray,
    fx: float,
    fy: float,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega: float,
    height: int,
    width: int,
) -> None:
    """Like `_guo_force_term_nb` but with uniform body force (fx, fy)."""
    if fx == 0.0 and fy == 0.0:
        return
    coef = 1.0 - 0.5 * omega
    for y in prange(height):
        for x in range(width):
            r = rho[y, x]
            if r <= 0.0:
                continue
            ux = u_eq[y, x]
            uy = v_eq[y, x]
            for i in range(9):
                cix = cx[i]
                ciy = cy[i]
                cu = cix * ux + ciy * uy
                cf = cix * fx + ciy * fy
                term = 3.0 * ((cix - ux) * fx + (ciy - uy) * fy) + 9.0 * cu * cf
                f[i, y, x] += coef * w[i] * term


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _bgk_collide_guo_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega: float,
    height: int,
    width: int,
) -> None:
    """Combined BGK collision + Guo force (dt=1, cs2=1/3).

    Bare momentum from populations is shifted by F/(2ρ) for equilibrium and
    the force term; stored ``u``, ``v`` are those shifted velocities.
    """
    coef = 1.0 - 0.5 * omega
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
            rho_safe = r if r > 0.0 else 1.0
            force_x = fx[y, x]
            force_y = fy[y, x]
            # u' = u_bare + F * dt / (2 * rho), dt=1
            ux = mu / rho_safe + 0.5 * force_x / rho_safe
            uy = mv / rho_safe + 0.5 * force_y / rho_safe
            u2 = ux * ux + uy * uy
            for i in range(9):
                cix = cx[i]
                ciy = cy[i]
                cu = cix * ux + ciy * uy
                feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f_post = f[i, y, x] * (1.0 - omega) + feq * omega
                cf = cix * force_x + ciy * force_y
                term = (
                    3.0 * ((cix - ux) * force_x + (ciy - uy) * force_y) + 9.0 * cu * cf
                )
                f[i, y, x] = f_post + coef * w[i] * term
            rho[y, x] = r
            u[y, x] = ux
            v[y, x] = uy


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _bgk_collide_guo_const_nb(
    f: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fx: float,
    fy: float,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega: float,
    height: int,
    width: int,
) -> None:
    """Combined BGK + Guo force with uniform body force."""
    coef = 1.0 - 0.5 * omega
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
            rho_safe = r if r > 0.0 else 1.0
            ux = mu / rho_safe + 0.5 * fx / rho_safe
            uy = mv / rho_safe + 0.5 * fy / rho_safe
            u2 = ux * ux + uy * uy
            for i in range(9):
                cix = cx[i]
                ciy = cy[i]
                cu = cix * ux + ciy * uy
                feq = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f_post = f[i, y, x] * (1.0 - omega) + feq * omega
                cf = cix * fx + ciy * fy
                term = 3.0 * ((cix - ux) * fx + (ciy - uy) * fy) + 9.0 * cu * cf
                f[i, y, x] = f_post + coef * w[i] * term
            rho[y, x] = r
            u[y, x] = ux
            v[y, x] = uy


def apply_guo_force(
    f: np.ndarray,
    rho: np.ndarray,
    u_eq: np.ndarray,
    v_eq: np.ndarray,
    fx: np.ndarray | float,
    fy: np.ndarray | float,
    w: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    omega: float,
) -> None:
    """Python wrapper: add Guo force term to post-collision ``f``."""
    height, width = rho.shape
    if isinstance(fx, np.ndarray) or isinstance(fy, np.ndarray):
        fx_a = np.broadcast_to(np.asarray(fx, dtype=np.float32), rho.shape).copy()
        fy_a = np.broadcast_to(np.asarray(fy, dtype=np.float32), rho.shape).copy()
        _guo_force_term_nb(
            f, rho, u_eq, v_eq, fx_a, fy_a, w, cx, cy, omega, height, width
        )
    else:
        _guo_force_term_const_nb(
            f,
            rho,
            u_eq,
            v_eq,
            float(fx),
            float(fy),
            w,
            cx,
            cy,
            omega,
            height,
            width,
        )
