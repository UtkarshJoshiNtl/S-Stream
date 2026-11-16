"""D3Q19 lattice constants and the metres-to-lattice conversion.

The direction table is the canonical ordering, not a hand-typed one.  Six of
the diagonal ``z`` components are easy to flip by accident, and a flipped pair
breaks isotropy while still looking plausible.  The invariants in
``tests/test_lattice.py`` are what keep that honest:

* ``sum(w) == 1``
* ``sum(w * c_a * c_b) == cs2 * delta_ab``
* ``sum(w * c_x^4) == 1/3``
* the opposite mapping negates the direction and is its own inverse

Only NumPy is used here so the conversion can be tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

CS2 = 1.0 / 3.0

# Single SRT relaxation floor for the whole package.  At tau == 0.5 the
# collision operator vanishes and the scheme is unstable; the step kernel,
# the unit conversion and the config validation all use this value.
TAU_MIN = 0.5001

# Visualisation mode pins the relaxation time into this band and fixes
# u_lattice and the Smagorinsky coefficient, so the run is stable and the
# wake is lively regardless of the physical viscosity.
VISUAL_TAU_MIN = 0.505
VISUAL_TAU_MAX = 0.52
VISUAL_U_LATTICE = 0.06
VISUAL_SMAGORINSKY = 0.12
VISUAL_EFFECTIVE_NOTE = "Visual mode: tau pinned and parameters chosen for stability/visuals; Cd not calibrated."


def clamp_visual_tau(tau: float) -> float:
    """Pin a relaxation time into the visualisation band [0.505, 0.52]."""
    return min(max(float(tau), VISUAL_TAU_MIN), VISUAL_TAU_MAX)

# Rest, six face neighbours, twelve edge neighbours.
DIRECTIONS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
        (1, 0, -1),
        (-1, 0, 1),
    ],
    dtype=np.int32,
)

WEIGHTS = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12, dtype=np.float32)

OPPOSITE = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17],
    dtype=np.int32,
)


def equilibrium(ux, uy, uz, rho=1.0) -> np.ndarray:
    """Second-order equilibrium populations, shaped ``(... , 19)``.

    Used to initialise a run and to check the kernels in tests.  ``ux``/``uy``/
    ``uz``/``rho`` broadcast against each other; scalars are enough for a
    uniform start.
    """
    ux, uy, uz, rho = np.broadcast_arrays(
        np.asarray(ux, dtype=np.float32),
        np.asarray(uy, dtype=np.float32),
        np.asarray(uz, dtype=np.float32),
        np.asarray(rho, dtype=np.float32),
    )
    speed2 = ux * ux + uy * uy + uz * uz
    out = np.empty(ux.shape + (DIRECTIONS.shape[0],), dtype=np.float32)
    for i, (cx, cy, cz) in enumerate(DIRECTIONS):
        cu = cx * ux + cy * uy + cz * uz
        out[..., i] = WEIGHTS[i] * rho * (
            1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * speed2 / CS2
        )
    return out


@dataclass(frozen=True)
class LatticeUnits:
    """The bridge between SI units and the solver's dimensionless lattice.

    A lattice run is defined by cell size ``dx`` and time step ``dt``.  Picking
    the lattice speed instead of the time step keeps the Mach number low
    enough for the weakly compressible model: ``dt = u_lattice * dx / speed``.
    The viscosity then fixes the relaxation time through
    ``tau = 1/2 + nu_lattice / cs2``.
    """

    dx: float
    dt: float
    u_lattice: float
    tau: float
    smagorinsky: float = 0.0
    viscosity_clamped: bool = False

    @property
    def nu_lattice(self) -> float:
        return CS2 * (self.tau - 0.5)

    @property
    def nu_physical(self) -> float:
        return self.nu_lattice * self.dx * self.dx / self.dt

    def with_tau(self, tau: float) -> "LatticeUnits":
        """The same conversion with ``tau`` replaced and the clamp recorded.

        Visualisation mode pins tau into the stable band
        ``[VISUAL_TAU_MIN, VISUAL_TAU_MAX]``.  The formula chain is
        ``nu_lattice = cs² (tau - 1/2)`` with ``cs² = 1/3``,
        ``nu_physical = nu_lattice dx² / dt`` and
        ``dt = u_lattice dx / speed``, so a pinned tau means a known,
        larger effective viscosity and an effective Reynolds number
        ``speed * length / nu_physical`` that the report states openly.
        """
        return replace(self, tau=float(tau), viscosity_clamped=True)

    @classmethod
    def from_physical(
        cls,
        speed: float,
        kinematic_viscosity: float,
        dx: float,
        u_lattice: float = 0.05,
        smagorinsky: float = 0.0,
        min_tau: float = 0.0,
    ) -> LatticeUnits:
        """Convert SI values to lattice units.

        ``min_tau`` raises the relaxation time when the physical viscosity is
        too small to resolve on the grid.  The conversion is then no longer
        exact: the extra viscosity is the price of a stable coarse solve, and
        the caller should say so.  ``viscosity_clamped`` records that.
        """
        if speed <= 0.0:
            raise ValueError("speed must be positive")
        if kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive")
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        if not 0.0 < u_lattice < 0.2:
            raise ValueError("u_lattice must be small enough for low Mach numbers")

        dt = u_lattice * dx / speed
        nu_lattice = kinematic_viscosity * dt / (dx * dx)
        tau = 0.5 + nu_lattice / CS2
        floor = max(TAU_MIN, float(min_tau))
        clamped = tau < floor
        return cls(
            dx=dx,
            dt=dt,
            u_lattice=u_lattice,
            tau=max(tau, floor),
            smagorinsky=smagorinsky,
            viscosity_clamped=clamped,
        )
