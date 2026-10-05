from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.physics import characteristic_length, reynolds_number
from engines.base import SimEngine
from scene.probe import Probe
from scene.scene import Scene

# --- Divergence thresholds -----------------------------------------------------
# These are the *field* limits, not setup limits, and they are deliberately far
# outside anything a healthy run produces so that a trigger means something.
#
# MAX_LATTICE_SPEED = 0.5 is twice the existing "very fast inlet" danger line
# (u_inflow > 0.25, already called "outside the low-Mach regime for this solver"), so
# a *field* value that large cannot be a legitimate steady state. Measured onset on a
# 48x48 channel with an r=5 cylinder: u_inflow=0.5 trips it at step 55, 0.4 at step
# 230, and 0.3 does not trip within 600 steps.
#
# The density bounds are [1e-3, 10] around the reference rho0 = 1. Both ends were set
# by measurement, and the first attempt at the floor was wrong: 0.1 flagged the
# Shan-Chen liquid engine at step 0, because that engine is *designed* with a
# low-density vapour background -- measured rho = [0.01, 2.0] on a fresh 48x48, the
# 2.0 being the droplet. A "low density" bound that fires on a healthy engine's
# initial condition is worse than no bound. 1e-3 sits an order of magnitude below
# that vapour background and still catches zero and negative mass, which violate
# positive mass outright. The ceiling of 10 keeps 5x headroom over the droplet.
MAX_LATTICE_SPEED = 0.5
DENSITY_FLOOR = 1e-3
DENSITY_CEILING = 10.0


@dataclass
class SanityWarning:
    level: str
    title: str
    message: str


@dataclass(frozen=True)
class FieldHealth:
    """What the conserved fields look like right now.

    `check_sanity` used to inspect only *setup* -- omega, viscosity, inlet speed,
    obstacle size, probe count -- so every warning it could raise was already true at
    step 0 and stayed true until the scene was changed. A run that had already filled
    the lattice with NaN produced no warning at all, because a sane setup that has
    since diverged looks exactly like a sane setup that has not.
    """

    available: bool
    finite: bool
    max_speed: float
    min_density: float
    max_density: float
    bad_cells: int

    @property
    def diverged(self) -> bool:
        if not self.available:
            return False
        if not self.finite:
            return True
        if self.max_speed > MAX_LATTICE_SPEED:
            return True
        return (
            not (DENSITY_FLOOR <= self.min_density)
            or self.max_density > DENSITY_CEILING
        )

    def describe(self) -> str:
        if not self.available:
            return "This engine does not expose the density and velocity fields."
        if not self.finite:
            return (
                f"{self.bad_cells} lattice cells hold a non-finite value. "
                "The distribution has blown up; results from this point are "
                "meaningless."
            )
        if self.max_speed > MAX_LATTICE_SPEED:
            return (
                f"peak lattice speed {self.max_speed:.3f} exceeds "
                f"{MAX_LATTICE_SPEED} (a lattice speed of that size is already "
                "super-sonic for this solver). Lower the inlet speed or raise "
                "the viscosity."
            )
        return (
            f"density has left [{DENSITY_FLOOR:g}, {DENSITY_CEILING:g}]: "
            f"rho in [{self.min_density:.3g}, {self.max_density:.3g}]. "
            "LBM requires positive, bounded mass; this run is unstable."
        )


def inspect_field_state(sim: SimEngine) -> FieldHealth:
    """Read the conserved fields and judge whether the run is still physical.

    Read through the engine's public accessors, not its attributes, and that is not
    a style preference: reaching for `sim.rho` covers the CPU engines but **not** the
    multicomponent one, which keeps `rho1`/`rho2` and has no `rho` at all -- so the
    first version of this function reported `available=False` there and skipped the
    engine entirely, with no warning that it had done so. `get_density()` returns the
    total for that engine and means the same thing as `rho` for the others.

    Velocity is taken from `get_velocity()` and reduced over *all* components rather
    than through `speed_magnitude()`, which deliberately sums only the horizontal
    pair to match the 2D `speed` render. A vertical runaway is still a divergence, so
    the check must see `w` in 3D.

    Density and velocity only. Smoke is a passive scalar that may legitimately be
    zero everywhere or large in a dense puff, so bounding it would fire on healthy
    runs; temperature is bounded in physical terms but only the thermal engines
    carry it, and a bound there belongs to that engine's own checks.
    """
    try:
        rho = np.asarray(sim.get_density(), dtype=np.float64)
        vel = np.asarray(sim.get_velocity(), dtype=np.float64)
    except Exception:
        # An engine that cannot produce its conserved fields is not evidence of
        # divergence; reporting `available=False` leaves it unjudged rather than
        # accusing it.
        return FieldHealth(False, True, 0.0, 0.0, 0.0, 0)

    finite = bool(np.isfinite(rho).all()) and bool(np.isfinite(vel).all())
    bad_cells = int((~np.isfinite(rho)).sum()) + int((~np.isfinite(vel)).sum())
    max_speed = float(np.abs(vel).max()) if vel.size else 0.0
    if not np.isfinite(max_speed):
        max_speed = float("inf")
    if not np.isfinite(rho.min()) or not np.isfinite(rho.max()):
        min_density, max_density = 0.0, float("inf")
    else:
        min_density, max_density = float(rho.min()), float(rho.max())
    return FieldHealth(
        available=True,
        finite=finite,
        max_speed=max_speed,
        min_density=min_density,
        max_density=max_density,
        bad_cells=bad_cells,
    )


def check_sanity(
    sim: SimEngine,
    scene: Scene,
    probes: list[Probe] | None = None,
    step_count: int = 0,
) -> list[SanityWarning]:
    probes = probes or []
    warnings: list[SanityWarning] = []
    length = characteristic_length(scene)
    re = reynolds_number(sim, obstacle_diameter=length)

    # Field state first: once the lattice has blown up, every other message below is
    # beside the point, and a setup warning is the wrong explanation for it.
    health = inspect_field_state(sim)
    if health.diverged:
        warnings.append(
            SanityWarning(
                "danger",
                "Simulation diverged",
                health.describe(),
            )
        )

    omega = None
    if hasattr(sim, "omega"):
        try:
            omega = float(sim.omega)
        except Exception:
            omega = None
    if omega is not None and not (0.0 < omega < 2.0):
        warnings.append(
            SanityWarning(
                "danger",
                "Unstable relaxation",
                (
                    f"omega={omega:.3f} is outside (0, 2). "
                    "Adjust viscosity so the simulation stays stable."
                ),
            )
        )

    if sim.viscosity < 0.003:
        warnings.append(
            SanityWarning(
                "danger",
                "Low viscosity",
                (
                    "BGK LBM can become noisy at very low viscosity; "
                    "increase nu if the field sparkles or blows up."
                ),
            )
        )
    # Low-Mach envelope: U ≲ 0.05 preferred; warn above 0.1
    if sim.u_inflow > 0.1:
        warnings.append(
            SanityWarning(
                "warn",
                "Fast inlet (Mach)",
                (
                    "Lattice velocity above 0.1 hurts hydro accuracy. "
                    "Prefer U ≤ 0.05 and set viscosity from Re = U L / ν."
                ),
            )
        )
    if sim.u_inflow > 0.25:
        warnings.append(
            SanityWarning(
                "danger",
                "Very fast inlet",
                "u_inflow > 0.25 is outside the low-Mach regime for this solver.",
            )
        )
    if not scene.obstacles:
        warnings.append(
            SanityWarning(
                "info",
                "No obstacle",
                "Drag, wake, and shedding readouts need an obstacle to be meaningful.",
            )
        )
    if scene.obstacles and length < 8:
        warnings.append(
            SanityWarning(
                "warn",
                "Coarse obstacle",
                (
                    "The main obstacle is under-resolved. "
                    "Use a larger radius/shape for cleaner Cd and wake structure."
                ),
            )
        )
    if not probes:
        warnings.append(
            SanityWarning(
                "info",
                "No wake probe",
                (
                    "Place a probe behind the obstacle to estimate "
                    "Strouhal number and shedding confidence."
                ),
            )
        )
    if step_count < 1000 and re > 40:
        warnings.append(
            SanityWarning(
                "info",
                "Still developing",
                (
                    "Run longer before using Cd/St; vortex wakes often "
                    "need thousands of steps to settle."
                ),
            )
        )
    return warnings
