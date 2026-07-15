from __future__ import annotations

from dataclasses import dataclass

from analysis.physics import characteristic_length, reynolds_number
from engines.base import SimEngine
from scene.probe import Probe
from scene.scene import Scene


@dataclass
class SanityWarning:
    level: str
    title: str
    message: str


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
