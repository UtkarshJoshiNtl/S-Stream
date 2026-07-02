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
    if sim.u_inflow > 0.25:
        warnings.append(
            SanityWarning(
                "warn",
                "Fast inlet",
                (
                    "High lattice velocity can reduce physical trust. "
                    "Prefer lower inflow and tune viscosity for Re."
                ),
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
