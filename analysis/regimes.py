from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.physics import (
    characteristic_length,
    drive_velocity,
    reynolds_number,
    strouhal_number,
)
from engines.base import SimEngine
from scene.probe import Probe
from scene.scene import Scene


@dataclass
class FlowRegime:
    label: str
    confidence: float
    explanation: str
    strouhal: float | None = None
    oscillation: float = 0.0
    max_vorticity: float = 0.0


def _max_vorticity(sim: SimEngine) -> float:
    # The one definition of the curl, on the engine (PLAN.md C38). This file
    # carried a seventh hand-written copy of `dv/dx - du/dy`.
    return float(np.abs(sim.vorticity()).max())


def _probe_oscillation(
    probes: list[Probe], diameter: float = 1.0, velocity: float = 1.0
) -> tuple[float, float | None]:
    """Wake oscillation amplitude and Strouhal number, from the first probe.

    `diameter` and `velocity` are the scene's characteristic length and the
    velocity actually driving the flow. They were left at the function defaults of
    1.0, which made the returned "Strouhal number" the raw oscillation frequency in
    cycles per step -- a different quantity wearing the same name, and one the
    panel (which does pass D and U) disagreed with by a factor of D/U. For the
    cylinder preset that is 24/0.08 = 300, and `scorecard` compares the result
    against the preset's expected St range of [0.12, 0.28], so the check could not
    pass for any scene at any setting. Measured on that preset at step 4000:
    regime.strouhal = 0.00977, verdict `pass=False`, while the same probe data
    through the panel's call gives 2.93.
    """
    if not probes:
        return 0.0, None
    v_history = probes[0].history.get("v", [])
    if len(v_history) < 32:
        return 0.0, None
    recent = np.asarray(list(v_history)[-256:], dtype=float)
    amplitude = float(np.std(recent))
    # The recorded step cadence, not a declared dt=1.0: the panel samples every
    # second tick, so dt=1.0 reported a frequency twice too high (PLAN.md C39).
    return amplitude, strouhal_number(
        v_history,
        dt=1.0,
        diameter=diameter,
        velocity=velocity,
        times=probes[0].times(),
    )


def detect_flow_regime(
    sim: SimEngine,
    scene: Scene,
    probes: list[Probe] | None = None,
    step_count: int = 0,
) -> FlowRegime:
    probes = probes or []

    if drive_velocity(sim) <= 0:
        return FlowRegime(
            "No driven flow",
            0.95,
            (
                "Neither the inlet nor the lid is moving, so there is no "
                "sustained flow to classify."
            ),
        )

    length = characteristic_length(scene)
    re = reynolds_number(sim, obstacle_diameter=length)
    vort = _max_vorticity(sim)
    oscillation, st = _probe_oscillation(
        probes, diameter=length, velocity=drive_velocity(sim)
    )

    if step_count < 250:
        return FlowRegime(
            "Developing flow",
            0.7,
            "The flow is still spinning up; run longer before trusting Cd or St.",
            st,
            oscillation,
            vort,
        )
    if not scene.obstacles:
        return FlowRegime(
            "Open channel flow",
            0.8,
            "No obstacle is present, so expect a mostly steady channel-like field.",
            st,
            oscillation,
            vort,
        )
    if st is not None and oscillation > 1e-4 and 80 <= re <= 1000:
        return FlowRegime(
            "Periodic vortex shedding",
            0.85,
            (
                "The wake probe shows a repeating cross-flow signal, "
                "consistent with a Karman street."
            ),
            st,
            oscillation,
            vort,
        )
    if re < 40:
        return FlowRegime(
            "Steady creeping/laminar wake",
            0.75,
            "At this Reynolds number the wake is expected to stay mostly steady.",
            st,
            oscillation,
            vort,
        )
    if 40 <= re < 80:
        return FlowRegime(
            "Separated laminar wake",
            0.65,
            (
                "The Reynolds number is near the onset of wake separation; "
                "shedding may be weak or slow."
            ),
            st,
            oscillation,
            vort,
        )
    return FlowRegime(
        "Unsteady qualitative wake",
        0.55,
        (
            "The flow is unsteady or not yet cleanly periodic; "
            "use it for intuition unless the probe settles."
        ),
        st,
        oscillation,
        vort,
    )
