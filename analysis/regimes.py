from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.physics import characteristic_length, reynolds_number, strouhal_number
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
    vel = sim.get_velocity()
    u = vel[:, :, 0]
    v = vel[:, :, 1]
    dvdx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
    return float(np.abs(dvdx - dudy).max())


def _probe_oscillation(probes: list[Probe]) -> tuple[float, float | None]:
    if not probes:
        return 0.0, None
    v_history = probes[0].history.get("v", [])
    if len(v_history) < 32:
        return 0.0, None
    recent = np.asarray(v_history[-256:], dtype=float)
    amplitude = float(np.std(recent))
    return amplitude, strouhal_number(v_history, dt=1.0)


def detect_flow_regime(
    sim: SimEngine,
    scene: Scene,
    probes: list[Probe] | None = None,
    step_count: int = 0,
) -> FlowRegime:
    probes = probes or []
    length = characteristic_length(scene)
    re = reynolds_number(sim, obstacle_diameter=length)
    vort = _max_vorticity(sim)
    oscillation, st = _probe_oscillation(probes)

    if sim.u_inflow <= 0:
        return FlowRegime(
            "No driven flow",
            0.95,
            "The inlet velocity is zero, so the simulation has no sustained flow.",
            st,
            oscillation,
            vort,
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
