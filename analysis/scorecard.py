from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.physics import characteristic_length, drag_coefficient, reynolds_number
from analysis.regimes import detect_flow_regime
from engines.base import SimEngine
from scene.probe import Probe
from scene.scene import Scene


@dataclass
class DesignScorecard:
    drag_coefficient: float
    reynolds_number: float
    wake_strength: float
    pressure_drop: float
    shedding_confidence: float
    summary: str

    def to_dict(self) -> dict:
        return {
            "drag_coefficient": self.drag_coefficient,
            "reynolds_number": self.reynolds_number,
            "wake_strength": self.wake_strength,
            "pressure_drop": self.pressure_drop,
            "shedding_confidence": self.shedding_confidence,
            "summary": self.summary,
        }


def compute_scorecard(
    sim: SimEngine,
    scene: Scene,
    probes: list[Probe] | None = None,
    step_count: int = 0,
) -> DesignScorecard:
    probes = probes or []
    vel = sim.get_velocity()
    speed = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
    rho = sim.get_density()
    pressure_drop = float(abs(np.mean(rho[:, 0]) - np.mean(rho[:, -1])))
    wake_strength = float(np.std(speed))
    regime = detect_flow_regime(sim, scene, probes, step_count)
    cd = drag_coefficient(sim)
    re = reynolds_number(sim, characteristic_length(scene))
    if not scene.obstacles:
        summary = "Open flow: add an obstacle to compare drag or wake designs."
    elif regime.strouhal is not None:
        summary = (
            "Wake has a measurable rhythm; "
            "this is a good candidate for a visual/report export."
        )
    else:
        summary = (
            "Flow is useful for intuition; "
            "run longer or add a wake probe for stronger metrics."
        )
    return DesignScorecard(
        drag_coefficient=float(cd),
        reynolds_number=float(re),
        wake_strength=wake_strength,
        pressure_drop=pressure_drop,
        shedding_confidence=(
            regime.confidence if "shedding" in regime.label.lower() else 0.0
        ),
        summary=summary,
    )
