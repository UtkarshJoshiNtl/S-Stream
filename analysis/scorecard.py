from __future__ import annotations

from dataclasses import dataclass, field

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
    range_checks: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "drag_coefficient": self.drag_coefficient,
            "reynolds_number": self.reynolds_number,
            "wake_strength": self.wake_strength,
            "pressure_drop": self.pressure_drop,
            "shedding_confidence": self.shedding_confidence,
            "summary": self.summary,
            "range_checks": self.range_checks,
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
    p = sim.get_pressure()
    pressure_drop = float(abs(np.mean(p[:, 0]) - np.mean(p[:, -1])))
    wake_strength = float(np.std(speed))
    regime = detect_flow_regime(sim, scene, probes, step_count)
    cd = drag_coefficient(sim)
    re = reynolds_number(sim, characteristic_length(scene))
    st = regime.strouhal

    expected = scene.product.expected_ranges
    range_checks: dict[str, dict] = {}
    summary_parts: list[str] = []

    for key, value in [("Re", re), ("Cd", cd), ("St", st)]:
        if key in expected and value is not None:
            lo, hi = expected[key]
            in_range = lo <= value <= hi
            range_checks[key] = {
                "value": value,
                "lo": lo,
                "hi": hi,
                "pass": in_range,
            }
            indicator = "within" if in_range else "outside"
            summary_parts.append(f"{key} {value:.2f} {indicator} [{lo:.1f}, {hi:.1f}]")

    if not scene.obstacles:
        summary = "Open flow: add an obstacle to compare drag or wake designs."
    elif summary_parts:
        summary = "; ".join(summary_parts)
    elif st is not None:
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
        range_checks=range_checks,
    )
