from __future__ import annotations

from dataclasses import dataclass

from analysis.physics import characteristic_length, drag_coefficient, reynolds_number
from analysis.regimes import FlowRegime
from analysis.sanity import SanityWarning
from engines.base import SimEngine
from scene.scene import Scene


@dataclass
class AISettings:
    enabled: bool = False
    api_key: str = ""
    model: str = "gemini-1.5-flash"


def build_ai_context(
    scene: Scene,
    sim: SimEngine,
    regime: FlowRegime | None = None,
    warnings: list[SanityWarning] | None = None,
    step_count: int = 0,
) -> str:
    warnings = warnings or []
    length = characteristic_length(scene)
    re = reynolds_number(sim, length)
    cd = drag_coefficient(sim)
    warning_lines = "\n".join(
        f"- {w.level.upper()}: {w.title} - {w.message}" for w in warnings
    ) or "- None"
    regime_text = (
        f"{regime.label} ({regime.confidence:.0%}): {regime.explanation}"
        if regime
        else "Not yet detected"
    )
    return f"""You are SStream's friendly fluid tutor.

Scene: {scene.name}
Description: {scene.description or "No description"}
Lesson: {scene.product.lesson_headline or "No lesson headline"}
Grid: {scene.width} x {scene.height}
Step: {step_count}
Inflow velocity: {sim.u_inflow}
Viscosity: {sim.viscosity}
Characteristic length: {length}
Reynolds number: {re:.2f}
Drag coefficient estimate: {cd:.3f}
Detected regime: {regime_text}
Warnings:
{warning_lines}

Explain what the user is seeing in plain language. Be honest that this is a
2D educational/intuition tool, not a replacement for validated CFD.
"""


def local_ai_response(context: str, has_api_key: bool) -> str:
    if has_api_key:
        return (
            "Gemini wiring is ready, but live API calls are disabled in this build. "
            "The prompt context below is what would be sent."
        )
    return (
        "AI tutor is in local preview mode. Add a Gemini API key later to enable "
        "live explanations. For now, use the outcome cards and prompt context."
    )
