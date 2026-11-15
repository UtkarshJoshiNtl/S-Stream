"""Run presets: the resolution/step trade-offs for a solve.

``Preview`` is the default so a new model gets an answer in seconds;
``Final`` is the offline grid.  Presets are consumed by the CLI and by
``pipeline.run_case``; nothing here decides how a result looks.  The
presets name the streamwise resolution and step count only; the tunnel
cross-section follows from ``TunnelSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunPreset:
    """One resolution/steps pair for a solve."""

    name: str
    cells_streamwise: int
    steps: int
    description: str

    @property
    def label(self) -> str:
        return f"{self.name} · {self.cells_streamwise} cells"


PRESETS = (
    RunPreset("Preview", 128, 2200, "Fast placement check"),
    RunPreset("Final", 384, 12000, "Offline render grid; expect tens of minutes"),
)
