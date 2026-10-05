from __future__ import annotations

from pathlib import Path

import numpy as np

from engines.base import SimEngine
from scene.probe import Probe


def export_probe_csv(
    probes: list[Probe],
    path: str | Path,
) -> None:
    """Write probe time-series data to CSV.

    Columns: `step` followed by whichever fields the probe records; the probe's name
    and position are in the comment header above each block, and blocks are
    separated by a blank line.

    The `step` column holds the *recorded simulation step* (`Probe.times`), not the
    sample index. Those differ whenever the sampling cadence is not 1, which is the
    normal case: a panel-cadence probe samples every second step, so its recorded
    steps are 2, 4, 6 ... and an exporter writing `0, 1, 2, 3 ...` in a column
    headed `step` halves the time axis of every exported figure and makes the file
    unable to reproduce the Strouhal number the report quotes beside it (PLAN.md
    C39 gave probes real timestamps; this was the last consumer still reading
    indices).
    """
    path = Path(path)
    with open(path, "w") as f:
        for probe in probes:
            hist = probe.history
            if not any(hist.values()):
                continue
            f.write(
                f"# Probe: {probe.spec.name}  "
                f"(x={probe.spec.x}, y={probe.spec.y})\n"
            )
            fields = probe.spec.fields
            available = [ff for ff in fields if ff in hist]
            f.write("step," + ",".join(available) + "\n")
            n = max(len(v) for v in hist.values()) if hist else 0
            steps = probe.times()
            for i in range(n):
                vals = []
                for ff in available:
                    if i < len(hist[ff]):
                        vals.append(f"{hist[ff][i]:.6f}")
                    else:
                        vals.append("")
                step_value = steps[i] if i < len(steps) else i
                f.write(f"{step_value}," + ",".join(vals) + "\n")
            f.write("\n")


def export_field_snapshot(
    sim: SimEngine,
    path: str | Path,
) -> None:
    """Save current velocity, density, and smoke fields as a .npz archive.

    Keys: u, v, rho, smoke
    """
    path = Path(path)
    vel = sim.get_velocity()
    rho = sim.get_density()
    smoke = sim.get_smoke()
    np.savez(
        path,
        u=vel[:, :, 0],
        v=vel[:, :, 1],
        rho=rho,
        smoke=smoke,
    )
