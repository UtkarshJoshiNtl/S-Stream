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

    Columns: step, probe_name, u, v, speed, pressure
    Each probe gets its own block separated by a blank line.
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
            for i in range(n):
                vals = []
                for ff in available:
                    if i < len(hist[ff]):
                        vals.append(f"{hist[ff][i]:.6f}")
                    else:
                        vals.append("")
                f.write(f"{i}," + ",".join(vals) + "\n")
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
