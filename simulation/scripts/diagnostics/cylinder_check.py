"""Why does the cylinder test accept Cd = 1.587 at Re = 100?

The pytest cylinder is a 2D (periodic in z) circular cylinder with only 60
cells across the periodic y direction and D = 8 cells, i.e. 13.3% confinement
by diameter.  This script repeats the same physical case at lower confinement
and at a finer resolution to separate blockage from discretisation error.

The incompressible benchmark for an unconfined cylinder at Re=100 is
Cd ~ 1.3-1.4.  The pytest band 1.35-1.85 is deliberately wide because the
test is a solver sanity check, not a blockage-free validation.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA_DIR, DEVICE, fmt, plot_trace, print_table, write_trace_csv  # noqa: E402
from common import Trace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sstream.lbm import LbmConfig, LbmSolver  # noqa: E402


def run_cylinder(diameter: int, ny: int, *, speed=0.05, reynolds=100.0, steps=30000, sample_start=10000, sample_every=4):
    nx = 10 * diameter
    nz = 4
    centre_x, centre_y = nx * 0.28, ny * 0.5
    xs = np.arange(nx)[:, None, None]
    ys = np.arange(ny)[None, :, None]
    solid = np.broadcast_to(
        (xs - centre_x) ** 2 + (ys - centre_y) ** 2 <= (diameter * 0.5) ** 2,
        (nx, ny, nz),
    ).copy()
    tau = 0.5 + 3.0 * speed * diameter / reynolds
    ref_area = diameter * nz
    scale = 2.0 / (speed * speed * ref_area)

    solver = LbmSolver(
        LbmConfig(
            shape=(nx, ny, nz),
            tau=tau,
            periodic=(False, True, True),
            inlet_speed=speed,
            device=DEVICE,
        ),
        solid,
    )
    solver.initialize(ux=speed)
    steps_out, cd_out, cy_out, cz_out = [], [], [], []
    for step in range(1, steps + 1):
        solver.step()
        if step >= sample_start and (step - sample_start) % sample_every == 0:
            force = solver.total_wall_force()
            steps_out.append(step)
            cd_out.append(float(force[0] * scale))
            cy_out.append(float(force[1] * scale))
            cz_out.append(float(force[2] * scale))
    trace = Trace(np.asarray(steps_out), np.asarray(cd_out), np.asarray(cy_out), np.asarray(cz_out))
    cd = float(trace.cd.mean())
    lift = trace.cy
    cl_amp = float(0.5 * (lift.max() - lift.min()))
    spectrum = np.abs(np.fft.rfft(lift - lift.mean()))
    freq = np.fft.rfftfreq(len(lift), d=sample_every)
    peak = float(freq[1:][np.argmax(spectrum[1:])])
    st = peak * diameter / speed
    return {
        "diameter": diameter,
        "ny": ny,
        "nx": nx,
        "nz": nz,
        "shape": (nx, ny, nz),
        "tau": tau,
        "blockage": diameter / ny,
        "cd": cd,
        "cd_std": float(trace.cd.std(ddof=1)),
        "cl_amplitude": cl_amp,
        "strouhal": st,
        "trace": trace,
        "max_speed": solver.max_speed(),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        (8, 60, "pytest case"),
        (8, 120, "same D, half confinement"),
        (8, 240, "same D, quarter confinement"),
        (16, 120, "finer D, 13.3% confinement"),
        (16, 240, "finer D, 6.7% confinement"),
        (16, 480, "finer D, 3.3% confinement"),
        (24, 180, "D=24, 13.3% confinement"),
        (24, 360, "D=24, 6.7% confinement"),
    ]
    results = []
    for diameter, ny, note in cases:
        print(f"running D={diameter} ny={ny} ({note}) ...", flush=True)
        r = run_cylinder(diameter, ny)
        r["note"] = note
        results.append(r)
        label = f"cylinder_D{diameter}_ny{ny}"
        write_trace_csv(DATA_DIR / f"{label}.csv", r["trace"])
        class _Case:
            label = f"Cylinder D={diameter} cells, ny={ny} ({note})"
        plot_trace(DATA_DIR / f"{label}.png", r["trace"], _Case())
        print(
            f"  Cd={r['cd']:.4f}  St={r['strouhal']:.4f}  "
            f"Cl_amp={r['cl_amplitude']:.4f}  blockage={r['blockage']:.1%}  "
            f"max|u|={r['max_speed']:.4f}",
            flush=True,
        )

    headers = ["case", "grid", "D (cells)", "confinement D/ny", "tau", "Cd", "Cd std", "St", "Cl amplitude", "max|u|"]
    rows = [
        [
            r["note"],
            str(r["shape"]),
            r["diameter"],
            fmt(r["blockage"]),
            fmt(r["tau"], 5),
            fmt(r["cd"]),
            fmt(r["cd_std"]),
            fmt(r["strouhal"]),
            fmt(r["cl_amplitude"]),
            fmt(r["max_speed"]),
        ]
        for r in results
    ]
    print()
    print_table(rows, headers)
    print()
    print("Unconfined Re=100 cylinder benchmark: Cd ~ 1.3-1.4, St ~ 0.165.")
    print("If Cd falls toward that band as confinement is removed, the pytest value 1.587")
    print("is the expected confined-cylinder number, not a solver failure.")

    out = DATA_DIR / "cylinder_summary.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
