"""Side-by-side setup audit for the two Re=100 sphere pytest cases.

No solver is run here.  The script rebuilds each pytest setup exactly, prints
the quantities the diagnosis needs, and recomputes the Reynolds number from
the lattice values.  The cylinder test is included as the third column because
its accepted Cd is part of the same question.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CS2,
    DATA_DIR,
    DEVICE,
    exact_test_cases,
    fmt,
    print_table,
)


def force_window(steps: int) -> dict:
    """Mirror ``pipeline.run_case``'s sampling schedule."""
    start = steps // 2
    sample_every = max(1, (steps - start) // 160)
    n = (steps - start) // sample_every + 1
    last = start + (n - 1) * sample_every
    return {
        "start": start,
        "sample_every": sample_every,
        "n_samples": n,
        "last": last,
        "covered": last - start + 1,
    }


def sphere_geometry(case) -> dict:
    solid = np.asarray(case.mask.solid, dtype=bool)
    idx = np.argwhere(solid)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    centre_cells = 0.5 * (lo + hi)
    centre_x = case.tunnel.grid_origin[0] + centre_cells[0] * case.dx
    rear_x = centre_x + 0.5 * case.diameter_m
    front_x = centre_x - 0.5 * case.diameter_m
    return {
        "centre_x": centre_x,
        "front_x": front_x,
        "rear_x": rear_x,
        "x_over_L": (centre_x - case.tunnel.x_min) / case.tunnel.length,
        "upstream_d": (front_x - case.tunnel.x_min) / case.diameter_m,
        "downstream_d": (case.tunnel.x_max - rear_x) / case.diameter_m,
        "outlet_from_centre_d": (case.tunnel.x_max - centre_x) / case.diameter_m,
    }


def cylinder_audit() -> dict:
    nx, ny, nz, diameter = 80, 60, 4, 8.0
    speed, reynolds = 0.05, 100.0
    tau = 0.5 + 3.0 * speed * diameter / reynolds
    nu = CS2 * (tau - 0.5)
    re_lat = speed * diameter / nu
    steps = 30000
    sample_start = 10000
    sample_every = 4
    n = (steps - sample_start) // sample_every + 1
    xs = np.arange(nx)[:, None]
    ys = np.arange(ny)[None, :]
    solid_cells = int((((xs - nx * 0.28) ** 2 + (ys - ny * 0.5) ** 2) <= (diameter * 0.5) ** 2).sum()) * nz
    effective_ratio = math.sqrt(4.0 * (solid_cells / nz) / math.pi) / diameter
    return {
        "shape": (nx, ny, nz),
        "dx": 1.0,
        "diameter_cells": diameter,
        "u_lattice": speed,
        "tau": tau,
        "nu_lattice": nu,
        "smagorinsky": 0.0,
        "tunnel": "80 x 60 x 4 cells, periodic y and z, open x",
        "blockage": diameter / ny,
        "speed_si": speed,
        "viscosity_si": nu,
        "steps": steps,
        "window_start": sample_start,
        "window_last": sample_start + (n - 1) * sample_every,
        "sample_every": sample_every,
        "n_samples": n,
        "re_lattice": re_lat,
        "re_si": reynolds,
        "cd_reference_area": diameter * nz,
        "cd_scale": 2.0 / (speed * speed * diameter * nz),
        "solid_cells": solid_cells,
        "effective_ratio": effective_ratio,
        "x_over_L": nx * 0.28 / nx,
        "upstream_d": (nx * 0.28 - 0.5 * diameter) / diameter,
        "downstream_d": (nx - (nx * 0.28 + 0.5 * diameter)) / diameter,
        "outlet_from_centre_d": (nx - nx * 0.28) / diameter,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases = exact_test_cases()
    sphere_geo = [sphere_geometry(c) for c, _, _ in cases]
    sphere_cols = []
    for case, steps, note in cases:
        win = force_window(steps)
        sphere_cols.append(
            {
                "case": case,
                "steps": steps,
                "note": note,
                "win": win,
                "geo": sphere_geo[len(sphere_cols)],
            }
        )
    cyl = cylinder_audit()

    print("=" * 110)
    print("Re=100 sphere setup audit (lattice values, then SI)")
    print(f"device: {DEVICE}")
    print("=" * 110)

    headers = ["quantity"] + [f"{c['case'].label}" for c in sphere_cols] + ["test_cylinder"]
    rows: list[list[str]] = []

    def add(name, values):
        rows.append([name] + [fmt(v) if not isinstance(v, str) else v for v in values])

    add("grid shape (nx, ny, nz)", [str(c["case"].shape) for c in sphere_cols] + [str(cyl["shape"])])
    add("dx (m)", [c["case"].dx for c in sphere_cols] + [cyl["dx"]])
    add("sphere D bbox mean (m)", [c["case"].diameter_m for c in sphere_cols] + ["-" for _ in [0]])
    add("D bbox mean (cells)", [c["case"].diameter_cells for c in sphere_cols] + [cyl["diameter_cells"]])
    add("D bbox x extent (cells)", [c["case"].diameter_cells_x for c in sphere_cols] + ["-"])
    add("u_lattice", [c["case"].units.u_lattice for c in sphere_cols] + [cyl["u_lattice"]])
    add("tau (molecular)", [c["case"].units.tau for c in sphere_cols] + [cyl["tau"]])
    add("nu_lattice = cs2(tau-1/2)", [c["case"].units.nu_lattice for c in sphere_cols] + [cyl["nu_lattice"]])
    add("Smagorinsky Cs", [c["case"].smagorinsky for c in sphere_cols] + [cyl["smagorinsky"]])
    add(
        "tunnel L x W x H (m)",
        [
            f"{c['case'].tunnel.length:g} x {c['case'].tunnel.width:g} x {c['case'].tunnel.height:g}"
            for c in sphere_cols
        ]
        + [cyl["tunnel"]],
    )
    add("silhouette blockage", [c["case"].blockage for c in sphere_cols] + [cyl["blockage"]])
    add("inlet speed SI (m/s)", [c["case"].fluid.speed for c in sphere_cols] + [cyl["speed_si"]])
    add("viscosity SI (m^2/s)", [c["case"].fluid.kinematic_viscosity for c in sphere_cols] + [cyl["viscosity_si"]])
    add("density SI (kg/m^3)", [c["case"].fluid.density for c in sphere_cols] + ["1 (lattice)"])
    add("steps run", [c["steps"] for c in sphere_cols] + [cyl["steps"]])
    add("force-window start step", [c["win"]["start"] for c in sphere_cols] + [cyl["window_start"]])
    add("force-window last step", [c["win"]["last"] for c in sphere_cols] + [cyl["window_last"]])
    add("force sample interval (steps)", [c["win"]["sample_every"] for c in sphere_cols] + [cyl["sample_every"]])
    add("force samples in window", [c["win"]["n_samples"] for c in sphere_cols] + [cyl["n_samples"]])
    add("Re = u_lat D_cells / nu_lat", [c["case"].reynolds_lattice for c in sphere_cols] + [cyl["re_lattice"]])
    add(
        "Re using bbox x extent",
        [c["case"].reynolds_lattice_x for c in sphere_cols] + ["-"],
    )
    add("Re from SI speed/D/nu", [c["case"].fluid.reynolds for c in sphere_cols] + [cyl["re_si"]])

    # Geometry / placement block.
    add("body x/L", [g["x_over_L"] for g in sphere_geo] + [cyl["x_over_L"]])
    add("upstream distance to body (D)", [g["upstream_d"] for g in sphere_geo] + [cyl["upstream_d"]])
    add("downstream distance from body (D)", [g["downstream_d"] for g in sphere_geo] + [cyl["downstream_d"]])
    add("outlet distance from centre (D)", [g["outlet_from_centre_d"] for g in sphere_geo] + [cyl["outlet_from_centre_d"]])
    add("solid cells", [c["case"].volume_cells for c in sphere_cols] + [cyl["solid_cells"]])
    add("effective D / bbox D", [c["case"].effective_diameter_ratio for c in sphere_cols] + [cyl["effective_ratio"]])
    add("frontal area used for Cd (cells^2)", [c["case"].area_lattice for c in sphere_cols] + [cyl["cd_reference_area"]])
    add("Cd force scale (2/(u^2 A))", [2.0 / (c["case"].units.u_lattice**2 * c["case"].area_lattice) for c in sphere_cols] + [cyl["cd_scale"]])

    print_table(rows, headers)
    print()
    print("Force-window note: the pipeline samples only step >= steps/2; the sphere table above")
    print("is the exact pytest schedule.  The cylinder samples step >= 10000 every 4 steps.")
    print()
    print("Re agreement (sphere tests): "
          f"{sphere_cols[0]['case'].reynolds_lattice:.6f} vs "
          f"{sphere_cols[1]['case'].reynolds_lattice:.6f} vs "
          f"{sphere_cols[2]['case'].reynolds_lattice:.6f}  -> "
          + ("AGREE by construction" if max(
              abs(c["case"].reynolds_lattice - sphere_cols[0]["case"].reynolds_lattice)
              for c in sphere_cols
          ) < 1e-6 else "DIFFER"))
    print()
    print("Case notes:")
    for c in sphere_cols:
        print(f"  - {c['case'].label}: {c['note']}; warnings={c['case'].build_warnings}")
    print()
    print("Cylinder note: 13.3% periodic confinement by diameter; the accepted Cd band is")
    print("1.35-1.85, not the unconfined 1.3-1.4.  The band is deliberately wide.")

    # Machine-readable copy for the report.
    out = DATA_DIR / "setup_audit.csv"
    import csv

    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
