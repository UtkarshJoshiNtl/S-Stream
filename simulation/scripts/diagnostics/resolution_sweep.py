"""Controlled Re=100 sphere sweep: Smagorinsky Cs and resolution.

One script, one table.  The base sweep holds the tunnel proportions and the
sphere placement fixed while varying:

* ``Cs`` = 0 and 0.12
* bounding-box sphere diameter = 10, 20, 30, 40 cells, with the domain scaled
  by the same factor (L = 10D, W = H = 6D, sphere at x = 2.5D)

The same sphere STL, the same floor placement, the same ``u_lattice = 0.05``
and Re = 100 are used throughout.  At ``u_lattice=0.05`` the molecular
relaxation time is ``tau = 0.5 + 0.0015 D_cells``; every requested resolution
therefore has ``tau > 0.5001``.  D=40 at the test proportions needs ~4.3 GB of
solver arrays and does not fit the 4 GiB RTX 2050; it is reported as an
impossible case rather than silently reduced.

Extra rows isolate the two pytest differences without changing the physics:

* exact pytest geometry (D ~ 9.58 cells) at x = 3.0 m and x = 1.0 m
* the x = 7.5D placement at D=10 and D=20
* a 20D-long tunnel with the sphere still at x = 7.5D

Run from the repository root::

    .venv/bin/python simulation/scripts/diagnostics/resolution_sweep.py

Use ``--only KEY[,KEY]`` to run a subset and ``--list`` to print the keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA_DIR,
    DEVICE,
    build_sphere_case,
    build_test_sphere_case,
    fmt,
    plot_trace,
    print_table,
    recirculation_length,
    run_sphere,
    window_stats,
    write_trace_csv,
)
from common import SphereCase  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sstream.schema import TunnelSpec  # noqa: E402

FIELDS_DIR = Path("/tmp/opencode/sstream_diag")


@dataclass
class RunSpec:
    key: str
    group: str
    d_cells: float
    cs: float
    length_d: float = 10.0
    width_d: float = 6.0
    height_d: float = 6.0
    position_x_d: float = 2.5
    position_z_d: float = 0.0
    steps: int = 0
    note: str = ""
    exact: bool = False
    position_x_m: float | None = None
    base_length: float | None = None
    base_cells: int | None = None
    tunnel_length: float | None = None
    tunnel_cells: int | None = None


def flow_through_steps(length_d: float, u_lattice: float = 0.05) -> float:
    return length_d / u_lattice


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []

    # Exact pytest geometry, rerun with a common step count and a dense trace.
    specs += [
        RunSpec(
            key="exact_x3p0",
            group="pytest geometry",
            d_cells=9.5795,
            cs=0.12,
            steps=12000,
            exact=True,
            position_x_m=3.0,
            tunnel_length=4.0,
            tunnel_cells=96,
            note="test_sphere_re100 placement (x/L=0.75, 2.0D behind the sphere)",
        ),
        RunSpec(
            key="exact_x1p0_L4",
            group="pytest geometry",
            d_cells=9.5795,
            cs=0.12,
            steps=12000,
            exact=True,
            position_x_m=1.0,
            tunnel_length=4.0,
            tunnel_cells=96,
            note="test_outlet_reflection 1x placement (x/L=0.25, 7.0D behind)",
        ),
        RunSpec(
            key="exact_x1p0_L6",
            group="pytest geometry",
            d_cells=9.5795,
            cs=0.12,
            steps=12000,
            exact=True,
            position_x_m=1.0,
            tunnel_length=6.0,
            tunnel_cells=144,
            base_length=4.0,
            base_cells=96,
            note="test_outlet_reflection 2x placement, same dx and mesh",
        ),
        RunSpec(
            key="exact_x3p0_8000",
            group="pytest timing",
            d_cells=9.5795,
            cs=0.12,
            steps=8000,
            exact=True,
            position_x_m=3.0,
            tunnel_length=4.0,
            tunnel_cells=96,
            note="test-1 placement at test-2 run length (8000 steps)",
        ),
        RunSpec(
            key="exact_x1p0_L4_8000",
            group="pytest timing",
            d_cells=9.5795,
            cs=0.12,
            steps=8000,
            exact=True,
            position_x_m=1.0,
            tunnel_length=4.0,
            tunnel_cells=96,
            note="test-2 placement and run length (8000 steps), the actual pytest case",
        ),
    ]

    # Base sweep: Cs x D.  Five flow-throughs at every resolution.
    for d_cells in (10, 20, 30, 40):
        for cs in (0.0, 0.12):
            steps = int(round(5.0 * flow_through_steps(10.0 * d_cells)))
            specs.append(
                RunSpec(
                    key=f"sweep_D{d_cells}_Cs{cs:g}",
                    group="base sweep",
                    d_cells=d_cells,
                    cs=cs,
                    length_d=10.0,
                    width_d=6.0,
                    height_d=6.0,
                    position_x_d=2.5,
                    steps=steps,
                    note="L=10D, W=H=6D, x=2.5D, floor",
                )
            )

    # Placement/outlet controls at fixed D and Cs.
    specs += [
        RunSpec(
            key="place_D10_x7p5",
            group="placement",
            d_cells=10,
            cs=0.12,
            steps=12000,
            position_x_d=7.5,
            note="test-1-like outlet distance at D=10 (2.0D downstream)",
        ),
        RunSpec(
            key="place_D20_x7p5",
            group="placement",
            d_cells=20,
            cs=0.12,
            steps=20000,
            position_x_d=7.5,
            note="test-1-like outlet distance at D=20 (2.0D downstream)",
        ),
        RunSpec(
            key="place_D10_x7p5_L20",
            group="placement",
            d_cells=10,
            cs=0.12,
            length_d=20.0,
            steps=20000,
            position_x_d=7.5,
            note="same D=10 placement with 12.0D of outlet instead of 2.0D",
        ),
        RunSpec(
            key="place_D10_x2p5_L20",
            group="placement",
            d_cells=10,
            cs=0.12,
            length_d=20.0,
            steps=20000,
            position_x_d=2.5,
            note="short upstream fetch (2.0D) with 17.5D of outlet",
        ),
        RunSpec(
            key="place_D10_x12p5_L20",
            group="placement",
            d_cells=10,
            cs=0.12,
            length_d=20.0,
            steps=20000,
            position_x_d=12.5,
            note="long upstream fetch (12.0D) with 7.5D of outlet",
        ),
        RunSpec(
            key="free_D20_Cs0.12",
            group="free-sphere control",
            d_cells=20,
            cs=0.12,
            steps=20000,
            position_x_d=2.5,
            position_z_d=2.5,
            note="same D=20 and proportions as sweep, sphere centred in Z (no floor contact)",
        ),
    ]
    return specs


def estimate_bytes(spec: RunSpec) -> int:
    nx = int(round(spec.length_d * spec.d_cells))
    ny = int(round(spec.width_d * spec.d_cells))
    nz = int(round(spec.height_d * spec.d_cells))
    cells = nx * ny * nz
    # f + f_new (2*19), rho (1), u (3), force (3), solid (1): 4 bytes each.
    return 4 * cells * (2 * 19 + 1 + 3 + 3 + 1)


def device_limit() -> int | None:
    try:
        import warp as wp

        devices = wp.get_cuda_devices()
        if devices:
            return int(devices[0].total_memory)
    except Exception:
        pass
    return None


def condition_number(spec: RunSpec) -> dict:
    """Analytic lattice parameters for the requested resolution."""
    d = spec.d_cells
    u = 0.05
    re = 100.0
    nu = u * d / re
    tau = 0.5 + nu / (1.0 / 3.0)
    return {
        "d_cells": d,
        "u_lattice": u,
        "nu_lattice": nu,
        "tau": tau,
        "tau_ok": tau > 0.5001,
        "re_lattice": u * d / nu,
    }


def run_spec(spec: RunSpec, *, save_crop: Path | None = None) -> dict | None:
    cond = condition_number(spec)
    if not cond["tau_ok"]:
        return {
            "key": spec.key,
            "group": spec.group,
            "note": spec.note,
            "status": "impossible: tau <= 0.5001",
            **cond,
        }

    if spec.exact:
        tunnel = TunnelSpec(
            length=float(spec.tunnel_length),
            width=2.4,
            height=2.4,
            floor_z=0.0,
            cells_streamwise=int(spec.tunnel_cells),
        )
        base = None
        if spec.base_length is not None:
            base = TunnelSpec(
                length=float(spec.base_length),
                width=2.4,
                height=2.4,
                floor_z=0.0,
                cells_streamwise=int(spec.base_cells),
            )
        case = build_test_sphere_case(
            spec.key,
            tunnel=tunnel,
            position_x=spec.position_x_m,
            smagorinsky=spec.cs,
            base_tunnel=base,
            note=spec.note,
        )
    else:
        case = build_sphere_case(
            spec.key,
            d_cells=spec.d_cells,
            smagorinsky=spec.cs,
            length_d=spec.length_d,
            width_d=spec.width_d,
            height_d=spec.height_d,
            position_x_d=spec.position_x_d,
            position_z_d=spec.position_z_d,
            note=spec.note,
        )

    # Memory guard for the 4 GiB GPU.
    nbytes = estimate_bytes(spec)
    limit = device_limit()
    if limit is not None and nbytes > 0.90 * limit:
        return {
            "key": spec.key,
            "group": spec.group,
            "note": spec.note,
            "status": "impossible: solver arrays exceed 90% of device memory",
            "requested_bytes": nbytes,
            "device_bytes": limit,
            "grid": case.shape,
            "d_bbox_cells": case.diameter_cells,
            "tau": case.units.tau,
            "re_lattice": case.reynolds_lattice,
        }

    print(
        f"\n=== {spec.key} ({spec.group}) ===\n"
        f"  grid {case.shape}, D={case.diameter_cells:.3f} cells, "
        f"tau={case.units.tau:.6f}, Cs={spec.cs:g}, "
        f"estimated solver memory {nbytes / 1e9:.2f} GB",
        flush=True,
    )
    trace, info = run_sphere(
        case,
        steps=spec.steps,
        save_crop=save_crop,
        log_prefix="  ",
    )

    stats = window_stats(trace)
    half = trace.between(spec.steps // 2, spec.steps)
    cy_mean = float(np.mean(trace.cy[-stats["n_samples"] :])) if len(trace) else float("nan")
    cz_mean = float(np.mean(trace.cz[-stats["n_samples"] :])) if len(trace) else float("nan")
    recirc0 = recirculation_length(case, info["velocity"], threshold=0.0)
    recirc1 = recirculation_length(case, info["velocity"], threshold=0.01 * case.units.u_lattice)

    label = f"sweep_{spec.key}"
    csv_path = write_trace_csv(DATA_DIR / f"{label}.csv", trace)
    plot_trace(DATA_DIR / f"{label}.png", trace, case, stats)

    solid_idx = np.argwhere(np.asarray(case.mask.solid, dtype=bool))
    centre_x = case.tunnel.grid_origin[0] + 0.5 * (
        solid_idx[:, 0].min() + solid_idx[:, 0].max()
    ) * case.dx
    actual_position_x_d = (centre_x - case.tunnel.x_min) / case.diameter_m
    downstream_d = (
        case.tunnel.x_max - (centre_x + 0.5 * case.diameter_m)
    ) / case.diameter_m
    return {
        "key": spec.key,
        "group": spec.group,
        "note": spec.note,
        "status": "ok" if info["finite"] else "failed: non-finite force/velocity",
        "grid": case.shape,
        "cells": int(np.prod(case.shape)),
        "dx": case.dx,
        "d_bbox_cells": case.diameter_cells,
        "d_eff_cells": case.effective_diameter_cells,
        "d_eff_ratio": case.effective_diameter_ratio,
        "blockage": case.blockage,
        "area_lattice": case.area_lattice,
        "cs": spec.cs,
        "tau": case.units.tau,
        "nu_lattice": case.units.nu_lattice,
        "u_lattice": case.units.u_lattice,
        "re_lattice": case.reynolds_lattice,
        "re_lattice_eff": case.units.u_lattice * case.effective_diameter_cells / case.units.nu_lattice,
        "tunnel_length_d": case.tunnel.length / case.diameter_m,
        "position_x_d": actual_position_x_d,
        "downstream_d": downstream_d,
        "steps_requested": spec.steps,
        "steps_completed": info["steps_completed"],
        "sample_every": info["sample_every"],
        "n_samples": len(trace),
        "elapsed_s": info["elapsed_s"],
        "mlups": info["mlups"],
        "max_speed": info["max_speed"],
        "cd_mean_pytest_window": float(np.mean(half.cd)) if len(half) else float("nan"),
        "n_pytest_window": len(half),
        "window_n": stats["n_samples"],
        "window_step_lo": stats["step_lo"],
        "window_step_hi": stats["step_hi"],
        "window_steps": stats["window_steps"],
        "cd_window_mean": stats["mean_cd"],
        "cd_window_std": stats["std_cd"],
        "cd_window_change": stats["change"],
        "cy_window_mean": cy_mean,
        "cz_window_mean": cz_mean,
        "recirc_cells": recirc0["length_cells"],
        "recirc_over_d": recirc0["length_d"],
        "recirc_over_d_1pct": recirc1["length_d"],
        "min_ux_centerline": recirc0["min_ux"],
        "bl_u_ref_over_nominal": info.get("bl_u_ref_over_nominal", float("nan")),
        "bl_u_centre_over_ref": info.get("bl_u_centre_over_ref", float("nan")),
        "bl_q_centre_over_nominal": info.get("bl_q_centre_over_nominal", float("nan")),
        "bl_delta99_d": info.get("bl_delta99_d", float("nan")),
        "bl_delta_star_d": info.get("bl_delta_star_d", float("nan")),
        "csv": str(csv_path),
        "png": str(DATA_DIR / f"{label}.png"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="comma-separated run keys")
    parser.add_argument("--list", action="store_true", help="print run keys and exit")
    parser.add_argument(
        "--fields-dir",
        default=str(FIELDS_DIR),
        help="where to save the control-volume crop for the finest case",
    )
    args = parser.parse_args()

    specs = build_specs()
    if args.list:
        for s in specs:
            print(f"{s.key:24s} {s.group:18s} D={s.d_cells:6.3f} Cs={s.cs:g} steps={s.steps}")
        return

    selected = specs
    if args.only:
        keys = {k.strip() for k in args.only.split(",") if k.strip()}
        selected = [s for s in specs if s.key in keys]
        missing = keys - {s.key for s in selected}
        if missing:
            raise SystemExit(f"unknown keys: {sorted(missing)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields_dir = Path(args.fields_dir)
    fields_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for spec in selected:
        # Save a control-volume crop for the finest successful base-sweep case.
        save_crop = None
        if spec.key == "sweep_D30_Cs0.12":
            save_crop = fields_dir / "fields_D30_Cs012.npz"
        elif spec.key == "free_D20_Cs0.12":
            save_crop = fields_dir / "fields_free_D20_Cs012.npz"
        try:
            result = run_spec(spec, save_crop=save_crop)
        except Exception as exc:  # keep going: one failed resolution must not lose the table
            result = {
                "key": spec.key,
                "group": spec.group,
                "note": spec.note,
                "status": f"failed: {type(exc).__name__}: {exc}",
            }
            print(f"  FAILED {spec.key}: {type(exc).__name__}: {exc}", flush=True)
        if result is not None:
            results.append(result)
        # Free the case/mask between runs; the solver goes out of scope inside
        # run_sphere, but the Python arrays can still be large.
        import gc

        gc.collect()

    # Table.
    headers = [
        "key",
        "group",
        "status",
        "grid",
        "D bbox",
        "D eff",
        "Cs",
        "tau",
        "Re",
        "blockage",
        "steps",
        "window steps",
        "Cd (pytest window)",
        "Cd (last window)",
        "std",
        "change",
        "Cy",
        "Cz",
        "Lr/D",
        "Lr/D 1%",
        "u_ref/u_nom",
        "u_c/u_ref",
        "q_c/q_nom",
        "delta*/D",
        "MLUPs",
    ]
    rows = []
    for r in results:
        if r.get("status") != "ok":
            head = [
                r.get("key", "?"),
                r.get("group", ""),
                r.get("status", ""),
                str(r.get("grid", "-")),
                fmt(r.get("d_bbox_cells", float("nan"))),
                fmt(r.get("d_eff_cells", float("nan"))),
                fmt(r.get("cs", float("nan")), 2),
                fmt(r.get("tau", float("nan")), 5),
                fmt(r.get("re_lattice", float("nan"))),
                fmt(r.get("blockage", float("nan"))),
            ]
            rows.append(head + ["-"] * (len(headers) - len(head)))
            continue
        rows.append(
            [
                r["key"],
                r["group"],
                r["status"],
                str(r["grid"]),
                fmt(r["d_bbox_cells"]),
                fmt(r["d_eff_cells"]),
                fmt(r["cs"], 2),
                fmt(r["tau"], 5),
                fmt(r["re_lattice"]),
                fmt(r["blockage"]),
                f"{r['steps_completed']:,}",
                f"{r['window_steps']:,}",
                fmt(r["cd_mean_pytest_window"]),
                fmt(r["cd_window_mean"]),
                fmt(r["cd_window_std"]),
                fmt(r["cd_window_change"], 5),
                fmt(r["cy_window_mean"], 2),
                fmt(r["cz_window_mean"], 2),
                fmt(r["recirc_over_d"]),
                fmt(r["recirc_over_d_1pct"]),
                fmt(r["bl_u_ref_over_nominal"]),
                fmt(r["bl_u_centre_over_ref"]),
                fmt(r["bl_q_centre_over_nominal"]),
                fmt(r["bl_delta_star_d"]),
                fmt(r["mlups"], 1),
            ]
        )
    print("\n" + "=" * 120)
    print("Controlled Re=100 sphere sweep")
    print("=" * 120)
    print_table(rows, headers)
    print()
    print("Cd (pytest window) = mean over the second half of the run, the pytest averaging rule.")
    print("Cd (last window)   = mean over the final 20% of samples (20-80 samples).")
    print("Lr/D               = attached centreline reverse-flow length / sphere diameter;")
    print("                     1% column uses u_x < 0.01 u_lattice.")

    summary_csv = DATA_DIR / "sweep_summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    summary_json = DATA_DIR / "sweep_summary.json"
    with summary_json.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {summary_csv}\nwrote {summary_json}")


if __name__ == "__main__":
    main()
