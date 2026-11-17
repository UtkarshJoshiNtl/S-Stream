"""Empty-tunnel boundary-variant experiment (Step 1 of the tunnel fix).

The empty tunnel in :mod:`tunnel_drift` never reaches steady state: with the
current velocity inlet and zero-gradient outlet the mean density climbs and
the whole flow decelerates for the whole run.  This script isolates the two
suspects -- the transverse (y/z) walls and the outlet treatment -- by running
the *same* empty tunnel under five configurations:

    A  baseline: no-slip on every face, zero-gradient outlet
    B  y and z periodic (no side/top/floor walls), zero-gradient outlet
    C  free-slip (specular) y/z walls, zero-gradient outlet
    D  no-slip walls, fixed-density equilibrium outlet
    E  free-slip y/z walls, fixed-density equilibrium outlet
    F  the tunnel default chosen from A-E: free-slip sides and roof, no-slip
       floor, fixed-density outlet

Every variant uses the sphere-test grid, relaxation time, Smagorinsky
coefficient and inlet speed, and runs 20,000 steps.  State is logged every 100
steps from the populations themselves (not from the solver's cached fields,
which are written before the inlet/outlet kernel overwrites x=0 and x=nx-1).

Run from the repository root::

    .venv/bin/python simulation/scripts/diagnostics/tunnel_variants.py

A short smoke test is available with ``--steps 1000``.  Output CSVs and a
summary table are written under ``docs/diag_data/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CS2, DATA_DIR, DEVICE, fmt, print_table  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sstream.lbm import LbmConfig, LbmSolver  # noqa: E402
from sstream.lattice import DIRECTIONS  # noqa: E402

SHAPE = (96, 58, 58)
TAU = 0.514369
SMAGORINSKY = 0.12
U_LATTICE = 0.05
STEPS = 20_000
SAMPLE_EVERY = 100

_FREE_YZ = {face: "free_slip" for face in ("y0", "y1", "z0", "z1")}
_TUNNEL = {"y0": "free_slip", "y1": "free_slip", "z0": "no_slip", "z1": "free_slip"}

VARIANTS = {
    "A": {
        "description": "baseline: no-slip walls, zero-gradient outlet",
        "periodic": (False, False, False),
        "wall_modes": {},
        "outlet_mode": "zero_gradient",
    },
    "B": {
        "description": "y/z periodic (no transverse walls), zero-gradient outlet",
        "periodic": (False, True, True),
        "wall_modes": {},
        "outlet_mode": "zero_gradient",
    },
    "C": {
        "description": "free-slip y/z walls, zero-gradient outlet",
        "periodic": (False, False, False),
        "wall_modes": _FREE_YZ,
        "outlet_mode": "zero_gradient",
    },
    "D": {
        "description": "no-slip walls, fixed-density equilibrium outlet",
        "periodic": (False, False, False),
        "wall_modes": {},
        "outlet_mode": "fixed_density",
    },
    "E": {
        "description": "free-slip y/z walls, fixed-density equilibrium outlet",
        "periodic": (False, False, False),
        "wall_modes": _FREE_YZ,
        "outlet_mode": "fixed_density",
    },
    "F": {
        "description": "tunnel default: slip sides/roof, no-slip floor, fixed-density outlet",
        "periodic": (False, False, False),
        "wall_modes": _TUNNEL,
        "outlet_mode": "fixed_density",
    },
}

FIELDS = (
    "step",
    "rho_mean",
    "speed_mean",
    "ux_mean",
    "ux_centerline",
    "speed_x25",
    "speed_x50",
    "speed_x75",
)


def _moments(solver: LbmSolver) -> tuple[np.ndarray, np.ndarray]:
    """Return density and velocity recomputed from the live populations."""
    f = np.asarray(solver.f.numpy(), dtype=np.float64)
    rho = f.sum(axis=0)
    vel = np.empty(rho.shape + (3,), dtype=np.float64)
    for k in range(3):
        vel[..., k] = np.einsum("i,ixyz->xyz", DIRECTIONS[:, k].astype(np.float64), f) / rho
    return rho, vel


def _sample(step: int, solver: LbmSolver) -> dict[str, float]:
    rho, vel = _moments(solver)
    speed = np.linalg.norm(vel, axis=-1)
    nx, ny, nz = solver.shape
    stations = (nx // 4, nx // 2, 3 * nx // 4)
    return {
        "step": step,
        "rho_mean": float(rho.mean()),
        "speed_mean": float(speed.mean()),
        "ux_mean": float(vel[..., 0].mean()),
        "ux_centerline": float(vel[:, ny // 2, nz // 2, 0].mean()),
        "speed_x25": float(speed[stations[0]].mean()),
        "speed_x50": float(speed[stations[1]].mean()),
        "speed_x75": float(speed[stations[2]].mean()),
    }


def run_variant(name: str, *, steps: int, sample_every: int) -> list[dict[str, float]]:
    spec = VARIANTS[name]
    config = LbmConfig(
        shape=SHAPE,
        tau=TAU,
        smagorinsky=SMAGORINSKY,
        periodic=spec["periodic"],
        inlet_speed=U_LATTICE,
        wall_modes=spec["wall_modes"],
        outlet_mode=spec["outlet_mode"],
        device=DEVICE,
    )
    solver = LbmSolver(config)
    solver.initialize(ux=U_LATTICE)
    rows: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        solver.step()
        if step % sample_every and step != steps:
            continue
        rows.append(_sample(step, solver))
        print(
            f"  {name} step {step:>6,}/{steps:,}  rho={rows[-1]['rho_mean']:.6f}  "
            f"|u|/U={rows[-1]['speed_mean'] / U_LATTICE:.4f}  "
            f"u_c/U={rows[-1]['ux_centerline'] / U_LATTICE:.4f}",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    mlups = SHAPE[0] * SHAPE[1] * SHAPE[2] * steps / max(elapsed, 1e-9) / 1e6
    print(f"  {name} done in {elapsed:.1f} s ({mlups:.1f} MLUPs)", flush=True)
    return rows


def summarize(name: str, rows: list[dict[str, float]]) -> dict[str, float | str]:
    """Final state and the largest drift after the 2,000-step transient."""
    late = [r for r in rows if r["step"] >= 2000] or rows[1:]
    baseline = next((r for r in late if r["step"] == 2000), late[0] if late else rows[0])
    final = rows[-1]

    def drift_pct(field: str) -> float:
        return 100.0 * max(abs(r[field] - baseline[field]) for r in late)

    drho = drift_pct("rho_mean")
    d_speed = drift_pct("speed_mean") / U_LATTICE
    d_center = drift_pct("ux_centerline") / U_LATTICE
    d_stations = max(drift_pct(f"speed_x{p}") / U_LATTICE for p in (25, 50, 75))
    return {
        "variant": name,
        "setup": VARIANTS[name]["description"],
        "final_rho": final["rho_mean"],
        "final_speed_fraction": final["speed_mean"] / U_LATTICE,
        "rho_drift_pct": drho,
        "speed_drift_pct": d_speed,
        "center_drift_pct": d_center,
        "station_drift_pct": d_stations,
        "max_drift_pct": max(drho, d_speed, d_center, d_stations),
        "drifts": "yes" if max(drho, d_speed, d_center, d_stations) > 2.0 else "no",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    parser.add_argument("--variants", default="ABCDE", help="subset of variants to run")
    args = parser.parse_args()

    print(
        f"empty-tunnel variants: grid={'x'.join(map(str, SHAPE))} tau={TAU:.6f} "
        f"Cs={SMAGORINSKY:g} U={U_LATTICE:g} steps={args.steps:,} "
        f"sample_every={args.sample_every} device={DEVICE}",
        flush=True,
    )
    summaries = []
    traces = {}
    for name in args.variants:
        print(f"variant {name}: {VARIANTS[name]['description']}", flush=True)
        rows = run_variant(name, steps=args.steps, sample_every=args.sample_every)
        traces[name] = rows
        _write_csv(DATA_DIR / f"tunnel_variant_{name}.csv", rows)
        summaries.append(summarize(name, rows))

    if set(args.variants) == set(VARIANTS):
        _write_csv(DATA_DIR / "tunnel_variant_summary.csv", summaries)
    headers = [
        "variant", "final rho", "final |u|/U", "rho drift %", "speed drift %",
        "center drift %", "station drift %", "max drift %", "drifts",
    ]
    table = [
        [
            s["variant"],
            fmt(s["final_rho"], 6),
            fmt(s["final_speed_fraction"], 4),
            fmt(s["rho_drift_pct"], 3),
            fmt(s["speed_drift_pct"], 3),
            fmt(s["center_drift_pct"], 3),
            fmt(s["station_drift_pct"], 3),
            fmt(s["max_drift_pct"], 3),
            s["drifts"],
        ]
        for s in summaries
    ]
    print()
    print_table(table, headers)

    if set(args.variants) == set(VARIANTS):
        _make_plot(traces)
        print(f"\nwrote variant traces and {DATA_DIR / 'tunnel_variant_summary.csv'}")
    else:
        print(f"\nwrote variant traces for {args.variants}")


def _make_plot(traces: dict[str, list[dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for name, rows in traces.items():
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["rho_mean"] for r in rows], label=name, lw=1.0)
        axes[1].plot(
            steps, [r["speed_mean"] / U_LATTICE for r in rows], label=name, lw=1.0
        )
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("mean density")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("mean |u| / U_nominal")
    axes[1].set_xlabel("lattice step")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=len(traces))
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    png = DATA_DIR / "tunnel_variants.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
