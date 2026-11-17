"""Empty-tunnel control: show that the test tunnel never reaches steady state.

The two Re=100 sphere tests use a velocity inlet (populations reset to
equilibrium at x=0) and a zero-gradient outlet (x=nx-1 copies nx-2).  This
script runs the *empty* tunnel with the same grid, relaxation time, Smagorinsky
coefficient, inlet speed and step count as the pytest sphere cases and records
the mean density, mean velocity and the streamwise mass flux at several
stations.  A steady tunnel would keep rho ~ 1 and u ~ u_lattice; instead the
mean density rises and the whole flow decelerates for the entire run.

That drift is independent of the sphere and is the reason the reported Cd,
which is normalised by the nominal u_lattice, keeps falling and depends on how
many steps a case was run for.

Run from the repository root::

    .venv/bin/python simulation/scripts/diagnostics/tunnel_drift.py

Optional arguments allow a reference field for ``force_balance.py``::

    .venv/bin/python simulation/scripts/diagnostics/tunnel_drift.py \
        --shape 100,60,60 --steps 10000 --tau 0.514968 \
        --output /tmp/opencode/sstream_diag/empty_D10.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CS2, DATA_DIR, DEVICE, fmt, print_table  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sstream.lbm import LbmConfig, LbmSolver  # noqa: E402
from sstream.lattice import DIRECTIONS  # noqa: E402

DEFAULT_SHAPE = (96, 58, 58)
DEFAULT_D_CELLS = 9.5795
DEFAULT_STEPS = 12000
DEFAULT_CS = 0.12
DEFAULT_U = 0.05
DEFAULT_RE = 100.0


def tau_for(d_cells: float, u: float = DEFAULT_U, re: float = DEFAULT_RE) -> float:
    nu = u * d_cells / re
    return 0.5 + nu / CS2


def run_empty(shape, steps, tau, cs, u, *, sample_every=200, output=None):
    solid = np.zeros(shape, dtype=bool)
    solver = LbmSolver(
        LbmConfig(
            shape=shape,
            tau=tau,
            smagorinsky=cs,
            periodic=(False, False, False),
            inlet_speed=u,
            device=DEVICE,
        ),
        solid,
    )
    solver.initialize(ux=u)
    cx = DIRECTIONS[:, 0]
    nx = shape[0]
    rows = []
    for step in range(1, steps + 1):
        solver.step()
        if step % sample_every and step != steps:
            continue
        f = solver.f.numpy()
        rho = solver.density()
        vel = solver.velocity()
        station = {}
        for x in (1, nx // 4, nx // 2, 3 * nx // 4, nx - 2):
            station[x] = float(np.mean(rho[x] * vel[x, :, :, 0]))
        rows.append(
            {
                "step": step,
                "rho_mean": float(rho.mean()),
                "u_mean": float(vel[..., 0].mean()),
                "flux_x0": float(np.mean(np.sum(f[:, 0] * cx[:, None, None], axis=0))),
                "flux_x1": station[1],
                "flux_mid": station[nx // 2],
                "flux_out": station[nx - 2],
                "q_over_nominal": (station[nx // 2] / u) ** 2 / float(np.mean(rho[nx // 2])),
            }
        )
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        vel = solver.velocity()
        rho = solver.density()
        np.savez_compressed(
            output,
            velocity=np.asarray(vel, dtype=np.float32),
            density=np.asarray(rho, dtype=np.float32),
            shape=np.asarray(shape, dtype=np.int64),
            tau=np.asarray([tau], dtype=np.float64),
            u_lattice=np.asarray([u], dtype=np.float64),
            smagorinsky=np.asarray([cs], dtype=np.float64),
            steps=np.asarray([steps], dtype=np.int64),
        )
        print(f"  wrote reference fields -> {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default=",".join(map(str, DEFAULT_SHAPE)))
    parser.add_argument("--d-cells", type=float, default=DEFAULT_D_CELLS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--cs", type=float, default=DEFAULT_CS)
    parser.add_argument("--u", type=float, default=DEFAULT_U)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--output", default=None, help="save final fields for force_balance.py")
    args = parser.parse_args()

    shape = tuple(int(v) for v in args.shape.split(","))
    tau = args.tau if args.tau is not None else tau_for(args.d_cells, args.u)
    print(
        f"empty tunnel: grid={shape} tau={tau:.6f} Cs={args.cs:g} "
        f"u_lattice={args.u:g} steps={args.steps:,}",
        flush=True,
    )
    rows = run_empty(
        shape,
        args.steps,
        tau,
        args.cs,
        args.u,
        sample_every=args.sample_every,
        output=args.output,
    )

    print()
    print_table(
        [
            [
                f"{r['step']:,}",
                fmt(r["rho_mean"], 5),
                fmt(r["u_mean"], 5),
                fmt(r["flux_x0"], 5),
                fmt(r["flux_x1"], 5),
                fmt(r["flux_mid"], 5),
                fmt(r["flux_out"], 5),
                fmt(r["q_over_nominal"], 4),
            ]
            for r in rows
        ],
        ["step", "rho mean", "u mean", "flux x=0", "flux x=1", "flux mid", "flux outlet", "q/q nominal"],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    steps = [r["step"] for r in rows]
    ax.plot(steps, [r["u_mean"] for r in rows], label="mean u")
    ax.axhline(args.u, color="k", ls="--", lw=1, label="nominal u_lattice")
    ax.set_ylabel("lattice velocity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2.plot(steps, [r["rho_mean"] for r in rows], label="mean rho")
    ax2.axhline(1.0, color="k", ls="--", lw=1, label="rho = 1")
    ax2.set_xlabel("lattice step")
    ax2.set_ylabel("density")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    png = DATA_DIR / "empty_tunnel_drift.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    csv_path = DATA_DIR / "empty_tunnel_drift.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}\nwrote {png}")


if __name__ == "__main__":
    main()
