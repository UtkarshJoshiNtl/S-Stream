"""Re=100 sphere matrix with the steady tunnel and probe-normalised Cd.

Twelve cases: diameter D = 10, 20, 30 cells, Cs = 0 and 0.12, and the sphere
at x/L = 0.25 or 0.75.  Tunnel proportions match the earlier diagnosis
(L = 10D, W = H = 6D, sphere resting on the floor) so the measured Cd can be
compared with Johnson & Patel's 1.087 and with the old drift-contaminated
sweep in ``docs/diag_data/sweep_summary.csv``.

The two placements at a given D use *exactly the same step count*, so their
difference is purely placement.  The count scales with D (1000 steps per
diameter cell) to keep the same number of flow-throughs across resolutions,
which is what makes the D-trend interpretable.

Run from the repository root::

    .venv/bin/python simulation/scripts/diagnostics/steady_tunnel_sphere.py
    .venv/bin/python simulation/scripts/diagnostics/steady_tunnel_sphere.py --only D10_Cs0.12

Results go to ``docs/diag_data/steady_tunnel_sphere.csv`` and a markdown
summary with the target checks goes next to it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA_DIR,
    build_sphere_case,
    fmt,
    print_table,
    run_sphere,
    window_stats,
)
from common import SphereCase  # noqa: E402

REFERENCE_CD = 1.087
PLACEMENT_TOL = 0.03
FINAL_TOL = 0.05

DIAMETERS = (10, 20, 30)
CS_VALUES = (0.0, 0.12)
PLACEMENTS = (("x25", 2.5), ("x75", 7.5))
STEPS_PER_D = 1000


def case_key(d: int, cs: float, place: str) -> str:
    return f"D{d}_Cs{cs:g}_{place}"


def steps_for(d: int) -> int:
    return STEPS_PER_D * d


def build_case(d: int, cs: float, place: str, name: str) -> SphereCase:
    _, position_x_d = next(p for p in PLACEMENTS if p[0] == place)
    return build_sphere_case(
        case_key(d, cs, place),
        d_cells=d,
        smagorinsky=cs,
        position_x_d=position_x_d,
        note=f"steady tunnel; {name}",
    )


def main() -> None:
    global STEPS_PER_D
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated case keys")
    parser.add_argument("--list", action="store_true", help="print keys and exit")
    parser.add_argument("--steps-per-d", type=int, default=STEPS_PER_D)
    args = parser.parse_args()

    keys = [
        case_key(d, cs, place)
        for d in DIAMETERS
        for cs in CS_VALUES
        for place, _ in PLACEMENTS
    ]
    if args.list:
        print("\n".join(keys))
        return
    selected = [k for k in keys if not args.only or k in args.only.split(",")]
    if not selected:
        raise SystemExit(f"no cases matched --only {args.only!r}")

    STEPS_PER_D = args.steps_per_d

    rows = []
    for key in selected:
        d = int(key.split("_")[0][1:])
        cs = float(key.split("_")[1][2:])
        place = key.rsplit("_", 1)[1]
        case = build_case(d, cs, place, f"x/L={0.25 if place == 'x25' else 0.75}")
        steps = steps_for(d)
        trace, info = run_sphere(case, steps=steps, log_prefix=f"{key}: ")
        stats = window_stats(trace, fraction=0.2, min_samples=20, max_samples=80)
        last = trace.window(max(1, stats["n_samples"]))
        cy = float(np.mean(last.cy))
        cz = float(np.mean(last.cz))
        row = {
            "key": key,
            "D_cells": f"{case.diameter_cells:.3f}",
            "Cs": f"{cs:g}",
            "placement": place,
            "x_over_L": f"{0.25 if place == 'x25' else 0.75:g}",
            "grid": str(case.shape),
            "tau": f"{case.units.tau:.6f}",
            "steps": steps,
            "u_probe": f"{info['freestream_lattice']:.6f}",
            "u_probe_over_nominal": f"{info['freestream_lattice'] / case.units.u_lattice:.4f}",
            "cd": f"{stats['mean_cd']:.4f}",
            "cd_std": f"{stats['std_cd']:.4f}",
            "cd_change": f"{stats['change']:.4f}",
            "late_window": f"{stats['step_lo']}-{stats['step_hi']}",
            "cy": f"{cy:.4f}",
            "cz": f"{cz:.4f}",
            "finite": info["finite"],
        }
        rows.append(row)
        print(
            f"  {key}: Cd={stats['mean_cd']:.4f} std={stats['std_cd']:.4f} "
            f"change={stats['change']:.4f} Cy={cy:.4f} Cz={cz:.4f} "
            f"u_probe/U={row['u_probe_over_nominal']}",
            flush=True,
        )

    out_csv = DATA_DIR / "steady_tunnel_sphere.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv = not args.only
    if write_csv:
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    headers = [
        "case", "D", "Cs", "x/L", "steps", "u_probe/U", "Cd", "std", "change", "Cy", "Cz",
    ]
    print()
    print_table(
        [
            [
                r["key"],
                r["D_cells"],
                r["Cs"],
                r["x_over_L"],
                f"{int(r['steps']):,}",
                r["u_probe_over_nominal"],
                r["cd"],
                r["cd_std"],
                r["cd_change"],
                r["cy"],
                r["cz"],
            ]
            for r in rows
        ],
        headers,
    )

    if write_csv:
        assessment = assess(rows)
        md = DATA_DIR / "steady_tunnel_sphere.md"
        md.write_text(assessment, encoding="utf-8")
        print(f"\nwrote {out_csv}\nwrote {md}\n")
        print(assessment)


def assess(rows: list[dict]) -> str:
    """Compare placements and resolutions against the Step 4 targets."""
    by_key = {r["key"]: r for r in rows}
    lines = [
        "# Step 4 — sphere matrix with the steady tunnel",
        "",
        f"Reference Cd (Johnson & Patel, Re = 100): **{REFERENCE_CD:.3f}**. "
        "Cd uses the measured probe freestream; the tunnel has slip sides/roof, "
        "a no-slip floor and the fixed-density outlet.",
        "",
        "| D | Cs | Cd x/L=0.25 | Cd x/L=0.75 | placement diff | Cd mean at D | |Cd−1.087|/1.087 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    failures = []
    placement_results = {}
    for d in DIAMETERS:
        for cs in CS_VALUES:
            a = by_key.get(case_key(d, cs, "x25"))
            b = by_key.get(case_key(d, cs, "x75"))
            if a is None or b is None:
                continue
            ca, cb = float(a["cd"]), float(b["cd"])
            diff = abs(ca - cb) / max(abs((ca + cb) / 2), 1e-12)
            placement_results[(d, cs)] = diff
            lines.append(
                f"| {d} | {cs:g} | {ca:.4f} | {cb:.4f} | {diff:.2%} | "
                f"{(ca + cb) / 2:.4f} | "
                f"{abs((ca + cb) / 2 - REFERENCE_CD) / REFERENCE_CD:.2%} |"
            )
            if diff > PLACEMENT_TOL:
                failures.append(
                    f"placement at D={d}, Cs={cs:g} differs by {diff:.2%} "
                    f"(target < {PLACEMENT_TOL:.0%})"
                )

    lines += ["", "## Target checks", ""]
    for d in DIAMETERS:
        values = [
            float(by_key[case_key(d, cs, p)]["cd"])
            for cs in CS_VALUES
            for p, _ in PLACEMENTS
            if case_key(d, cs, p) in by_key
        ]
        if not values:
            continue
        mean_cd = float(np.mean(values))
        error = abs(mean_cd - REFERENCE_CD) / REFERENCE_CD
        lines.append(
            f"- D={d}: mean Cd = {mean_cd:.4f}, error vs 1.087 = {error:.2%}"
        )
        if d == 30 and error > FINAL_TOL:
            failures.append(
                f"D=30 Cd error {error:.2%} exceeds the {FINAL_TOL:.0%} target"
            )

    errors = {
        d: abs(
            float(np.mean([
                float(by_key[case_key(d, cs, p)]["cd"])
                for cs in CS_VALUES
                for p, _ in PLACEMENTS
                if case_key(d, cs, p) in by_key
            ])) - REFERENCE_CD
        )
        for d in DIAMETERS
        if any(case_key(d, cs, p) in by_key for cs in CS_VALUES for p, _ in PLACEMENTS)
    }
    if len(errors) == len(DIAMETERS):
        trend = all(
            errors[DIAMETERS[i + 1]] < errors[DIAMETERS[i]]
            for i in range(len(DIAMETERS) - 1)
        )
        lines.append(f"- Cd error decreases monotonically with D: **{trend}**")
        if not trend:
            failures.append("Cd does not approach 1.087 monotonically as D increases")

    lines.append("")
    if failures:
        lines.append("## GATE — FAILED")
        lines.append("")
        for item in failures:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Stop here; Step 5 is not run.")
    else:
        lines.append("## GATE — PASSED")
        lines.append("")
        lines.append("Both targets met; continue to Step 5.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
