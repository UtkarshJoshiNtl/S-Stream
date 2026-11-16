"""Headless solver front end: one mesh in, one result out.

Usage (Windows or Linux, same command)::

    sstream car.obj --preset preview --speed 30 --out results

No window, no OpenGL, no Qt.  The solve runs on the GPU via Warp and the
outputs are a JSON summary plus the same plain-text report the old app
showed.  Everything is written under ``--out`` so the command is safe to
run from any working directory, including from a frozen bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .lbm import pick_device
from .mesh import DEFAULT_MODEL, build_model, load_surface
from .options import add_orientation_args, mesh_spec_from_args
from .snapshots import placement_dict
from .pipeline import SolvePlan, SolveCancelled, run_case
from .preview import PRESETS
from .quality import (
    check_vram,
    cells_streamwise_for_target,
    format_bytes,
    grid_cells,
    resolve_target_cells,
)
from .report import summary
from .schema import FluidSpec, TunnelSpec
from .voxelize import voxelize


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    presets = {p.name.lower(): p for p in PRESETS}
    parser = argparse.ArgumentParser(
        prog="sstream",
        description="S-Stream headless wind-tunnel solver (LBM, GPU).",
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        default=str(DEFAULT_MODEL),
        help=f"surface mesh to solve (.obj/.stl/.ply; default: bundled car model)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(presets),
        default="preview",
        help="resolution preset (default: preview)",
    )
    parser.add_argument("--speed", type=float, default=30.0, help="freestream m/s")
    parser.add_argument(
        "--viscosity", type=float, default=1.5e-5, help="kinematic viscosity m^2/s"
    )
    parser.add_argument(
        "--reference-length", type=float, default=1.2, help="reference length m"
    )
    parser.add_argument(
        "--u-lattice", type=float, default=0.06, help="lattice inlet speed"
    )
    add_orientation_args(parser)
    parser.add_argument(
        "--snapshots",
        metavar="DIR",
        help="write velocity snapshots (float16) into DIR during the solve",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=0,
        metavar="N",
        help="number of snapshots, spaced over the last 2 flow-throughs",
    )
    parser.add_argument(
        "--snapshot-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="dtype for velocity snapshots (default float16; use float32 for higher precision)",
    )
    parser.add_argument(
        "--quality",
        choices=("low", "medium", "high", "ultra"),
        help="target cell count: low ~1M, medium ~3M, high ~9M, ultra ~17M "
        "(default: the --preset grid)",
    )
    parser.add_argument(
        "--cells",
        type=int,
        help="exact target cell count; overrides --quality",
    )
    parser.add_argument(
        "--smagorinsky", type=float, default=0.12, help="LES coefficient"
    )
    parser.add_argument(
        "--out", default="results", help="output directory (default: ./results)"
    )
    parser.add_argument(
        "--save-fields",
        action="store_true",
        help="also write velocity/density/force fields as .npz",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if bool(args.snapshots) != bool(args.snapshot_count):
        print("sstream: --snapshots DIR and --snapshot-count N must be used together",
              file=sys.stderr)
        return 1
    preset = next(p for p in PRESETS if p.name.lower() == args.preset)
    device = pick_device()
    u_lattice = args.u_lattice
    smagorinsky = args.smagorinsky
    target = resolve_target_cells(args.quality, args.cells)
    
    # Load surface first to get model length for tunnel sizing
    surface = load_surface(args.mesh)
    build_temp = build_model(surface, mesh_spec_from_args(args), TunnelSpec())
    model_bounds = np.asarray(build_temp.mesh.bounds, dtype=float).reshape(3, 2)
    model_length = float(model_bounds[0, 1] - model_bounds[0, 0])
    
    if target is not None:
        base = TunnelSpec.from_model_length(model_length)
        tunnel = base.with_resolution(cells_streamwise_for_target(base, target))
    else:
        tunnel = TunnelSpec.from_model_length(model_length).with_resolution(preset.cells_streamwise)
    cells = grid_cells(tunnel)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        build = build_model(surface, mesh_spec_from_args(args), tunnel)
        for warning in build.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        placed = np.asarray(build.mesh.bounds, dtype=float).reshape(3, 2)
        placed_extent = placed[:, 1] - placed[:, 0]
        bbox_blockage = (
            placed_extent[1] * placed_extent[2] / (tunnel.width * tunnel.height)
        )
        print(
            f"Placed        {placed_extent[0]:.2f} x {placed_extent[1]:.2f} x "
            f"{placed_extent[2]:.2f} m, bbox blockage {bbox_blockage:.1%}"
        )
        vram = check_vram(cells, device)
        free_text = (
            f" of {format_bytes(vram[1])} free on {device}"
            if vram is not None
            else " (free device memory unavailable)"
        )
        print(
            f"Grid          {tunnel.grid_shape[0]} x {tunnel.grid_shape[1]} x "
            f"{tunnel.grid_shape[2]}  ({cells:,} cells)"
        )
        print(f"VRAM est      {format_bytes(vram[0])}{free_text}" if vram else "VRAM est      unavailable")
        mask = voxelize(build.mesh, tunnel)
        if mask.empty:
            raise ValueError("voxelization produced no solid cells")

        fluid = FluidSpec(
            speed=args.speed,
            kinematic_viscosity=args.viscosity,
            reference_length=args.reference_length,
        )
        placement = (
            placement_dict(args.mesh, mesh_spec_from_args(args), build)
            if args.snapshots
            else None
        )
        plan = SolvePlan(
            mask=mask,
            tunnel=tunnel,
            fluid=fluid,
            preset=preset,
            u_lattice=u_lattice,
            smagorinsky=smagorinsky,
            snapshots_dir=args.snapshots,
            snapshot_count=args.snapshot_count,
            snapshot_dtype=args.snapshot_dtype,
            placement=placement,
        )

        start = time.perf_counter()
        result = run_case(
            plan,
            progress=lambda pct, msg: print(
                f"\r{solve_label(pct, msg)}", end="", flush=True
            ),
        )
        print()

        text = summary(result, Path(args.mesh).name)
        (out_dir / "report.txt").write_text(text, encoding="utf-8")

        payload = {
            "mesh": str(args.mesh),
            "preset": preset.name,
            "cells": result.cells,
            "solid_cells": result.mask.solid_count,
            "cd": float(result.cd),
            "cl": float(result.cl),
            "cl_amplitude": float(result.cl_amplitude),
            "reynolds": float(result.reynolds),
            "steps": result.steps,
            "elapsed_s": round(result.elapsed, 3),
            "device": result.device,
            "warnings": result.warnings,
        }
        if result.snapshot_dir:
            payload["snapshots"] = {
                "dir": result.snapshot_dir,
                "steps": result.snapshot_steps,
                "bytes": result.snapshot_bytes,
            }
        (out_dir / "results.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if args.save_fields:
            np.savez_compressed(
                out_dir / "fields.npz",
                velocity=result.velocity,
                density=result.density,
                vorticity=result.vorticity,
                force=result.force,
            )
        print(text)
        print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'report.txt'}")
        return 0
    except SolveCancelled:
        print("solve cancelled", file=sys.stderr)
        return 2
    except Exception as exc:  # CLI boundary: one clean message, exit code 1
        print(f"sstream: {exc}", file=sys.stderr)
        return 1


def solve_label(percent: int, message: str) -> str:
    return f"[{percent:3d}%] {message}"


if __name__ == "__main__":
    raise SystemExit(main())
