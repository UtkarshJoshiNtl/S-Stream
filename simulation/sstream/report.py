"""A plain-text result sheet.

The report is intentionally quiet about certainty.  It states the grid, the
model, the numbers, and what those numbers are not.  A reader should be able to
tell from this text alone that the result is a voxelized large-eddy simulation
rather than a certified drag measurement.
"""

from __future__ import annotations

import numpy as np

from . import fields
from .pipeline import SolveResult


def summary(result: SolveResult, model_name: str) -> str:
    tunnel = result.tunnel
    fluid = result.fluid
    nx, ny, nz = tunnel.grid_shape
    area = fields.frontal_area(result.mask.solid, tunnel.grid_spacing)
    peak = float(np.sqrt(np.sum(result.velocity**2, axis=-1)).max())
    rows = [
        "S-Stream — wind tunnel report",
        "",
        f"Model          {model_name}",
        f"Preset         {result.preset.name} ({result.preset.cells_streamwise} cells streamwise)",
        f"Grid           {nx} x {ny} x {nz}  ({result.cells:,} cells)",
        f"Solid cells    {result.mask.solid_count:,}",
        f"Frontal area   {area:.4f} m²",
        "",
        f"Air speed      {fluid.speed:.1f} m/s",
        f"Reynolds       {result.reynolds:,.0f}  (reference length {fluid.reference_length:.2f} m)",
        f"Cd             {result.cd:.3f}",
        f"Cl (up)        {result.cl:+.3f}",
        f"Cl unsteady    ±{result.cl_amplitude:.3f}  (half peak-to-peak)",
        "",
        f"Lattice        u = {result.units.u_lattice:.3f}, tau = {result.units.tau:.6f}",
        f"Eddy viscosity Cs = {result.units.smagorinsky:.2f}",
        f"Peak |u|/U∞    {peak / result.units.u_lattice:.2f}",
        f"Steps          {result.steps:,} in {result.elapsed:.1f} s on {result.device}",
        "",
        "Method",
        "  D3Q19 lattice Boltzmann, half-way bounce-back walls, Guo forcing,",
        "  Smagorinsky eddy viscosity.  Pressure is the LBM equation of state",
        "  p = cs² rho.  Cd uses the voxel silhouette as the reference area.",
        "",
        "Limits",
        "  The geometry is voxelized, so small features and thin panels are lost.",
        "  At this Reynolds number the boundary layer is far below the cell size;",
        "  the eddy-viscosity floor is doing real work.  Treat Cd/Cl as relative",
        "  values for comparing shapes, not as certified drag figures.",
        "  Cd is indicative, not calibrated: it is valid for comparing shapes and",
        "  orientations of the same model, not as a drag figure to quote against",
        "  a physical measurement.",
        "  The model sits in a closed tunnel with a no-slip floor and walls; both",
        "  confinement and floor interference are part of these numbers.",
    ]
    if result.warnings:
        rows.extend(["", "Warnings"] + [f"  - {line}" for line in result.warnings])
    return "\n".join(rows)
