"""Low-Re sphere validation at Re ~ 100 (slow, GPU).

Reference: Johnson, T. A., & Patel, V. C. (1999). Flow past a sphere up to a
Reynolds number of 300. Journal of Fluid Mechanics, 378, 19–70.
https://doi.org/10.1017/S0022112098003206
Reported total drag coefficient C_D = 1.087 at Re = 100 (steady, axisymmetric),
as recorded in docs/REFERENCES.md (value cited from commonly reported result;
see references for verification status).

No blockage correction is applied to the measured C_D in this test; blockage
ratio is recorded for context.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sstream.fields import frontal_area
from sstream.lbm import VALIDATION_WALL_MODES
from sstream.mesh import load_surface, build_model
from sstream.pipeline import SolvePlan, run_case
from sstream.preview import RunPreset
from sstream.schema import FluidSpec, MeshSpec, TunnelSpec
from sstream.voxelize import voxelize

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "3d-files"
REFERENCE_CD = 1.087  # Johnson & Patel (1999), Re=100, docs/REFERENCES.md


@pytest.mark.slow
def test_sphere_re100_drag_and_force_window_convergence():
    surface = load_surface(ASSETS / "sphere.stl")
    # Wide tunnel to keep blockage low; enough resolution/steps for a stable Re~100 estimate.
    tunnel = TunnelSpec(length=4.0, width=2.4, height=2.4, floor_z=0.0, cells_streamwise=96)
    build = build_model(surface, MeshSpec(orient="none", fraction=0.1), tunnel)
    mask = voxelize(build.mesh, tunnel)
    assert not mask.empty

    bounds = np.asarray(build.mesh.bounds, dtype=float).reshape(3, 2)
    extent = bounds[:, 1] - bounds[:, 0]
    diameter = float(extent.mean())  # sphere: extents ~ equal
    speed = 0.05
    reynolds = 100.0
    viscosity = speed * diameter / reynolds
    fluid = FluidSpec(speed=speed, kinematic_viscosity=viscosity, reference_length=diameter)

    # Enough steps for startup transient to pass and for a stable force window.
    preset = RunPreset("SphereRe100", cells_streamwise=tunnel.cells_streamwise, steps=12000, description="slow Re=100 sphere")
    plan = SolvePlan(
        mask=mask,
        tunnel=tunnel,
        fluid=fluid,
        preset=preset,
        u_lattice=0.05,
        smagorinsky=0.12,
        wall_modes=VALIDATION_WALL_MODES,
        moving_wall_speed=0.0,
    )
    result = run_case(plan)

    assert np.isfinite(result.cd)
    assert np.isfinite(result.velocity).all()
    assert np.isfinite(result.density).all()
    assert result.units.tau >= 0.5
    mach = result.units.u_lattice / np.sqrt(1.0 / 3.0)
    assert mach < 0.3

    area = frontal_area(result.mask.solid, tunnel.grid_spacing)
    silhouette_blockage = area / (tunnel.width * tunnel.height)
    bbox_blockage = (extent[1] * extent[2]) / (tunnel.width * tunnel.height)

    cd_trace = np.asarray(result.cd_trace, dtype=float)
    assert cd_trace.size >= 20
    # Force-window convergence check on late samples (steady wake expected).
    third = max(5, cd_trace.size // 3)
    early_late = cd_trace[cd_trace.size // 2 - third // 2 : cd_trace.size // 2 + third // 2]
    last = cd_trace[-third:]
    window_change = abs(float(last.mean()) - float(early_late.mean()))
    # Allow modest residual drift/noise at this coarse resolution.
    window_tol = 0.08 * max(abs(float(last.mean())), 1e-6)

    print(
        f"sphere Re~100: Cd={result.cd:.4f} (ref {REFERENCE_CD}, no blockage correction), "
        f"Re={result.reynolds:.2f}, bbox_blockage={bbox_blockage:.4%}, "
        f"silhouette_blockage={silhouette_blockage:.4%}, tau={result.units.tau:.6f}, "
        f"mach={mach:.4f}, cd_trace_n={cd_trace.size}, "
        f"window_change={window_change:.5f}, window_tol={window_tol:.5f}"
    )

    assert window_change < window_tol, (
        f"force window not converged: change {window_change:.5f} >= tol {window_tol:.5f} "
        f"(early-late mean {early_late.mean():.4f}, last mean {last.mean():.4f})"
    )
    assert abs(result.cd - REFERENCE_CD) / REFERENCE_CD <= 0.20, (
        f"Cd {result.cd:.4f} differs from reference {REFERENCE_CD} by more than 20% "
        f"(no blockage correction applied; bbox_blockage={bbox_blockage:.4%}, "
        f"silhouette_blockage={silhouette_blockage:.4%})"
    )
