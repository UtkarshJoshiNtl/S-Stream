"""Outlet reflection sensitivity check (slow, GPU).

Same sphere case run with outlet 1x and 2x farther downstream (same cell size,
model size/position relative to inlet, fluid, and numerics). Drag (Cd) must
differ by under 5% to indicate negligible outlet reflection/boundary influence
for this setup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sstream.mesh import build_model, load_surface
from sstream.pipeline import SolvePlan, run_case
from sstream.preview import RunPreset
from sstream.schema import FluidSpec, MeshSpec, MeshTransform, TunnelSpec
from sstream.voxelize import voxelize

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "3d-files"


def _run_sphere_outlet_case(length: float, cells_streamwise: int, mesh, diameter: float) -> float:
    tunnel = TunnelSpec(length=length, width=2.4, height=2.4, floor_z=0.0, cells_streamwise=cells_streamwise)
    mask = voxelize(mesh, tunnel)
    assert not mask.empty
    fluid = FluidSpec(speed=0.05, kinematic_viscosity=0.05 * diameter / 100.0, reference_length=diameter)
    plan = SolvePlan(
        mask=mask,
        tunnel=tunnel,
        fluid=fluid,
        preset=RunPreset("OutletCheck", cells_streamwise=cells_streamwise, steps=8000, description="outlet reflection"),
        u_lattice=0.05,
        smagorinsky=0.12,
    )
    result = run_case(plan)
    assert np.isfinite(result.cd)
    return float(result.cd)


@pytest.mark.slow
def test_outlet_reflection_sphere_cd_diff_under_5_percent():
    surface = load_surface(ASSETS / "sphere.stl")
    # Reference placement: fixed physical size and distance from inlet.
    ref = TunnelSpec(length=4.0, width=2.4, height=2.4, floor_z=0.0, cells_streamwise=96)
    build = build_model(
        surface,
        MeshSpec(orient="none", fraction=0.1, transform=MeshTransform(position_x=1.0)),
        ref,
    )
    mesh = build.mesh.copy()
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    diameter = float((bounds[:, 1] - bounds[:, 0]).mean())

    cd_1x = _run_sphere_outlet_case(length=4.0, cells_streamwise=96, mesh=mesh, diameter=diameter)
    # 2x farther outlet: extend downstream only, same dx/model placement.
    cd_2x = _run_sphere_outlet_case(length=6.0, cells_streamwise=144, mesh=mesh, diameter=diameter)

    rel_diff = abs(cd_2x - cd_1x) / max(abs(cd_1x), 1e-12)
    print(
        f"outlet reflection sphere: Cd_1x={cd_1x:.5f}, Cd_2x={cd_2x:.5f}, "
        f"rel_diff={rel_diff:.3%}, diameter={diameter:.4f} m"
    )
    assert rel_diff < 0.05, (
        f"outlet reflection too large: Cd differs by {rel_diff:.3%} "
        f"(1x outlet {cd_1x:.5f}, 2x outlet {cd_2x:.5f})"
    )
