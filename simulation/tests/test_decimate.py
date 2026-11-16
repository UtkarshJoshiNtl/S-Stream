from pathlib import Path

import numpy as np

from sstream.mesh import build_model, decimate_surface, load_surface
from sstream.options import mesh_spec_from_args
from sstream.quality import cells_streamwise_for_target
from sstream.schema import MeshSpec, TunnelSpec
from sstream.voxelize import voxelize

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "3d-files"


class _Args:
    orient = "auto-vehicle"
    up = "+Y"
    flip = False
    yaw = 0.0
    pitch = 0.0
    fraction = 0.2


def _low_quality_tunnel(surface, orient_args=_Args()):
    build_temp = build_model(surface, mesh_spec_from_args(orient_args), TunnelSpec())
    model_bounds = np.asarray(build_temp.mesh.bounds, dtype=float).reshape(3, 2)
    model_length = float(model_bounds[0, 1] - model_bounds[0, 0])
    base = TunnelSpec.from_model_length(model_length)
    return base.with_resolution(cells_streamwise_for_target(base, 1_000_000))


def test_decimate_keeps_bbox_and_solid_count_at_low_quality():
    path = ASSETS / "ahmed_1.stl"
    original = load_surface(path)
    target_faces = 20000
    decimated = decimate_surface(original, target_faces)

    assert decimated.n_cells <= target_faces
    np.testing.assert_allclose(decimated.bounds, original.bounds, atol=1e-6)

    spec = MeshSpec(orient="auto-vehicle", up="+Y", flip=False, pitch=0.0, yaw=0.0, fraction=0.2)
    tunnel = _low_quality_tunnel(original)

    orig_build = build_model(original, spec, tunnel)
    dec_build = build_model(decimated, spec, tunnel)
    orig_mask = voxelize(orig_build.mesh, tunnel)
    dec_mask = voxelize(dec_build.mesh, tunnel)

    assert orig_mask.solid_count > 0
    delta = abs(dec_mask.solid_count - orig_mask.solid_count) / orig_mask.solid_count
    assert delta <= 0.05, (
        f"solid-cell count changed by {delta:.2%}: "
        f"original={orig_mask.solid_count} decimated={dec_mask.solid_count}"
    )
