from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from sstream.mesh import (
    build_model,
    load_surface,
    normalized,
    orient_to_wind,
    setup_rotated,
    yawed_about_bottom_center,
)
from sstream.schema import MeshSpec, MeshTransform, TunnelSpec

ASSET = Path(__file__).resolve().parents[1] / "assets" / "3d-files" / "car-model.obj"


def _save_cube(tmp_path: Path, suffix: str) -> Path:
    path = tmp_path / f"cube.{suffix}"
    pv.Cube(center=(1, 2, 3), x_length=4, y_length=2, z_length=6).save(path)
    return path


@pytest.mark.parametrize("suffix", ["obj", "stl"])
def test_loads_obj_and_stl_as_triangles(tmp_path, suffix):
    mesh = load_surface(_save_cube(tmp_path, suffix))
    assert mesh.n_points >= 8
    assert mesh.n_cells >= 12
    assert np.isfinite(np.asarray(mesh.points)).all()
    extent = np.asarray(mesh.bounds).reshape(3, 2)
    np.testing.assert_allclose(extent[:, 1] - extent[:, 0], (4, 2, 6), atol=1e-5)


def test_rejects_unsupported_formats(tmp_path):
    path = tmp_path / "cube.ply"
    pv.Cube().save(path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_surface(path)


def test_normalize_centres_and_scales_to_target():
    mesh = pv.Cube(center=(8, -3, 2), x_length=4, y_length=2, z_length=6).triangulate()
    out, scale = normalized(mesh, 2.0)
    bounds = np.asarray(out.bounds).reshape(3, 2)
    np.testing.assert_allclose(out.center, (0, 0, 0), atol=1e-6)
    np.testing.assert_allclose(
        bounds[:, 1] - bounds[:, 0], np.array((4, 2, 6)) * scale
    )
    assert pytest.approx(scale) == 2.0 / 6.0


def test_setup_rotation_round_trip():
    mesh = pv.Cube(x_length=4, y_length=2, z_length=1).triangulate()
    forward = mesh
    for step in (
        MeshTransform(roll=-35),
        MeshTransform(pitch=20),
        MeshTransform(yaw=70),
    ):
        forward = setup_rotated(forward, step)
    restored = forward
    for step in (
        MeshTransform(yaw=-70),
        MeshTransform(pitch=-20),
        MeshTransform(roll=35),
    ):
        restored = setup_rotated(restored, step)
    np.testing.assert_allclose(
        np.asarray(restored.points), np.asarray(mesh.points), atol=1e-6
    )


def test_yaw_keeps_the_model_on_the_floor():
    mesh = pv.Cube(center=(10, 0, 0.5), x_length=2, y_length=1, z_length=1).triangulate()
    rotated = yawed_about_bottom_center(mesh, 90)
    bounds = np.asarray(rotated.bounds).reshape(3, 2)
    assert bounds[2, 0] == pytest.approx(0.0)
    assert bounds[2, 1] == pytest.approx(1.0)
    assert rotated.center[0] == pytest.approx(10.0)
    assert rotated.center[1] == pytest.approx(0.0)


def test_build_clamps_inside_the_tunnel_and_touches_the_floor():
    mesh = pv.Cube(x_length=4, y_length=2, z_length=1).triangulate()
    spec = MeshSpec(
        target_size=1.0,
        transform=MeshTransform(position_x=100.0, position_y=-100.0),
    )
    build = build_model(mesh, spec, TunnelSpec())
    bounds = build.bounds
    tunnel = TunnelSpec()
    assert bounds[2, 0] == pytest.approx(tunnel.z_min)
    assert bounds[0, 0] >= tunnel.x_min
    assert bounds[0, 1] <= tunnel.x_max
    assert bounds[1, 0] >= tunnel.y_min
    assert bounds[1, 1] <= tunnel.y_max
    assert not build.warnings


def test_bundled_car_loads_and_keeps_a_sane_bbox():
    assert ASSET.exists(), f"missing bundled asset: {ASSET}"
    mesh = load_surface(ASSET)
    assert mesh.n_points > 1000
    assert mesh.n_cells > 1000
    assert np.isfinite(np.asarray(mesh.points)).all()
    extent = np.asarray(mesh.bounds).reshape(3, 2)
    longest = float((extent[:, 1] - extent[:, 0]).max())
    assert 20.0 < longest < 60.0


def test_auto_orient_maps_longest_axis_to_x_and_shortest_to_z():
    mesh = pv.Box(bounds=(0, 1.0, 0, 4.0, 0, 2.0)).triangulate()
    out = orient_to_wind(mesh)
    extent = np.asarray(out.bounds).reshape(3, 2)
    size = extent[:, 1] - extent[:, 0]
    assert size[0] == pytest.approx(4.0)
    assert size[2] == pytest.approx(1.0)
    assert size[1] == pytest.approx(2.0)


def test_auto_orient_is_a_proper_rotation():
    mesh = pv.PolyData(
        np.array([[1.0, 0.0, 0.0], [0.0, 4.0, 0.5], [0.0, 0.0, 2.0], [2.0, 3.0, 1.0]]),
        np.array([3, 0, 1, 2, 3, 0, 2, 3]),
    )

    def signed_volume(points):
        v0, v1, v2, v3 = points[:4]
        return np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)

    out = orient_to_wind(mesh)
    assert signed_volume(np.asarray(out.points)) == pytest.approx(
        signed_volume(np.asarray(mesh.points)), rel=1e-6
    )


def test_build_model_can_opt_out_of_auto_orient():
    mesh = pv.Box(bounds=(0, 1.0, 0, 4.0, 0, 2.0)).triangulate()
    oriented = build_model(mesh, MeshSpec(target_size=1.2), TunnelSpec())
    untouched = build_model(mesh, MeshSpec(target_size=1.2, auto_orient=False), TunnelSpec())
    o_size = (oriented.bounds[:, 1] - oriented.bounds[:, 0])
    u_size = (untouched.bounds[:, 1] - untouched.bounds[:, 0])
    assert o_size[0] == pytest.approx(1.2)
    # longest axis stays y without auto_orient
    assert u_size[1] == pytest.approx(1.2)
