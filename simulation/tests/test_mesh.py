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
    upright_rotation,
    yawed_about_bottom_center,
)
from sstream.schema import MeshSpec, MeshTransform, TunnelSpec

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "3d-files"
AHMED = ASSETS / "ahmed_1.stl"


def _save_cube(tmp_path: Path, suffix: str) -> Path:
    path = tmp_path / f"cube.{suffix}"
    pv.Cube(center=(1, 2, 3), x_length=4, y_length=2, z_length=6).save(path)
    return path


@pytest.mark.parametrize("suffix", ["obj", "stl", "ply"])
def test_loads_obj_stl_and_ply_as_triangles(tmp_path, suffix):
    mesh = load_surface(_save_cube(tmp_path, suffix))
    assert mesh.n_points >= 8
    assert mesh.n_cells >= 12
    assert np.isfinite(np.asarray(mesh.points)).all()
    extent = np.asarray(mesh.bounds).reshape(3, 2)
    np.testing.assert_allclose(extent[:, 1] - extent[:, 0], (4, 2, 6), atol=1e-5)


def test_rejects_blender_and_fbx_with_an_export_message(tmp_path):
    for suffix in ("blend", "fbx"):
        path = tmp_path / f"cube.{suffix}"
        path.write_bytes(b"not a real model")
        with pytest.raises(ValueError, match="export the model from Blender"):
            load_surface(path)


def test_rejects_unknown_formats(tmp_path):
    path = tmp_path / "cube.glb"
    path.write_bytes(b"not a real model")
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
        fraction=1.0 / 6.0,  # default tunnel length is 6 m
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


def test_bundled_ahmed_loads_and_keeps_a_sane_bbox():
    assert AHMED.exists(), f"missing bundled asset: {AHMED}"
    mesh = load_surface(AHMED)
    assert mesh.n_points > 1000
    assert mesh.n_cells > 1000
    assert np.isfinite(np.asarray(mesh.points)).all()
    extent = np.asarray(mesh.bounds).reshape(3, 2)
    longest = float((extent[:, 1] - extent[:, 0]).max())
    assert 1.0 < longest < 1.3

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


def test_build_model_orient_none_keeps_the_file_as_is():
    mesh = pv.Box(bounds=(0, 1.0, 0, 4.0, 0, 2.0)).triangulate()
    placed = build_model(mesh, MeshSpec(orient="none"), TunnelSpec())
    size = placed.bounds[:, 1] - placed.bounds[:, 0]
    # longest axis stays y when no orientation is applied
    assert size[1] == pytest.approx(1.2)
    assert size[0] == pytest.approx(1.2 / 4.0)


def test_upright_keeps_a_tall_thin_mesh_upright():
    # Blender-style Y-up export: the tall axis is Y in file coordinates.
    mesh = pv.Box(bounds=(0, 1.0, 0, 4.0, 0, 1.0)).triangulate()
    placed = build_model(mesh, MeshSpec(orient="upright", up="+Y", fraction=0.2), TunnelSpec())
    size = placed.bounds[:, 1] - placed.bounds[:, 0]
    assert size[2] == pytest.approx(0.2 * 6.0)   # still tallest, and vertical
    assert size[0] == pytest.approx(0.2 * 6.0 / 4.0)
    assert size[1] == pytest.approx(0.2 * 6.0 / 4.0)


def test_upright_picks_the_longer_horizontal_extent_as_wind():
    # Z-up model, wider than deep: wind must follow the wide axis.
    wide = pv.Box(bounds=(0, 3.0, 0, 1.0, 0, 4.0)).triangulate()
    placed = build_model(wide, MeshSpec(orient="upright", up="+Z", fraction=0.2), TunnelSpec())
    size = placed.bounds[:, 1] - placed.bounds[:, 0]
    scale = 0.2 * 6.0 / 4.0   # longest extent (Z, 4 m) fills the target
    assert size[0] == pytest.approx(3.0 * scale)
    assert size[1] == pytest.approx(1.0 * scale)
    assert size[2] == pytest.approx(4.0 * scale)

    narrow = pv.Box(bounds=(0, 1.0, 0, 3.0, 0, 4.0)).triangulate()
    placed = build_model(narrow, MeshSpec(orient="upright", up="+Z", fraction=0.2), TunnelSpec())
    size = placed.bounds[:, 1] - placed.bounds[:, 0]
    assert size[0] == pytest.approx(3.0 * scale)   # wind now follows old Y
    assert size[1] == pytest.approx(1.0 * scale)
    assert size[2] == pytest.approx(4.0 * scale)


@pytest.mark.parametrize("up", ["+Y", "+Z"])
def test_upright_rotation_is_proper(up):
    extents = np.array([1.0, 2.0, 4.0])
    rotation = upright_rotation(extents, up)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # the up axis maps to world Z, no matter what
    up_axis = 1 if up == "+Y" else 2
    assert rotation[2, up_axis] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "spec",
    [
        MeshSpec(orient="auto-vehicle"),
        MeshSpec(orient="upright", up="+Y"),
        MeshSpec(orient="upright", up="+Z"),
        MeshSpec(orient="none"),
        MeshSpec(orient="auto-vehicle", flip=True),
        MeshSpec(orient="auto-vehicle", yaw=30.0),
        MeshSpec(orient="auto-vehicle", pitch=-8.0),
        MeshSpec(orient="upright", up="+Y", flip=True, pitch=5.0, yaw=-12.0),
    ],
)
def test_every_placement_transform_is_proper(spec):
    mesh = pv.Box(bounds=(0, 1.0, 0, 4.0, 0, 2.0)).triangulate()

    def signed_volume(points):
        points = np.asarray(points, dtype=np.float64)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    for l in range(k + 1, len(points)):
                        v = np.dot(
                            np.cross(points[j] - points[i], points[k] - points[i]),
                            points[l] - points[i],
                        )
                        if abs(v) > 1e-6:
                            return v
        raise AssertionError("no non-coplanar vertex quadruple found")

    placed = build_model(mesh, spec, TunnelSpec())
    # scale multiplies volume; only the sign of the determinant can flip it
    assert np.sign(signed_volume(np.asarray(placed.mesh.points))) == np.sign(
        signed_volume(np.asarray(mesh.points))
    )
