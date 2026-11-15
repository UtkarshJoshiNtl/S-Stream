from pathlib import Path

import numpy as np
import pyvista as pv

from sstream.mesh import build_model, load_surface
from sstream.schema import MeshSpec, MeshTransform, TunnelSpec
from sstream.voxelize import voxelize

ASSET = Path(__file__).resolve().parents[1] / "assets" / "3d-files" / "car-model.obj"
NO_WALL = TunnelSpec(length=2.0, width=2.0, height=2.0, cells_streamwise=40)


def _cube(side: float = 0.5) -> pv.PolyData:
    return pv.Cube(
        center=(1.0, 0.0, 1.0), x_length=side, y_length=side, z_length=side
    ).triangulate()


def test_analytic_cube_volume_and_orientation():
    result = voxelize(_cube(0.5), NO_WALL)
    assert result.dimensions == (40, 40, 40)
    assert result.watertight
    assert not result.warnings
    # 0.5 m / 0.05 m = 10 cells along each axis, so 1000 interior points.
    assert result.solid_count == 1000
    assert result.boundary.sum() > 0
    assert result.boundary.sum() < result.solid_count

    indices = np.argwhere(result.solid)
    centres = np.asarray(result.origin) + indices * np.asarray(result.spacing)
    np.testing.assert_allclose(centres.mean(axis=0), (1.0, 0.0, 1.0), atol=1e-6)


def test_mask_shape_matches_the_tunnel_grid():
    result = voxelize(_cube(0.5), NO_WALL)
    assert result.solid.shape == NO_WALL.grid_shape
    assert result.boundary.shape == result.solid.shape
    assert result.origin == NO_WALL.grid_origin
    assert result.spacing == NO_WALL.grid_spacing

    image = result.to_image()
    assert image.dimensions == result.dimensions
    assert image.n_points == int(np.prod(result.dimensions))
    assert int(np.asarray(image.point_data["solid"]).sum()) == result.solid_count


def test_open_mesh_warns_without_crashing():
    triangle = pv.PolyData(
        np.array([[0.5, 0.0, 0.5], [1.5, 0.0, 0.5], [1.0, 1.0, 0.5]]),
        np.array([3, 0, 1, 2]),
    )
    result = voxelize(triangle, NO_WALL)
    assert not result.watertight
    assert result.warnings
    assert any("not watertight" in warning for warning in result.warnings)
    assert result.dimensions == NO_WALL.grid_shape


def test_car_pipeline_produces_a_non_empty_mask():
    if not ASSET.exists():
        return
    surface = load_surface(ASSET)
    build = build_model(
        surface,
        MeshSpec(fraction=0.2, transform=MeshTransform(roll=-90, yaw=-90)),
        TunnelSpec(),
    )
    tunnel = TunnelSpec().with_resolution(64)
    result = voxelize(build.mesh, tunnel)
    assert result.solid_count > 0
    assert result.dimensions == tunnel.grid_shape
    assert result.boundary.sum() > 0
    image = result.to_image()
    assert image.dimensions == result.dimensions


def test_flood_outside_marks_a_closed_shell_interior_unreachable():
    from sstream.voxelize import _flood_outside

    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[2:7, 2, 2:7] = True
    mask[2:7, 6, 2:7] = True
    mask[2, 2:7, 2:7] = True
    mask[6, 2:7, 2:7] = True
    mask[2:7, 2:7, 2] = True
    mask[2:7, 2:7, 6] = True
    outside = _flood_outside(mask)
    assert not outside[4, 4, 4]        # sealed interior stays unreached
    assert outside[0, 4, 4]            # domain boundary is outside

    mask_open = mask.copy()
    mask_open[4, 6, 4] = False         # one-cell hole in the top face
    outside = _flood_outside(mask_open)
    assert outside[4, 4, 4]            # leaks in through the hole


def test_open_box_becomes_a_solid_body():
    import pyvista as pv

    cube = pv.Cube(bounds=(1.0, 1.5, -0.25, 0.25, 0.75, 1.25)).triangulate()
    faces = np.asarray(cube.faces).reshape(-1, 4)
    points = np.asarray(cube.points)
    keep = [f for f in faces if not np.all(points[f[1:], 0] > 1.49)]
    assert keep                        # some cells removed, shell remains
    open_box = pv.PolyData(points, np.concatenate(keep))

    result = voxelize(open_box, NO_WALL)
    assert not result.watertight
    assert any("not watertight" in warning for warning in result.warnings)
    assert result.solid_count > 0
    # A non-watertight shell must behave as a solid body: the missing +x
    # face does not leave an empty pocket inside the grid.  (Whether the
    # interior is solidified by the raster stencil or by the unreachable
    # fill is an implementation detail; the flood-fill leak semantics are
    # covered by test_flood_outside_marks_a_closed_shell_interior_unreachable.)
    origin = np.asarray(NO_WALL.grid_origin)
    spacing = np.asarray(NO_WALL.grid_spacing)
    centre = np.round((np.array([1.25, 0.0, 1.0]) - origin) / spacing).astype(int)
    assert result.solid[tuple(centre)]
