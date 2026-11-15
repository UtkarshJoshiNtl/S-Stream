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
        MeshSpec(target_size=1.2, transform=MeshTransform(roll=-90, yaw=-90)),
        TunnelSpec(),
    )
    tunnel = TunnelSpec().with_resolution(64)
    result = voxelize(build.mesh, tunnel)
    assert result.solid_count > 0
    assert result.dimensions == tunnel.grid_shape
    assert result.boundary.sum() > 0
    image = result.to_image()
    assert image.dimensions == result.dimensions
