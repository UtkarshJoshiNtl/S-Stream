from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from scene.scene import (
    AirfoilObstacle,
    ChannelObstacle,
    CircleObstacle,
    EllipseObstacle,
    ImageObstacle,
    LatticeObstacle,
    RectObstacle,
    STLObstacle,
    apply_to_sim,
)
from scene.serializer import dict_to_scene, scene_to_dict


@pytest.fixture
def sim() -> LBM2D:
    s = LBM2D(width=64, height=64)
    s.initialize()
    return s


class TestEllipseObstacle:
    def test_ellipse_creates_obstacle(self, sim: LBM2D) -> None:
        obs = EllipseObstacle(name="E", x=32, y=32, rx=10, ry=5)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()
        assert mask[32, 32]
        assert not mask[0, 0]

    def test_ellipse_is_elliptical(self, sim: LBM2D) -> None:
        obs = EllipseObstacle(name="E", x=32, y=32, rx=10, ry=5)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask[32, 32 + 9]
        assert not mask[32, 32 + 11]

    def test_ellipse_rotation(self, sim: LBM2D) -> None:
        obs = EllipseObstacle(name="E", x=32, y=32, rx=10, ry=5, rotation=90.0)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask[32, 32]
        assert mask[32 + 4, 32]

    def test_ellipse_small_radius(self, sim: LBM2D) -> None:
        obs = EllipseObstacle(name="E", x=32, y=32, rx=1, ry=1)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask[32, 32]
        count = mask.sum()
        assert count >= 1

    def test_ellipse_serialization(self) -> None:
        obs = EllipseObstacle(name="E", x=32, y=32, rx=10, ry=5, rotation=45.0)
        d = {
            "type": "ellipse",
            "name": "E",
            "x": 32,
            "y": 32,
            "rx": 10,
            "ry": 5,
            "rotation": 45.0,
        }
        restored = dict_to_scene(
            {
                "width": 64,
                "height": 64,
                "obstacles": [d],
            }
        )
        assert len(restored.obstacles) == 1
        assert isinstance(restored.obstacles[0], EllipseObstacle)
        assert restored.obstacles[0].rx == 10


class TestSTLObstacle:
    def _make_sphere_stl(self, path: str, radius: float = 5.0) -> None:
        import trimesh

        mesh = trimesh.creation.uv_sphere(radius=radius, count=[16, 16])
        mesh.export(path)

    def test_stl_2d_creates_obstacle(self, sim: LBM2D) -> None:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            self._make_sphere_stl(f.name, radius=5.0)
            path = f.name
        obs = STLObstacle(name="Sphere", path=path, scale=1.0, offset_x=32, offset_y=32)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()
        Path(path).unlink()

    def test_stl_empty_path(self, sim: LBM2D) -> None:
        obs = STLObstacle(name="Empty", path="")
        obs.apply(sim)
        assert not sim.get_obstacles().any()

    def test_stl_serialization(self) -> None:
        d = {
            "type": "stl",
            "name": "Test",
            "path": "/tmp/test.stl",
            "scale": 2.0,
            "offset_x": 10,
            "offset_y": 20,
            "filled": True,
        }
        restored = dict_to_scene({"width": 64, "height": 64, "obstacles": [d]})
        assert isinstance(restored.obstacles[0], STLObstacle)
        assert restored.obstacles[0].scale == 2.0
        assert restored.obstacles[0].path == "/tmp/test.stl"


class TestImageObstacle:
    def _make_test_image(self, path: str, w: int = 32, h: int = 32) -> None:
        from PIL import Image

        arr = np.zeros((h, w), dtype=np.uint8)
        arr[10:20, 10:20] = 255
        Image.fromarray(arr).save(path)

    def test_image_creates_obstacle(self, sim: LBM2D) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            self._make_test_image(f.name)
            path = f.name
        obs = ImageObstacle(name="TestImg", path=path, threshold=128)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()
        Path(path).unlink()

    def test_image_threshold(self, sim: LBM2D) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            self._make_test_image(f.name)
            path = f.name
        obs_high = ImageObstacle(name="High", path=path, threshold=200)
        obs_high.apply(sim)
        count_high = sim.get_obstacles().sum()
        sim.clear_obstacles()
        obs_low = ImageObstacle(name="Low", path=path, threshold=50)
        obs_low.apply(sim)
        count_low = sim.get_obstacles().sum()
        assert count_low >= count_high
        Path(path).unlink()

    def test_image_invert(self, sim: LBM2D) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            self._make_test_image(f.name)
            path = f.name
        obs = ImageObstacle(name="Inv", path=path, threshold=128, invert=True)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()
        Path(path).unlink()

    def test_image_empty_path(self, sim: LBM2D) -> None:
        obs = ImageObstacle(name="Empty", path="")
        obs.apply(sim)
        assert not sim.get_obstacles().any()

    def test_image_serialization(self) -> None:
        d = {
            "type": "image",
            "name": "Test",
            "path": "/tmp/test.png",
            "threshold": 200,
            "invert": True,
            "scale_x": 2.0,
            "scale_y": 2.0,
        }
        restored = dict_to_scene({"width": 64, "height": 64, "obstacles": [d]})
        assert isinstance(restored.obstacles[0], ImageObstacle)
        assert restored.obstacles[0].threshold == 200
        assert restored.obstacles[0].invert is True


class TestAirfoilObstacle:
    def test_airfoil_creates_obstacle(self, sim: LBM2D) -> None:
        obs = AirfoilObstacle(name="NACA", x=32, y=32, chord=20, naca_code="0012")
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()
        assert mask.sum() > 10

    def test_airfoil_symmetric(self, sim: LBM2D) -> None:
        obs = AirfoilObstacle(name="Sym", x=32, y=32, chord=20, naca_code="0012")
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask[32, 32]

    def test_airfoil_cambered(self, sim: LBM2D) -> None:
        obs = AirfoilObstacle(name="Cam", x=32, y=32, chord=20, naca_code="2412")
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()

    def test_airfoil_angle_of_attack(self, sim: LBM2D) -> None:
        obs_zero = AirfoilObstacle(
            name="A0", x=32, y=32, chord=20, angle_of_attack=0.0, naca_code="0012"
        )
        obs_zero.apply(sim)
        assert sim.get_obstacles().any()
        sim.clear_obstacles()
        obs_aoa = AirfoilObstacle(
            name="A10", x=32, y=32, chord=20, angle_of_attack=10.0, naca_code="0012"
        )
        obs_aoa.apply(sim)
        assert sim.get_obstacles().any()

    def test_airfoil_serialization(self) -> None:
        d = {
            "type": "airfoil",
            "name": "AF",
            "x": 32,
            "y": 32,
            "chord": 20,
            "angle_of_attack": 5.0,
            "naca_code": "2412",
        }
        restored = dict_to_scene({"width": 64, "height": 64, "obstacles": [d]})
        assert isinstance(restored.obstacles[0], AirfoilObstacle)
        assert restored.obstacles[0].naca_code == "2412"


class TestChannelObstacle:
    def test_channel_creates_obstacle(self, sim: LBM2D) -> None:
        obs = ChannelObstacle(
            name="Ch", x=10, y=10, w=40, h=30, inlet_ratio=0.5, outlet_ratio=0.5
        )
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()

    def test_channel_converging(self, sim: LBM2D) -> None:
        obs = ChannelObstacle(
            name="Conv", x=10, y=10, w=40, h=30, inlet_ratio=1.0, outlet_ratio=0.5
        )
        obs.apply(sim)
        mask = sim.get_obstacles()
        wall_cells_left = mask[10:40, 10].sum()
        wall_cells_right = mask[10:40, 49].sum()
        assert wall_cells_right > wall_cells_left

    def test_channel_diverging(self, sim: LBM2D) -> None:
        obs = ChannelObstacle(
            name="Div", x=10, y=10, w=40, h=30, inlet_ratio=0.5, outlet_ratio=1.0
        )
        obs.apply(sim)
        mask = sim.get_obstacles()
        wall_cells_left = mask[10:40, 10].sum()
        wall_cells_right = mask[10:40, 49].sum()
        assert wall_cells_left > wall_cells_right

    def test_channel_serialization(self) -> None:
        d = {
            "type": "channel",
            "name": "Ch",
            "x": 10,
            "y": 10,
            "w": 40,
            "h": 30,
            "inlet_ratio": 0.6,
            "outlet_ratio": 1.0,
        }
        restored = dict_to_scene({"width": 64, "height": 64, "obstacles": [d]})
        assert isinstance(restored.obstacles[0], ChannelObstacle)
        assert restored.obstacles[0].inlet_ratio == 0.6


class TestLatticeObstacle:
    def test_lattice_creates_obstacle(self, sim: LBM2D) -> None:
        obs = LatticeObstacle(name="Lat", x=10, y=10, w=30, h=30, cell_size=8)
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert mask.any()
        total_cells = 30 * 30
        assert mask.sum() < total_cells

    def test_lattice_has_holes(self, sim: LBM2D) -> None:
        obs = LatticeObstacle(
            name="Lat", x=10, y=10, w=32, h=32, cell_size=8, wall_thickness=1
        )
        obs.apply(sim)
        mask = sim.get_obstacles()
        assert not mask[14, 14]

    def test_lattice_serialization(self) -> None:
        d = {
            "type": "lattice",
            "name": "Lat",
            "x": 10,
            "y": 10,
            "w": 30,
            "h": 30,
            "cell_size": 8,
            "wall_thickness": 2,
        }
        restored = dict_to_scene({"width": 64, "height": 64, "obstacles": [d]})
        assert isinstance(restored.obstacles[0], LatticeObstacle)
        assert restored.obstacles[0].wall_thickness == 2


class TestPhase4Serialization:
    def test_roundtrip_all_types(self) -> None:
        scene_dict = {
            "width": 128,
            "height": 128,
            "obstacles": [
                {"type": "circle", "name": "C", "x": 32, "y": 32, "radius": 5},
                {"type": "rect", "name": "R", "x": 10, "y": 10, "w": 20, "h": 15},
                {
                    "type": "ellipse",
                    "name": "E",
                    "x": 64,
                    "y": 64,
                    "rx": 10,
                    "ry": 5,
                    "rotation": 0.0,
                },
                {
                    "type": "airfoil",
                    "name": "AF",
                    "x": 50,
                    "y": 64,
                    "chord": 20,
                    "angle_of_attack": 5.0,
                    "naca_code": "0012",
                },
                {
                    "type": "channel",
                    "name": "CH",
                    "x": 10,
                    "y": 30,
                    "w": 80,
                    "h": 60,
                    "inlet_ratio": 0.5,
                    "outlet_ratio": 1.0,
                },
                {
                    "type": "lattice",
                    "name": "LA",
                    "x": 10,
                    "y": 10,
                    "w": 40,
                    "h": 40,
                    "cell_size": 8,
                    "wall_thickness": 1,
                },
            ],
            "emitters": [{"name": "E1", "x": 2, "y": 64, "strength": 0.05}],
        }
        scene = dict_to_scene(scene_dict)
        assert len(scene.obstacles) == 6
        assert isinstance(scene.obstacles[0], CircleObstacle)
        assert isinstance(scene.obstacles[1], RectObstacle)
        assert isinstance(scene.obstacles[2], EllipseObstacle)
        assert isinstance(scene.obstacles[3], AirfoilObstacle)
        assert isinstance(scene.obstacles[4], ChannelObstacle)
        assert isinstance(scene.obstacles[5], LatticeObstacle)
        d = scene_to_dict(scene)
        assert len(d["obstacles"]) == 6
        assert d["obstacles"][0]["type"] == "circle"
        assert d["obstacles"][2]["type"] == "ellipse"
        assert d["obstacles"][3]["type"] == "airfoil"
        assert d["obstacles"][4]["type"] == "channel"
        assert d["obstacles"][5]["type"] == "lattice"

    def test_stl_image_roundtrip(self) -> None:
        scene_dict = {
            "width": 64,
            "height": 64,
            "obstacles": [
                {
                    "type": "stl",
                    "name": "M",
                    "path": "/tmp/test.stl",
                    "scale": 1.5,
                    "offset_x": 10,
                    "offset_y": 20,
                    "filled": True,
                },
                {
                    "type": "image",
                    "name": "I",
                    "path": "/tmp/test.png",
                    "threshold": 200,
                    "invert": False,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                },
            ],
        }
        scene = dict_to_scene(scene_dict)
        assert isinstance(scene.obstacles[0], STLObstacle)
        assert isinstance(scene.obstacles[1], ImageObstacle)
        d = scene_to_dict(scene)
        assert d["obstacles"][0]["type"] == "stl"
        assert d["obstacles"][1]["type"] == "image"
