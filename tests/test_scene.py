from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scene.scene import (
    CircleObstacle,
    EmitterSpec,
    PolygonObstacle,
    ProbeSpec,
    RectObstacle,
    Scene,
    default_scene,
)
from scene.serializer import load, save, scene_to_dict


def _make_scene() -> Scene:
    return Scene(
        name="TestScene",
        width=64,
        height=32,
        viscosity=0.01,
        u_inflow=0.2,
        smoke_diffusion=0.03,
        smoke_decay=0.99,
        obstacles=[
            CircleObstacle(name="cyl", x=20, y=16, radius=5),
            RectObstacle(name="wall", x=0, y=0, w=3, h=32),
            PolygonObstacle(name="tri", points=[(10, 10), (20, 10), (15, 20)]),
        ],
        emitters=[EmitterSpec(name="inlet", x=2, y=16, strength=0.1)],
        probes=[ProbeSpec(name="wake", x=40, y=16, fields=["u", "v"])],
    )


def test_default_scene() -> None:
    s = default_scene()
    assert s.name == "Untitled"
    assert s.width == 128
    assert s.height == 128
    assert len(s.emitters) == 1
    assert s.emitters[0].name == "Inlet"


def test_serialize_round_trip() -> None:
    scene = _make_scene()
    d = scene_to_dict(scene)
    assert d["schema_version"] == 1
    assert d["name"] == "TestScene"
    assert len(d["obstacles"]) == 3
    assert len(d["emitters"]) == 1
    assert len(d["probes"]) == 1

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(d, f)
        tmppath = f.name

    try:
        loaded = load(tmppath)
        assert loaded.name == "TestScene"
        assert loaded.width == 64
        assert loaded.height == 32
        assert loaded.viscosity == 0.01
        assert loaded.u_inflow == 0.2
        assert loaded.smoke_diffusion == 0.03
        assert loaded.smoke_decay == 0.99

        assert len(loaded.obstacles) == 3
        assert isinstance(loaded.obstacles[0], CircleObstacle)
        assert loaded.obstacles[0].x == 20
        assert loaded.obstacles[0].radius == 5
        assert isinstance(loaded.obstacles[1], RectObstacle)
        assert loaded.obstacles[1].w == 3
        assert isinstance(loaded.obstacles[2], PolygonObstacle)
        assert len(loaded.obstacles[2].points) == 3

        assert len(loaded.emitters) == 1
        assert loaded.emitters[0].x == 2
        assert loaded.emitters[0].strength == 0.1

        assert len(loaded.probes) == 1
        assert loaded.probes[0].fields == ["u", "v"]
    finally:
        Path(tmppath).unlink()


def test_save_load_file() -> None:
    scene = _make_scene()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.json"
        save(scene, path)
        assert path.exists()
        loaded = load(path)
        assert loaded.name == "TestScene"
        assert len(loaded.obstacles) == 3


def test_obstacle_apply_circle() -> None:
    from engines.lbm2d import LBM2D

    sim = LBM2D(32, 32)
    obs = CircleObstacle(name="c", x=16, y=16, radius=4)
    obs.apply(sim)
    mask = sim.get_obstacles()
    assert mask[16, 16]
    assert mask[16, 12]
    assert not mask[16, 8]


def test_obstacle_apply_rect() -> None:
    from engines.lbm2d import LBM2D

    sim = LBM2D(32, 32)
    obs = RectObstacle(name="r", x=8, y=8, w=5, h=10)
    obs.apply(sim)
    mask = sim.get_obstacles()
    assert mask[8, 8]
    assert mask[17, 12]
    assert not mask[7, 8]


def test_obstacle_apply_polygon() -> None:
    from engines.lbm2d import LBM2D

    sim = LBM2D(32, 32)
    obs = PolygonObstacle(name="t", points=[(10, 10), (20, 10), (15, 20)])
    obs.apply(sim)
    mask = sim.get_obstacles()
    assert mask[15, 15]
    assert not mask[0, 0]
