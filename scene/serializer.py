from __future__ import annotations

import json
from pathlib import Path

from scene.scene import (
    CircleObstacle,
    EmitterSpec,
    ObstacleSpec,
    PolygonObstacle,
    ProbeSpec,
    RectObstacle,
    Scene,
)

SCHEMA_VERSION = 1

_OBSTACLE_DECODERS: dict[str, type[ObstacleSpec]] = {
    "circle": CircleObstacle,
    "rect": RectObstacle,
    "polygon": PolygonObstacle,
}


def _obs_to_dict(obs: ObstacleSpec) -> dict:
    if isinstance(obs, CircleObstacle):
        return {
            "type": "circle",
            "name": obs.name,
            "x": obs.x,
            "y": obs.y,
            "radius": obs.radius,
        }
    if isinstance(obs, RectObstacle):
        return {
            "type": "rect",
            "name": obs.name,
            "x": obs.x,
            "y": obs.y,
            "w": obs.w,
            "h": obs.h,
        }
    if isinstance(obs, PolygonObstacle):
        return {"type": "polygon", "name": obs.name, "points": obs.points}
    raise TypeError(f"Unknown obstacle type: {type(obs)}")


def _obs_from_dict(d: dict) -> ObstacleSpec:
    cls = _OBSTACLE_DECODERS.get(d["type"])
    if cls is None:
        raise ValueError(f"Unknown obstacle type: {d['type']}")
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(**kwargs)


def scene_to_dict(scene: Scene) -> dict:
    d = {
        "schema_version": SCHEMA_VERSION,
        "name": scene.name,
        "width": scene.width,
        "height": scene.height,
        "viscosity": scene.viscosity,
        "u_inflow": scene.u_inflow,
        "smoke_diffusion": scene.smoke_diffusion,
        "smoke_decay": scene.smoke_decay,
        "obstacles": [_obs_to_dict(o) for o in scene.obstacles],
        "emitters": [
            {"name": e.name, "x": e.x, "y": e.y, "strength": e.strength}
            for e in scene.emitters
        ],
        "probes": [
            {"name": p.name, "x": p.x, "y": p.y, "fields": list(p.fields)}
            for p in scene.probes
        ],
    }
    if scene.description:
        d["description"] = scene.description
    if scene.sweeps:
        d["sweeps"] = scene.sweeps
    return d


def dict_to_scene(d: dict) -> Scene:
    return Scene(
        name=d.get("name", "Untitled"),
        width=d.get("width", 128),
        height=d.get("height", 128),
        viscosity=d.get("viscosity", 0.02),
        u_inflow=d.get("u_inflow", 0.15),
        smoke_diffusion=d.get("smoke_diffusion", 0.05),
        smoke_decay=d.get("smoke_decay", 0.999),
        description=d.get("description", ""),
        obstacles=[_obs_from_dict(o) for o in d.get("obstacles", [])],
        emitters=[EmitterSpec(**e) for e in d.get("emitters", [])],
        probes=[ProbeSpec(**p) for p in d.get("probes", [])],
        sweeps=d.get("sweeps", []),
    )


def save(scene: Scene, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(scene_to_dict(scene), f, indent=2)


def load(path: str | Path) -> Scene:
    with open(path) as f:
        d = json.load(f)
    return dict_to_scene(d)
