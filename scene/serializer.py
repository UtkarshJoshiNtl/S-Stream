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
    SceneProductMeta,
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
    product = scene.product
    product_dict = {
        "recommended_colormap": product.recommended_colormap,
        "autorun_steps": product.autorun_steps,
        "lesson_headline": product.lesson_headline,
        "expected_ranges": product.expected_ranges,
        "flow_regime_labels": product.flow_regime_labels,
        "export_caption": product.export_caption,
        "classroom_prompts": product.classroom_prompts,
        "recommended_sweep": product.recommended_sweep,
        "recipe": product.recipe,
    }
    if any(product_dict.values()):
        d["product"] = product_dict
    return d


def _product_from_dict(d: dict) -> SceneProductMeta:
    product = d.get("product", {})
    return SceneProductMeta(
        recommended_colormap=product.get("recommended_colormap", "smoke"),
        autorun_steps=int(product.get("autorun_steps", 0)),
        lesson_headline=product.get("lesson_headline", ""),
        expected_ranges=product.get("expected_ranges", {}),
        flow_regime_labels=product.get("flow_regime_labels", []),
        export_caption=product.get("export_caption", ""),
        classroom_prompts=product.get("classroom_prompts", []),
        recommended_sweep=product.get("recommended_sweep", {}),
        recipe=product.get("recipe", ""),
    )


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
        product=_product_from_dict(d),
    )


def save(scene: Scene, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(scene_to_dict(scene), f, indent=2)


def load(path: str | Path) -> Scene:
    with open(path) as f:
        d = json.load(f)
    return dict_to_scene(d)
