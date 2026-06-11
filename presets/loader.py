from __future__ import annotations

from pathlib import Path

from scene.scene import Scene
from scene.serializer import load as scene_load

_PRESETS_DIR = Path(__file__).parent / "scenes"


def list_presets() -> list[dict]:
    presets = []
    for f in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            scene = scene_load(f)
            thumb = f.with_suffix(".png")
            presets.append({
                "name": scene.name,
                "file": str(f),
                "description": scene.description,
                "thumbnail": str(thumb) if thumb.exists() else "",
                "headline": scene.product.lesson_headline,
                "recipe": scene.product.recipe,
            })
        except Exception:
            continue
    return presets


def load_preset(path: str) -> Scene:
    return scene_load(path)
