from __future__ import annotations

import logging
from pathlib import Path

from scene.scene import Scene
from scene.serializer import load as scene_load

logger = logging.getLogger(__name__)

_PRESETS_DIR = Path(__file__).parent / "scenes"


def list_presets() -> list[dict]:
    """Every loadable preset, keyed by its stable id.

    The id is read from the scene's ``product.preset_id`` rather than the filename, so
    a file can be renamed or moved without breaking anything that refers to it.
    """
    presets = []
    for f in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            scene = scene_load(f)
            thumb = f.with_suffix(".png")
            presets.append(
                {
                    "id": scene.product.preset_id or f.stem,
                    "name": scene.name,
                    "file": str(f),
                    "description": scene.description,
                    "thumbnail": str(thumb) if thumb.exists() else "",
                    "headline": scene.product.lesson_headline,
                    "recipe": scene.product.recipe,
                }
            )
        except Exception as e:
            logger.warning("Failed to load preset %s: %s", f, e)
    return presets


def load_preset(path: str) -> Scene:
    return scene_load(path)


def find_preset(preset_id: str) -> dict | None:
    """Exact id lookup. Returns None when the id is unknown."""
    for preset in list_presets():
        if preset["id"] == preset_id:
            return preset
    return None


def validate_presets() -> list[str]:
    """Check the library for problems that would break id-based lookup.

    Returns a list of human-readable problems, empty when the library is sound.
    Callers should surface these at startup rather than letting a recipe fail
    silently the first time somebody clicks it.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    for f in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            scene = scene_load(f)
        except Exception as e:
            problems.append(f"{f.name}: unreadable ({e})")
            continue
        preset_id = scene.product.preset_id
        if not preset_id:
            problems.append(f"{f.name}: missing product.preset_id")
            continue
        if preset_id in seen:
            problems.append(
                f"{f.name}: id '{preset_id}' already used by {seen[preset_id]}"
            )
            continue
        seen[preset_id] = f.name
    return problems
