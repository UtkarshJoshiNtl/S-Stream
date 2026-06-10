"""Headless preset thumbnail generator.
Run from project root: python scripts/render_presets.py
Saves a PNG per preset in presets/scenes/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.lbm2d import LBM2D
from presets.loader import list_presets, load_preset
from scene.scene import apply_to_sim

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def render_thumbnail(scene, steps: int = 3000) -> np.ndarray:
    sim = LBM2D(width=scene.width, height=scene.height, viscosity=scene.viscosity)
    apply_to_sim(scene, sim)
    for _ in range(steps):
        sim.step()
    return sim.get_smoke()


def save_thumbnail(smoke: np.ndarray, path: Path) -> None:
    if not HAS_PIL:
        print("PIL not available, skipping thumbnail save")
        return
    data = (np.clip(smoke, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(data, mode="L")
    img = img.resize((200, int(200 * smoke.shape[0] / smoke.shape[1])), Image.LANCZOS)
    img.save(path)


def main() -> None:
    presets_dir = Path(__file__).parent.parent / "presets" / "scenes"
    for preset in list_presets():
        path = Path(preset["file"])
        print(f"Rendering {path.stem}...")
        scene = load_preset(preset["file"])
        smoke = render_thumbnail(scene)
        thumb_path = presets_dir / f"{path.stem}.png"
        save_thumbnail(smoke, thumb_path)
        print(f"  Saved {thumb_path}")


if __name__ == "__main__":
    main()
