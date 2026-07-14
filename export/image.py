from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

from engines.base import SimEngine
from scene.scene import Scene


from resources.colormaps import CMAP_LUTS, MODE_TO_CMAP


def _get_lut(mode: str) -> np.ndarray:
    return CMAP_LUTS.get(MODE_TO_CMAP.get(mode, "viridis"), CMAP_LUTS["viridis"])


def _field_label(name: str) -> str:
    from resources.colormaps import FIELD_REGISTRY

    info = FIELD_REGISTRY.get(name)
    return info.label if info else name.capitalize()


def export_image(
    sim: SimEngine,
    scene: Scene,
    path: str | Path,
    scale: int = 2,
    include_colorbar: bool = True,
    include_annotations: bool = True,
    step_count: int = 0,
    colormap: str = "smoke",
) -> None:
    try:
        field = sim.get_field(colormap)
    except ValueError:
        field = sim.get_field("smoke")
    h, w = field.shape
    out_w = w * scale
    out_h = h * scale
    pad = 96 if include_colorbar else 24
    total_w = out_w + pad

    field_img = QImage(out_w, out_h, QImage.Format.Format_RGB32)
    _render_field(field_img, field, colormap)

    result = QImage(total_w, out_h, QImage.Format.Format_RGB32)
    result.fill(QColor(18, 18, 30))
    painter = QPainter(result)
    painter.drawImage(0, 0, field_img)

    if include_colorbar:
        _draw_colorbar(painter, out_w + 12, 28, 20, max(20, out_h - 96), colormap)

    if include_annotations:
        _draw_annotations(painter, scene, sim, step_count, total_w, out_h, colormap)

    painter.end()
    if not result.save(str(path)):
        raise RuntimeError(f"Failed to save image to {path}")


def _render_field(img: QImage, field: np.ndarray, cmap: str) -> None:
    h, w = field.shape
    lut = _get_lut(cmap)
    idx = np.clip((field * 255).astype(np.int32), 0, 255)
    rgb = (lut[idx] * 255).astype(np.uint8).reshape(h, w, 3)
    pixels = rgb.tobytes()
    qimg = QImage(pixels, w, h, w * 3, QImage.Format.Format_RGB888)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.drawImage(img.rect(), qimg)
    painter.end()


def _draw_colorbar(
    painter: QPainter,
    x: int,
    y: int,
    w: int,
    h: int,
    cmap: str,
) -> None:
    lut = _get_lut(cmap)
    bar_img = QImage(w, h, QImage.Format.Format_RGB32)
    bar_painter = QPainter(bar_img)
    for i in range(h):
        t = 1.0 - i / (h - 1) if h > 1 else 0.0
        idx = int(t * 255)
        r, g, b = lut[idx]
        r8, g8, b8 = int(r * 255), int(g * 255), int(b * 255)
        bar_painter.setPen(QColor(r8, g8, b8))
        bar_painter.drawLine(0, i, w - 1, i)
    bar_painter.end()
    painter.drawImage(x, y, bar_img)

    painter.setPen(QPen(QColor(210, 210, 220)))
    painter.drawRect(x, y, w, h)
    painter.setFont(QFont("monospace", 8))
    painter.drawText(x + w + 5, y + 8, "1.0")
    painter.drawText(x + w + 5, y + h // 2 + 3, "0.5")
    painter.drawText(x + w + 5, y + h + 4, "0.0")


def _draw_annotations(
    painter: QPainter,
    scene: Scene,
    sim: SimEngine,
    step: int,
    img_w: int,
    img_h: int,
    colormap: str,
) -> None:
    from analysis.physics import (
        characteristic_length,
        drag_coefficient,
        reynolds_number,
    )

    length = characteristic_length(scene)
    re = reynolds_number(sim, obstacle_diameter=length)
    cd = drag_coefficient(sim)

    title_font = QFont("monospace", 11)
    title_font.setBold(True)
    body_font = QFont("monospace", 9)

    painter.setPen(QPen(QColor(235, 235, 245)))
    painter.setFont(title_font)
    painter.drawText(8, 18, scene.name)

    caption = scene.product.export_caption or scene.product.lesson_headline
    if caption:
        painter.setFont(body_font)
        painter.drawText(8, 34, caption[:90])

    painter.setFont(body_font)
    painter.setPen(QPen(QColor(205, 205, 215)))
    lines = [
        f"Re = {re:.1f}",
        f"Cd = {cd:.3f}",
        f"nu = {sim.viscosity}",
        f"U_in = {sim.u_inflow}",
        f"View = {_field_label(colormap)}",
        f"Step = {step}",
        "SStream educational CFD",
    ]
    y = img_h - 14 * len(lines) - 4
    for line in lines:
        painter.drawText(8, y, line)
        y += 14
