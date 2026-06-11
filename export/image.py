from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

from engines.base import SimEngine
from scene.scene import Scene


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
    field = _compute_field(sim, colormap)
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


def _compute_field(sim: SimEngine, cmap: str) -> np.ndarray:
    if cmap == "smoke":
        return sim.get_smoke()
    vel = sim.get_velocity()
    if cmap == "speed":
        speed = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        mx = max(sim.u_inflow * 1.5, float(speed.max()), 0.001)
        return np.clip(speed / mx, 0, 1)
    if cmap == "vorticity":
        u = vel[:, :, 0]
        v = vel[:, :, 1]
        dvdx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
        vort = dvdx - dudy
        mx = max(float(abs(vort).max()), 0.001)
        return np.clip(vort / mx * 0.5 + 0.5, 0, 1)
    if cmap == "pressure":
        rho = sim.get_density()
        p = rho - 1.0
        mx = max(float(abs(p).max()), 0.001)
        return np.clip(p / mx * 0.5 + 0.5, 0, 1)
    return sim.get_smoke()


def _render_field(img: QImage, field: np.ndarray, cmap: str) -> None:
    h, w = field.shape
    pixels = bytearray()
    for val in field.ravel():
        if cmap == "smoke":
            pixels.extend(_smoke_color(val))
        elif cmap == "vorticity":
            pixels.extend(_vorticity_color(val))
        elif cmap == "pressure":
            pixels.extend(_pressure_color(val))
        else:
            v = int(max(0.0, min(1.0, float(val))) * 255)
            pixels.extend([v, v, v])
    qimg = QImage(bytes(pixels), w, h, w * 3, QImage.Format.Format_RGB888)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.drawImage(img.rect(), qimg)
    painter.end()


def _smoke_color(val: float) -> bytes:
    v = max(0.0, min(1.0, float(val)))
    bg = (5, 5, 20)
    smoke = (255, 191, 102)
    return bytes(
        int(bg[i] + (smoke[i] - bg[i]) * (v**0.5))
        for i in range(3)
    )


def _vorticity_color(val: float) -> bytes:
    v = max(0.0, min(1.0, float(val)))
    if v < 0.5:
        t = v / 0.5
        return bytes([int(20 + 40 * t), int(80 + 120 * t), int(190 + 45 * t)])
    t = (v - 0.5) / 0.5
    return bytes([int(240 + 15 * t), int(220 - 120 * t), int(120 - 80 * t)])


def _pressure_color(val: float) -> bytes:
    v = max(0.0, min(1.0, float(val)))
    return bytes([int(40 + 180 * v), int(60 + 80 * (1 - abs(v - 0.5))), int(220 - 150 * v)])


def _draw_colorbar(
    painter: QPainter,
    x: int,
    y: int,
    w: int,
    h: int,
    cmap: str,
) -> None:
    bar_img = QImage(w, h, QImage.Format.Format_RGB32)
    bar_painter = QPainter(bar_img)
    for i in range(h):
        t = 1.0 - i / (h - 1) if h > 1 else 0.0
        if cmap == "smoke":
            color = QColor(*_smoke_color(t))
        elif cmap == "vorticity":
            color = QColor(*_vorticity_color(t))
        elif cmap == "pressure":
            color = QColor(*_pressure_color(t))
        else:
            v = int(t * 255)
            color = QColor(v, v, v)
        bar_painter.setPen(color)
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
    from analysis.physics import characteristic_length, drag_coefficient, reynolds_number

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
        f"View = {colormap}",
        f"Step = {step}",
        "SStream educational CFD",
    ]
    y = img_h - 14 * len(lines) - 4
    for line in lines:
        painter.drawText(8, y, line)
        y += 14
