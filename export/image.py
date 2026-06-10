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
    """Render the fluid field to a high-resolution PNG.

    Parameters
    ----------
    sim: Engine with fluid data (get_smoke/get_velocity/get_density).
    scene: Scene metadata (grid dims, name).
    path: Output file path.
    scale: Upscale factor (2 = 2x the simulation grid resolution).
    include_colorbar: Draw a vertical colorbar on the right.
    include_annotations: Draw Re, viscosity, step, timestamp text.
    step_count: Current simulation step (for annotation).
    colormap: Field to render ("smoke", "speed", "vorticity", "pressure").
    """
    field = _compute_field(sim, colormap)
    h, w = field.shape
    out_w = w * scale
    out_h = h * scale
    pad = 80 if include_colorbar else 20
    total_w = out_w + pad

    field_img = QImage(out_w, out_h, QImage.Format.Format_RGB32)
    _render_field(field_img, field, colormap)

    result = QImage(total_w, out_h, QImage.Format.Format_RGB32)
    result.fill(QColor(18, 18, 30))
    painter = QPainter(result)
    painter.drawImage(0, 0, field_img)

    if include_colorbar:
        _draw_colorbar(painter, out_w + 10, 20, 20, out_h - 80, colormap)

    if include_annotations:
        _draw_annotations(painter, scene, sim, step_count, total_w, out_h)

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
    """Paint the 2D field onto *img* using the appropriate colormap."""
    h, w = field.shape
    flat = field.ravel()
    if cmap == "smoke":
        pixels = bytearray()
        for val in flat:
            c = _smoke_color(val)
            pixels.extend(c)
    else:
        pixels = bytearray()
        for val in flat:
            v = int(val * 255)
            pixels.extend([v, v, v])
    qimg = QImage(bytes(pixels), w, h, w * 3, QImage.Format.Format_RGB888)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.drawImage(img.rect(), qimg)
    painter.end()


def _smoke_color(val: float) -> bytes:
    v = max(0.0, min(1.0, val))
    bg = (5, 5, 20)
    smoke = (255, 191, 102)
    r = int(bg[0] + (smoke[0] - bg[0]) * (v**0.5))
    g = int(bg[1] + (smoke[1] - bg[1]) * (v**0.5))
    b = int(bg[2] + (smoke[2] - bg[2]) * (v**0.5))
    return bytes([r, g, b])


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
        else:
            v = int(t * 255)
            color = QColor(v, v, v)
        bar_painter.setPen(color)
        bar_painter.drawLine(0, i, w - 1, i)
    bar_painter.end()
    painter.drawImage(x, y, bar_img)

    pen = QPen(QColor(200, 200, 200))
    painter.setPen(pen)
    painter.drawRect(x, y, w, h)
    font = QFont("monospace", 8)
    painter.setFont(font)
    painter.drawText(x + w + 4, y + 8, "1.0")
    painter.drawText(x + w + 4, y + h // 2 + 3, "0.5")
    painter.drawText(x + w + 4, y + h + 4, "0.0")


def _draw_annotations(
    painter: QPainter,
    scene: Scene,
    sim: SimEngine,
    step: int,
    img_w: int,
    img_h: int,
) -> None:
    from analysis.physics import reynolds_number

    L = max(
        (obs.radius * 2 for obs in scene.obstacles if hasattr(obs, "radius")),
        default=float(scene.width),
    )
    re = reynolds_number(sim, obstacle_diameter=L)

    font = QFont("monospace", 9)
    painter.setFont(font)
    painter.setPen(QPen(QColor(200, 200, 200)))
    lines = [
        f"Re = {re:.1f}",
        f"ν = {sim.viscosity}",
        f"U_in = {sim.u_inflow}",
        f"Step = {step}",
    ]
    y = img_h - 14 * len(lines) - 4
    for line in lines:
        painter.drawText(8, y, line)
        y += 14
