from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

from engines.base import SimEngine
from scene.scene import Scene


# --- Shared colormap data ---

def _interp_cmap(stops, n=256):
    pos = np.array([s[0] for s in stops], dtype=np.float64)
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([
        np.interp(x, pos, cols[:, i]) for i in range(3)
    ]).astype(np.float32)

_VIRIDIS_STOPS = [
    (0.0, (0.267, 0.004, 0.329)),
    (0.1, (0.282, 0.098, 0.460)),
    (0.2, (0.254, 0.185, 0.551)),
    (0.3, (0.207, 0.270, 0.602)),
    (0.4, (0.163, 0.354, 0.619)),
    (0.5, (0.128, 0.437, 0.609)),
    (0.6, (0.135, 0.520, 0.572)),
    (0.7, (0.206, 0.602, 0.508)),
    (0.8, (0.368, 0.680, 0.401)),
    (0.9, (0.603, 0.736, 0.242)),
    (1.0, (0.993, 0.906, 0.144)),
]

_PLASMA_STOPS = [
    (0.0, (0.050, 0.030, 0.528)),
    (0.1, (0.215, 0.022, 0.593)),
    (0.2, (0.385, 0.002, 0.601)),
    (0.3, (0.539, 0.031, 0.554)),
    (0.4, (0.670, 0.105, 0.470)),
    (0.5, (0.780, 0.194, 0.367)),
    (0.6, (0.872, 0.298, 0.252)),
    (0.7, (0.950, 0.414, 0.132)),
    (0.8, (0.990, 0.546, 0.051)),
    (0.9, (0.966, 0.695, 0.119)),
    (1.0, (0.940, 0.851, 0.212)),
]

_COOLWARM_STOPS = [
    (0.0, (0.231, 0.299, 0.754)),
    (0.25, (0.490, 0.620, 0.890)),
    (0.5, (0.865, 0.865, 0.865)),
    (0.75, (0.890, 0.560, 0.440)),
    (1.0, (0.706, 0.016, 0.150)),
]

_BLUES_STOPS = [
    (0.0, (0.02, 0.02, 0.08)),
    (0.3, (0.03, 0.06, 0.20)),
    (0.5, (0.05, 0.20, 0.50)),
    (0.7, (0.10, 0.50, 0.80)),
    (0.85, (0.30, 0.75, 0.95)),
    (1.0, (0.80, 0.95, 1.0)),
]

_INFERNO_STOPS = [
    (0.0, (0.001, 0.000, 0.014)),
    (0.1, (0.088, 0.025, 0.174)),
    (0.2, (0.210, 0.036, 0.388)),
    (0.3, (0.356, 0.043, 0.569)),
    (0.4, (0.512, 0.065, 0.679)),
    (0.5, (0.661, 0.126, 0.714)),
    (0.6, (0.794, 0.209, 0.668)),
    (0.7, (0.910, 0.315, 0.549)),
    (0.8, (0.980, 0.444, 0.384)),
    (0.9, (0.987, 0.590, 0.203)),
    (1.0, (0.940, 0.782, 0.057)),
]

_CMAP_LUTS: dict[str, np.ndarray] = {
    "viridis": _interp_cmap(_VIRIDIS_STOPS),
    "plasma": _interp_cmap(_PLASMA_STOPS),
    "coolwarm": _interp_cmap(_COOLWARM_STOPS),
    "blues": _interp_cmap(_BLUES_STOPS),
    "inferno": _interp_cmap(_INFERNO_STOPS),
}

_MODE_TO_CMAP: dict[str, str] = {
    "smoke": "viridis",
    "speed": "plasma",
    "vorticity": "coolwarm",
    "pressure": "coolwarm",
    "density": "inferno",
    "phase": "blues",
}


def _get_lut(mode: str) -> np.ndarray:
    return _CMAP_LUTS.get(_MODE_TO_CMAP.get(mode, "viridis"), _CMAP_LUTS["viridis"])


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
        field = sim.get_smoke()
        mx = max(float(np.percentile(field, 98)), 0.001)
        return np.clip(field / mx, 0, 1)
    vel = sim.get_velocity()
    if cmap == "speed":
        speed = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        mx = max(
            sim.u_inflow * 1.5 if hasattr(sim, "u_inflow") else 0.0,
            float(np.percentile(speed, 98)),
            0.001,
        )
        return np.clip(speed / mx, 0, 1)
    if cmap == "vorticity":
        u = vel[:, :, 0]
        v = vel[:, :, 1]
        dvdx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
        vort = dvdx - dudy
        mx = max(float(np.percentile(abs(vort), 98)), 0.001)
        return np.clip(vort / mx * 0.5 + 0.5, 0, 1)
    if cmap == "pressure":
        rho = sim.get_density()
        p = rho - 1.0
        mx = max(float(np.percentile(abs(p), 98)), 0.001)
        return np.clip(p / mx * 0.5 + 0.5, 0, 1)
    if cmap == "density":
        rho = sim.get_density()
        lo, hi = float(np.min(rho)), float(np.max(rho))
        if hi - lo < 0.001:
            return np.full_like(rho, 0.5, dtype=np.float32)
        return np.clip((rho - lo) / (hi - lo), 0, 1).astype(np.float32)
    if cmap == "phase":
        rho = sim.get_density()
        field = 1.0 / (1.0 + np.exp(-15 * (rho - 0.5)))
        return np.clip(field, 0, 1).astype(np.float32)
    return sim.get_smoke()


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
