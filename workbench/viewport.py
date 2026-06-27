from __future__ import annotations

import ctypes
import math

import numpy as np
import OpenGL.platform as _ogl_plat

# Workaround for PyOpenGL 3.1.x + PySide6 context detection incompatibility.
# PyOpenGL's GetCurrentContext fails to find the PySide6 GL context, so we
# provide a dummy to allow the rest of the binding machinery to proceed.
_ogl_plat.GetCurrentContext = lambda: object()

from OpenGL.GL import (  # noqa: E402
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_R16F,
    GL_RED,
    GL_RGB,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBufferData,
    glClear,
    glClearColor,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glUniform1f,
    glUniform1i,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader  # noqa: E402
from PySide6.QtCore import Qt, QPointF, Signal  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen  # noqa: E402
from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: E402

from engines.base import SimEngine  # noqa: E402
from scene.probe import Probe  # noqa: E402
from scene.scene import (  # noqa: E402
    CircleObstacle,
    ObstacleSpec,
    PolygonObstacle,
    ProbeSpec,
    RectObstacle,
    Scene,
)
from resources.colormaps import CMAP_LUTS, MODE_TO_CMAP  # noqa: E402

_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 uv;
void main() {
    uv = vec2(aUV.x, 1.0 - aUV.y);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_FRAG_SRC = """
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D smokeTex;
uniform sampler2D cmapTex;
uniform float uBoost;
void main() {
    float s = texture(smokeTex, uv).r;
    s = pow(s, 1.0 / (1.0 + uBoost));
    vec3 c = texture(cmapTex, vec2(s, 0.5)).rgb;
    float glow = exp(-(1.0 - s) * 10.0) * 0.2 * uBoost;
    fragColor = vec4(c + glow, 1.0);
}
"""


class Viewport(QOpenGLWidget):
    obstacle_created = Signal(ObstacleSpec)
    probe_placed = Signal(ProbeSpec)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sim: SimEngine | None = None
        self.scene: Scene | None = None
        self.probes: list[Probe] = []
        self._colormap = "speed"
        self._gamma = 0.7
        self._perf_mode = False
        self._tex_init = False
        self._cmap_tex: int | None = None
        self._cmap_uploaded: str | None = None
        self._show_quiver = False
        self._show_streamlines = False
        self.draw_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_end: tuple[float, float] | None = None
        self._poly_points: list[tuple[float, float]] = []
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_sim(self, sim: SimEngine) -> None:
        self.sim = sim

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def set_probes(self, probes: list[Probe]) -> None:
        self.probes = probes

    def set_colormap(self, name: str) -> None:
        self._colormap = name
        self._upload_colormap()

    def set_perf_mode(self, enabled: bool) -> None:
        if enabled != self._perf_mode:
            self._perf_mode = enabled
            self._tex_init = False

    def set_show_quiver(self, show: bool) -> None:
        self._show_quiver = show
        self.update()

    def set_show_streamlines(self, show: bool) -> None:
        self._show_streamlines = show
        self.update()

    def set_draw_mode(self, mode: str | None) -> None:
        self.draw_mode = mode
        self._drag_start = None
        self._drag_end = None
        self._poly_points = []
        cursor = Qt.CursorShape.CrossCursor if mode else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        self.update()

    # --- coordinate helpers ---

    def _widget_to_grid(self, wx: float, wy: float) -> tuple[float, float]:
        if self.scene is None:
            return (0.0, 0.0)
        w = self.width()
        h = self.height()
        gx = wx / w * self.scene.width
        gy = wy / h * self.scene.height
        return (gx, gy)

    def _grid_to_widget(self, gx: float, gy: float) -> tuple[float, float]:
        if self.scene is None:
            return (0.0, 0.0)
        w = self.width()
        h = self.height()
        wx = gx / self.scene.width * w
        wy = gy / self.scene.height * h
        return (wx, wy)

    # --- OpenGL init / paint ---

    def initializeGL(self) -> None:
        self.shader = compileProgram(
            compileShader(_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        self.texture = glGenTextures(1)
        self._cmap_tex = glGenTextures(1)
        verts = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                -1.0, -1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                -1.0, 1.0, 0.0, 1.0,
            ],
            dtype=np.float32,
        )
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, self._cmap_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self._upload_colormap()

    def paintGL(self) -> None:
        if self.sim is None:
            return
        glClearColor(0.08, 0.08, 0.14, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self._upload_smoke()
        glUseProgram(self.shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.shader, "smokeTex"), 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._cmap_tex)
        glUniform1i(glGetUniformLocation(self.shader, "cmapTex"), 1)
        glUniform1f(glGetUniformLocation(self.shader, "uBoost"), self._gamma)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

        vel = self._get_vel()
        self._draw_overlay(vel)

    def _upload_smoke(self) -> None:
        cmap = self._colormap
        if cmap == "smoke":
            field = self.sim.get_smoke()
            mx = max(float(np.percentile(field, 98)), 0.001)
            field = np.clip(field / mx, 0, 1).astype(np.float32)
        elif cmap in ("speed", "vorticity"):
            u = getattr(self.sim, "u", None)
            v = getattr(self.sim, "v", None)
            if u is None or v is None:
                vel = self.sim.get_velocity()
                u, v = vel[:, :, 0], vel[:, :, 1]
            if cmap == "speed":
                speed = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)
                mx = max(
                    self.sim.u_inflow * 1.5,
                    float(np.percentile(speed, 98)),
                    0.001,
                )
                field = np.clip(speed / mx, 0, 1).astype(np.float32)
            else:
                dvdx = np.zeros_like(u, dtype=np.float32)
                dudy = np.zeros_like(u, dtype=np.float32)
                dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
                dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
                vort = dvdx - dudy
                mx = max(float(np.percentile(abs(vort), 98)), 0.001)
                field = np.clip(vort / mx * 0.5 + 0.5, 0, 1).astype(np.float32)
        elif cmap == "pressure":
            rho = self.sim.get_density()
            p = rho - 1.0
            mx = max(float(np.percentile(abs(p), 98)), 0.001)
            field = np.clip(p / mx * 0.5 + 0.5, 0, 1).astype(np.float32)
        elif cmap == "density":
            rho = self.sim.get_density()
            lo, hi = float(np.min(rho)), float(np.max(rho))
            if hi - lo < 0.001:
                field = np.full_like(rho, 0.5, dtype=np.float32)
            else:
                field = np.clip((rho - lo) / (hi - lo), 0, 1).astype(np.float32)
        elif cmap == "phase":
            rho = self.sim.get_density()
            field = 1.0 / (1.0 + np.exp(-15 * (rho - 0.5)))
            field = np.clip(field, 0, 1).astype(np.float32)
        else:
            field = self.sim.get_smoke()

        field = np.ascontiguousarray(field)
        h, w = field.shape
        if self._perf_mode:
            field_u8 = (field * 255).astype(np.uint8)
            fmt = GL_RGB
            upload = field_u8
        else:
            fmt = GL_RED
            upload = field
        glBindTexture(GL_TEXTURE_2D, self.texture)
        if not self._tex_init:
            internal = GL_RGB if self._perf_mode else GL_R16F
            pix_type = GL_UNSIGNED_BYTE if self._perf_mode else GL_FLOAT
            glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, pix_type, upload)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            self._tex_init = True
        else:
            pix_type = GL_UNSIGNED_BYTE if self._perf_mode else GL_FLOAT
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, pix_type, upload)

    def _upload_colormap(self) -> None:
        cmap_name = MODE_TO_CMAP.get(self._colormap, "viridis")
        lut = CMAP_LUTS.get(cmap_name, CMAP_LUTS["viridis"])
        data = np.clip(lut * 255, 0, 255).astype(np.uint8).reshape(256, 1, 3)
        glBindTexture(GL_TEXTURE_2D, self._cmap_tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 256, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, data
        )

    # --- overlay ---

    def _get_vel(self) -> np.ndarray | None:
        u = getattr(self.sim, "u", None)
        v = getattr(self.sim, "v", None)
        if u is not None and v is not None:
            return np.stack(
                [np.ascontiguousarray(u), np.ascontiguousarray(v)], axis=2
            )
        return None

    def _draw_overlay(self, vel: np.ndarray | None = None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.scene:
            pen = QPen(QColor(255, 90, 90, 200), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 90, 90, 35))
            for obs in self.scene.obstacles:
                self._draw_obstacle_shape(painter, obs)

        if self.scene:
            pen = QPen(QColor(0, 220, 220, 200), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 220, 220, 60))
            s = (
                min(
                    self.width() / self.scene.width,
                    self.height() / self.scene.height,
                )
                if self.scene.width > 0 and self.scene.height > 0
                else 1
            )
            er = max(4, 4 * s)
            label_font = QFont("monospace", 9)
            for emit in self.scene.emitters:
                wx, wy = self._grid_to_widget(emit.x, emit.y)
                painter.drawEllipse(
                    int(wx - er), int(wy - er), int(er * 2), int(er * 2)
                )
                painter.setFont(label_font)
                painter.drawText(int(wx + er + 4), int(wy + 4), emit.name)

        if self._show_quiver and vel is not None:
            self._draw_quiver(painter, vel)
        if self._show_streamlines and vel is not None:
            self._draw_streamlines(painter, vel)

        if self._drag_start is not None and self._drag_end is not None:
            pen = QPen(QColor(255, 255, 255, 200), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 255, 255, 20))
            sx, sy = self._grid_to_widget(*self._drag_start)
            ex, ey = self._grid_to_widget(*self._drag_end)
            if self.draw_mode == "circle":
                r_grid = math.dist(self._drag_start, self._drag_end)
                s = (
                    min(
                        self.width() / self.scene.width,
                        self.height() / self.scene.height,
                    )
                    if self.scene
                    else 1
                )
                r = r_grid * s
                painter.drawEllipse(int(sx - r), int(sy - r), int(r * 2), int(r * 2))
            elif self.draw_mode == "rect":
                rect = self._widget_rect(sx, sy, ex, ey)
                painter.drawRect(rect)

        for p in self.probes:
            wx, wy = self._grid_to_widget(p.spec.x, p.spec.y)
            ix, iy = int(wx), int(wy)
            pen = QPen(QColor(0, 255, 100, 220), 2)
            painter.setPen(pen)
            painter.drawLine(ix - 6, iy, ix + 6, iy)
            painter.drawLine(ix, iy - 6, ix, iy + 6)
            painter.drawText(ix + 8, iy + 4, p.spec.name)

        if self.draw_mode == "polygon" and len(self._poly_points) > 1:
            pen = QPen(QColor(255, 255, 255, 200), 2)
            painter.setPen(pen)
            pts = [self._grid_to_widget(px, py) for px, py in self._poly_points]
            for i in range(len(pts) - 1):
                x0, y0 = int(pts[i][0]), int(pts[i][1])
                x1, y1 = int(pts[i + 1][0]), int(pts[i + 1][1])
                painter.drawLine(x0, y0, x1, y1)

        self._draw_colorbar(painter)

        if (self.scene is not None
                and self.scene.name == "Untitled"
                and len(self.scene.obstacles) == 0):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 42, 200))
            painter.drawRoundedRect(self.rect().adjusted(40, 40, -40, -40), 12, 12)
            painter.setPen(QColor(56, 189, 248))
            font = painter.font()
            font.setPointSize(22)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                self.rect().adjusted(0, 0, 0, -40),
                Qt.AlignmentFlag.AlignCenter,
                "SStream",
            )
            font.setPointSize(13)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(
                self.rect().adjusted(60, 40, -60, -60),
                Qt.AlignmentFlag.AlignCenter,
                "Click Presets to start a guided flow story,\n"
                "or draw an obstacle to see the flow adapt.",
            )

        painter.end()

    def _draw_obstacle_shape(self, painter: QPainter, obs: ObstacleSpec) -> None:
        s = 1
        if self.scene and self.scene.width > 0 and self.scene.height > 0:
            s = min(self.width() / self.scene.width,
                    self.height() / self.scene.height)
        label_font = QFont("monospace", 10)
        label_font.setBold(True)
        if isinstance(obs, CircleObstacle):
            cx, cy = self._grid_to_widget(obs.x, obs.y)
            r = obs.radius * s
            painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
            painter.setFont(label_font)
            painter.drawText(int(cx + r + 4), int(cy + 4), obs.name)
        elif isinstance(obs, RectObstacle):
            x1, y1 = self._grid_to_widget(obs.x, obs.y)
            x2, y2 = self._grid_to_widget(obs.x + obs.w, obs.y + obs.h)
            rect = self._widget_rect(x1, y1, x2, y2)
            painter.drawRect(rect)
            painter.setFont(label_font)
            painter.drawText(rect.topRight() + QPointF(4, 14), obs.name)
        elif isinstance(obs, PolygonObstacle) and len(obs.points) > 1:
            pts = [self._grid_to_widget(px, py) for px, py in obs.points]
            qt_pts = [QPointF(*pt) for pt in pts]
            if len(pts) >= 3:
                painter.drawPolygon(qt_pts)
            else:
                for i in range(len(pts)):
                    j = (i + 1) % len(pts)
                    x0, y0 = int(pts[i][0]), int(pts[i][1])
                    x1, y1 = int(pts[j][0]), int(pts[j][1])
                    painter.drawLine(x0, y0, x1, y1)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            painter.setFont(label_font)
            painter.drawText(int(cx + 4), int(cy + 4), obs.name)

    @staticmethod
    def _widget_rect(x1: float, y1: float, x2: float, y2: float):
        from PySide6.QtCore import QRect
        x, y = int(min(x1, x2)), int(min(y1, y2))
        w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
        return QRect(x, y, w, h)

    # --- quiver plot ---

    def _draw_quiver(self, painter: QPainter, vel: np.ndarray) -> None:
        h, w = vel.shape[:2]
        spacing = max(2, min(h, w) // 16)
        sw = self.width() / w
        sh = self.height() / h
        u_inflow = getattr(self.sim, "u_inflow", 0.15)
        max_speed = max(u_inflow * 1.5, 0.001)

        pen = QPen(QColor(220, 220, 240, 140), 1)
        painter.setPen(pen)
        brush = QBrush(QColor(220, 220, 240, 180))
        painter.setBrush(brush)

        for y in range(spacing // 2, h, spacing):
            for x in range(spacing // 2, w, spacing):
                if hasattr(self.sim, "obstacles") and self.sim.obstacles[y, x]:
                    continue
                u = float(vel[y, x, 0])
                v = float(vel[y, x, 1])
                speed = math.sqrt(u * u + v * v)
                if speed < 0.001:
                    continue
                cx = x * sw
                cy = y * sh
                length = (speed / max_speed) * spacing * sw * 0.75
                angle = math.atan2(v, u)
                dx = math.cos(angle) * length
                dy = math.sin(angle) * length
                painter.drawLine(
                    int(cx - dx), int(cy - dy), int(cx + dx), int(cy + dy)
                )
                hl = length * 0.35
                ha = 0.5
                ax1 = cx + dx - math.cos(angle - ha) * hl
                ay1 = cy + dy - math.sin(angle - ha) * hl
                ax2 = cx + dx - math.cos(angle + ha) * hl
                ay2 = cy + dy - math.sin(angle + ha) * hl
                painter.drawPolygon(
                    QPointF(cx + dx, cy + dy),
                    QPointF(ax1, ay1),
                    QPointF(ax2, ay2),
                )

    # --- streamlines ---

    def _draw_streamlines(self, painter: QPainter, vel: np.ndarray) -> None:
        h, w = vel.shape[:2]
        sw = self.width() / w
        sh = self.height() / h
        obs = getattr(self.sim, "obstacles", None)

        num_seeds = max(4, h // 8)
        seed_ys = np.linspace(2, h - 3, num_seeds)

        max_steps = 400
        step = 0.5

        for sy in seed_ys:
            x, y = 1.0, float(sy)
            pts = []
            for _ in range(max_steps):
                ix = max(0, min(int(x), w - 2))
                iy = max(0, min(int(y), h - 2))
                if obs is not None and obs[iy, ix]:
                    break

                u = self._bilerp(vel[:, :, 0], x, y)
                v = self._bilerp(vel[:, :, 1], x, y)
                speed = math.sqrt(u * u + v * v)
                if speed < 0.0001:
                    break

                pts.append((int(x * sw), int(y * sh)))
                x += (u / speed) * step
                y += (v / speed) * step

                if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
                    break

            if len(pts) > 1:
                painter.setPen(QPen(QColor(255, 255, 255, 55), 1))
                for i in range(len(pts) - 1):
                    painter.drawLine(
                        pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]
                    )

    @staticmethod
    def _bilerp(field: np.ndarray, x: float, y: float) -> float:
        ix = max(0, min(int(x), field.shape[1] - 2))
        iy = max(0, min(int(y), field.shape[0] - 2))
        fx = x - ix
        fy = y - iy
        return float(
            field[iy, ix] * (1 - fx) * (1 - fy)
            + field[iy, ix + 1] * fx * (1 - fy)
            + field[iy + 1, ix] * (1 - fx) * fy
            + field[iy + 1, ix + 1] * fx * fy
        )

    # --- colorbar ---

    def _draw_colorbar(self, painter: QPainter) -> None:
        bar_w = 18
        bar_h = min(180, self.height() - 40)
        bx = self.width() - bar_w - 20
        by = 24

        cmap_name = MODE_TO_CMAP.get(self._colormap, "viridis")
        lut = CMAP_LUTS.get(cmap_name, CMAP_LUTS["viridis"])

        for i in range(bar_h):
            t = 1.0 - i / (bar_h - 1) if bar_h > 1 else 0.0
            idx = int(t * 255)
            r, g, b = lut[idx]
            painter.setPen(QColor(int(r * 255), int(g * 255), int(b * 255)))
            painter.drawLine(bx, by + i, bx + bar_w - 1, by + i)

        painter.setPen(QPen(QColor(100, 110, 140, 180), 1))
        painter.drawRect(bx, by, bar_w, bar_h)
        font = QFont("monospace", 9)
        painter.setFont(font)
        painter.setPen(QColor(180, 190, 210))
        painter.drawText(bx + bar_w + 6, by + 9, "1.0")
        painter.drawText(bx + bar_w + 6, by + bar_h // 2 + 3, "0.5")
        painter.drawText(bx + bar_w + 6, by + bar_h + 4, "0.0")

    # --- mouse events ---

    def mousePressEvent(self, event) -> None:
        if self.draw_mode is None or self.scene is None:
            super().mousePressEvent(event)
            return
        gx, gy = self._widget_to_grid(event.position().x(), event.position().y())

        if self.draw_mode == "probe":
            ix, iy = int(round(gx)), int(round(gy))
            name = f"Probe_{len(self.scene.probes) + 1}"
            spec = ProbeSpec(name=name, x=ix, y=iy)
            self.probe_placed.emit(spec)
            return

        if self.draw_mode == "polygon":
            self._poly_points.append((gx, gy))
            self.update()
        else:
            self._drag_start = (gx, gy)
            self._drag_end = (gx, gy)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.draw_mode == "polygon" and len(self._poly_points) >= 3:
            pts = [
                (int(round(x)), int(round(y)))
                for x, y in self._poly_points
            ]
            obs = PolygonObstacle(name="Polygon", points=pts)
            self.obstacle_created.emit(obs)
            self._poly_points = []
            self.update()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._drag_start = None
            self._drag_end = None
            self._poly_points = []
            self.update()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.draw_mode == "polygon" and len(self._poly_points) >= 3:
                pts = [
                    (int(round(x)), int(round(y)))
                    for x, y in self._poly_points
                ]
                obs = PolygonObstacle(name="Polygon", points=pts)
                self.obstacle_created.emit(obs)
                self._poly_points = []
                self.update()
                return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.draw_mode is None or self.scene is None:
            super().mouseMoveEvent(event)
            return
        gx, gy = self._widget_to_grid(event.position().x(), event.position().y())
        if self.draw_mode == "polygon":
            if self._poly_points:
                self._poly_points[-1] = (gx, gy)
                self.update()
        elif self._drag_start is not None:
            self._drag_end = (gx, gy)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.draw_mode is None or self.scene is None:
            super().mouseReleaseEvent(event)
            return
        if self.draw_mode == "polygon":
            if len(self._poly_points) >= 3:
                pts = [
                    (int(round(x)), int(round(y)))
                    for x, y in self._poly_points
                ]
                obs = PolygonObstacle(name="Polygon", points=pts)
                self.obstacle_created.emit(obs)
            self._poly_points = []
            self.update()
        elif self._drag_start is not None and self._drag_end is not None:
            gx1, gy1 = self._drag_start
            gx2, gy2 = self._drag_end
            ix1, iy1 = int(round(gx1)), int(round(gy1))
            ix2, iy2 = int(round(gx2)), int(round(gy2))
            if self.draw_mode == "circle":
                r = int(round(math.dist((gx1, gy1), (gx2, gy2))))
                if r > 1:
                    obs = CircleObstacle(name="Circle", x=ix1, y=iy1, radius=r)
                    self.obstacle_created.emit(obs)
            elif self.draw_mode == "rect":
                rx, ry = min(ix1, ix2), min(iy1, iy2)
                rw, rh = abs(ix2 - ix1), abs(iy2 - iy1)
                if rw > 1 and rh > 1:
                    obs = RectObstacle(name="Rect", x=rx, y=ry, w=rw, h=rh)
                    self.obstacle_created.emit(obs)
            self._drag_start = None
            self._drag_end = None
            self.update()

    def resizeGL(self, w: int, h: int) -> None:
        pass
