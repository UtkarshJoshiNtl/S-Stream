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
from PySide6.QtCore import QPointF, Qt, Signal  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen  # noqa: E402
from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: E402

from engines.base import SimEngine  # noqa: E402
from resources.colormaps import CMAP_LUTS, MODE_TO_CMAP  # noqa: E402
from scene.probe import Probe  # noqa: E402
from scene.scene import (  # noqa: E402
    AirfoilObstacle,
    ChannelObstacle,
    CircleObstacle,
    EllipseObstacle,
    ImageObstacle,
    LatticeObstacle,
    ObstacleSpec,
    PolygonObstacle,
    ProbeSpec,
    RectObstacle,
    Scene,
    STLObstacle,
)

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
        self._show_contours = False
        self._show_force_arrows = False
        self._show_particles = False
        self.draw_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_end: tuple[float, float] | None = None
        self._poly_points: list[tuple[float, float]] = []
        self._overlay_frame = 0
        self._streamline_cache: list | None = None
        self._contour_cache: tuple[list, int] | None = None
        self._particle_cache: tuple | None = None
        self._colorbar_pixmap = None
        self._colorbar_cmap: str | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_sim(self, sim: SimEngine) -> None:
        self.sim = sim

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def set_probes(self, probes: list[Probe]) -> None:
        self.probes = probes

    def get_colormap(self) -> str:
        return self._colormap

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

    def set_show_contours(self, show: bool) -> None:
        self._show_contours = show
        self.update()

    def set_show_force_arrows(self, show: bool) -> None:
        self._show_force_arrows = show
        self.update()

    def set_show_particles(self, show: bool) -> None:
        self._show_particles = show
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
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
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
        self._overlay_frame += 1
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
        obs = self.sim.get_obstacles()
        self._draw_overlay(vel, obs)

    def _upload_smoke(self) -> None:
        cmap = self._colormap
        try:
            field = self.sim.get_field(cmap)
        except ValueError:
            field = self.sim.get_field("smoke")

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
        return self.sim.get_velocity_view()

    def _draw_overlay(
        self, vel: np.ndarray | None = None, obs: np.ndarray | None = None
    ) -> None:
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
            self._draw_quiver(painter, vel, obs)
        if self._show_streamlines and vel is not None:
            self._draw_streamlines(painter, vel, obs)
        if self._show_contours:
            self._draw_pressure_contours(painter)
        if self._show_force_arrows and vel is not None:
            self._draw_force_arrows(painter, vel, obs)
        if self._show_particles:
            self._draw_particles(painter)

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

        if (
            self.scene is not None
            and self.scene.name == "Untitled"
            and len(self.scene.obstacles) == 0
        ):
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
            s = min(self.width() / self.scene.width, self.height() / self.scene.height)
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
            painter.drawText(QPointF(rect.topRight()) + QPointF(4, 14), obs.name)
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
        elif isinstance(obs, EllipseObstacle):
            cx, cy = self._grid_to_widget(obs.x, obs.y)
            rx = obs.rx * s
            ry = obs.ry * s
            painter.drawEllipse(int(cx - rx), int(cy - ry), int(rx * 2), int(ry * 2))
            painter.setFont(label_font)
            painter.drawText(int(cx + rx + 4), int(cy + 4), obs.name)
        elif isinstance(obs, STLObstacle):
            x1, y1 = self._grid_to_widget(obs.offset_x, obs.offset_y)
            painter.setFont(label_font)
            painter.drawText(int(x1), int(y1), obs.name)
            painter.drawText(int(x1), int(y1 + 14), "(STL mesh)")
        elif isinstance(obs, ImageObstacle):
            painter.setFont(label_font)
            painter.drawText(20, 20, obs.name)
            painter.drawText(20, 34, "(Image mask)")
        elif isinstance(obs, AirfoilObstacle):
            cx, cy = self._grid_to_widget(obs.x, obs.y)
            painter.setFont(label_font)
            painter.drawText(int(cx + 10), int(cy), obs.name)
            painter.drawText(int(cx + 10), int(cy + 14), f"NACA {obs.naca_code}")
        elif isinstance(obs, ChannelObstacle):
            x1, y1 = self._grid_to_widget(obs.x, obs.y)
            x2, y2 = self._grid_to_widget(obs.x + obs.w, obs.y + obs.h)
            rect = self._widget_rect(x1, y1, x2, y2)
            painter.drawRect(rect)
            painter.setFont(label_font)
            painter.drawText(QPointF(rect.topRight()) + QPointF(4, 14), obs.name)
        elif isinstance(obs, LatticeObstacle):
            x1, y1 = self._grid_to_widget(obs.x, obs.y)
            x2, y2 = self._grid_to_widget(obs.x + obs.w, obs.y + obs.h)
            rect = self._widget_rect(x1, y1, x2, y2)
            painter.drawRect(rect)
            painter.setFont(label_font)
            painter.drawText(QPointF(rect.topRight()) + QPointF(4, 14), obs.name)

    @staticmethod
    def _widget_rect(x1: float, y1: float, x2: float, y2: float):
        from PySide6.QtCore import QRect

        x, y = int(min(x1, x2)), int(min(y1, y2))
        w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
        return QRect(x, y, w, h)

    # --- quiver plot ---

    def _draw_quiver(
        self, painter: QPainter, vel: np.ndarray, obs: np.ndarray | None = None
    ) -> None:
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
                if obs is not None and obs[y, x]:
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
                painter.drawLine(int(cx - dx), int(cy - dy), int(cx + dx), int(cy + dy))
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

    def _draw_streamlines(
        self, painter: QPainter, vel: np.ndarray, obs: np.ndarray | None = None
    ) -> None:
        h, w = vel.shape[:2]
        sw = self.width() / w
        sh = self.height() / h

        if self._overlay_frame % 6 != 0 and self._streamline_cache is not None:
            pen = QPen(QColor(255, 255, 255, 55), 1)
            painter.setPen(pen)
            for pts in self._streamline_cache:
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            return

        num_seeds = max(4, h // 8)
        seed_ys = np.linspace(2, h - 3, num_seeds)
        max_steps = 400
        step_size = 0.5

        xs = np.ones(num_seeds, dtype=np.float64)
        ys = seed_ys.copy()
        alive = np.ones(num_seeds, dtype=bool)
        all_pts: list[list[tuple[int, int]]] = [[] for _ in range(num_seeds)]

        for _ in range(max_steps):
            if not alive.any():
                break
            idx_alive = np.where(alive)[0]
            cur_x = xs[alive]
            cur_y = ys[alive]

            ix = np.clip(cur_x.astype(int), 0, w - 2)
            iy = np.clip(cur_y.astype(int), 0, h - 2)
            fx = cur_x - ix
            fy = cur_y - iy

            u_interp = (
                vel[iy, ix, 0] * (1 - fx) * (1 - fy)
                + vel[iy, ix + 1, 0] * fx * (1 - fy)
                + vel[iy + 1, ix, 0] * (1 - fx) * fy
                + vel[iy + 1, ix + 1, 0] * fx * fy
            )
            v_interp = (
                vel[iy, ix, 1] * (1 - fx) * (1 - fy)
                + vel[iy, ix + 1, 1] * fx * (1 - fy)
                + vel[iy + 1, ix, 1] * (1 - fx) * fy
                + vel[iy + 1, ix + 1, 1] * fx * fy
            )
            speed = np.sqrt(u_interp**2 + v_interp**2)

            too_slow = speed < 0.0001
            oob = (cur_x < 0) | (cur_x >= w - 1) | (cur_y < 0) | (cur_y >= h - 1)
            if obs is not None:
                hit_obs = obs[iy, ix]
            else:
                hit_obs = np.zeros(len(cur_x), dtype=bool)

            dead = too_slow | oob | hit_obs
            cur_alive = ~dead
            alive[idx_alive[dead]] = False

            safe_speed = np.where(cur_alive, speed, 1.0)
            for i_local in range(len(idx_alive)):
                i_global = idx_alive[i_local]
                px = int(xs[i_global] * sw)
                py = int(ys[i_global] * sh)
                all_pts[i_global].append((px, py))

            xs[alive] += (u_interp[cur_alive] / safe_speed[cur_alive]) * step_size
            ys[alive] += (v_interp[cur_alive] / safe_speed[cur_alive]) * step_size

        self._streamline_cache = all_pts
        pen = QPen(QColor(255, 255, 255, 55), 1)
        painter.setPen(pen)
        for pts in all_pts:
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

    # --- pressure contour lines ---

    def _draw_pressure_contours(self, painter: QPainter) -> None:
        if self._overlay_frame % 6 != 0 and self._contour_cache is not None:
            pen = QPen(QColor(255, 255, 255, 60), 1)
            painter.setPen(pen)
            lines, sw, sh = self._contour_cache
            for ex1, ey1, ex2, ey2 in lines:
                x1, y1 = int(ex1 * sw), int(ey1 * sh)
                x2, y2 = int(ex2 * sw), int(ey2 * sh)
                painter.drawLine(x1, y1, x2, y2)
            return

        p = self.sim.get_pressure()
        h, w = p.shape
        sw = self.width() / w
        sh = self.height() / h
        mx = max(float(np.percentile(np.abs(p), 95)), 0.001)
        levels = np.linspace(-mx, mx, 11)
        all_lines = []
        for level in levels:
            tl = p[:-1:2, :-1:2]
            tr = p[:-1:2, 1::2]
            br = p[1::2, 1::2]
            bl = p[1::2, :-1:2]
            idx = (
                ((tl >= level).astype(np.int32) << 3)
                | ((tr >= level).astype(np.int32) << 2)
                | ((br >= level).astype(np.int32) << 1)
                | ((bl >= level).astype(np.int32))
            )
            active = (idx > 0) & (idx < 15)
            cell_ys, cell_xs = np.where(active)
            for cy, cx in zip(cell_ys, cell_xs, strict=False):
                x2, y2 = cx * 2, cy * 2
                tl_v = p[y2, x2]
                tr_v = p[y2, x2 + 1]
                br_v = p[y2 + 1, x2 + 1]
                bl_v = p[y2 + 1, x2]
                cell_idx = int(idx[cy, cx])
                edges = self._ms_edges(cell_idx, level, tl_v, tr_v, br_v, bl_v, x2, y2)
                all_lines.extend(edges)
        self._contour_cache = (all_lines, sw, sh)
        pen = QPen(QColor(255, 255, 255, 60), 1)
        painter.setPen(pen)
        for ex1, ey1, ex2, ey2 in all_lines:
            painter.drawLine(int(ex1 * sw), int(ey1 * sh), int(ex2 * sw), int(ey2 * sh))

    def _marching_squares_contour(
        self,
        painter: QPainter,
        field: np.ndarray,
        level: float,
        sw: float,
        sh: float,
    ) -> None:
        h, w = field.shape
        for y in range(0, h - 1, 2):
            for x in range(0, w - 1, 2):
                tl = field[y, x]
                tr = field[y, x + 1]
                br = field[y + 1, x + 1]
                bl = field[y + 1, x]
                idx = 0
                if tl >= level:
                    idx |= 8
                if tr >= level:
                    idx |= 4
                if br >= level:
                    idx |= 2
                if bl >= level:
                    idx |= 1
                if idx == 0 or idx == 15:
                    continue
                edges = self._ms_edges(idx, level, tl, tr, br, bl, x, y)
                for ex1, ey1, ex2, ey2 in edges:
                    painter.drawLine(
                        int(ex1 * sw),
                        int(ey1 * sh),
                        int(ex2 * sw),
                        int(ey2 * sh),
                    )

    @staticmethod
    def _ms_edges(
        idx: int,
        level: float,
        tl: float,
        tr: float,
        br: float,
        bl: float,
        x: float,
        y: float,
    ) -> list[tuple[float, float, float, float]]:
        def lerp(a: float, b: float) -> float:
            d = b - a
            return (level - a) / d if abs(d) > 1e-10 else 0.5

        top = (x + lerp(tl, tr), y)
        right = (x + 1, y + lerp(tr, br))
        bottom = (x + lerp(bl, br), y + 1)
        left = (x, y + lerp(tl, bl))

        lookup = {
            1: [(left, bottom)],
            2: [(bottom, right)],
            3: [(left, right)],
            4: [(top, right)],
            5: [(top, left), (bottom, right)],
            6: [(top, bottom)],
            7: [(top, left)],
            8: [(top, left)],
            9: [(top, bottom)],
            10: [(top, right), (bottom, left)],
            11: [(top, right)],
            12: [(left, right)],
            13: [(bottom, right)],
            14: [(left, bottom)],
        }
        pairs = lookup.get(idx, [])
        return [(a[0], a[1], b[0], b[1]) for a, b in pairs]

    # --- force arrows on obstacles ---

    def _draw_force_arrows(
        self, painter: QPainter, vel: np.ndarray, obs: np.ndarray | None = None
    ) -> None:
        if not self.scene or not self.scene.obstacles:
            return
        h, w = vel.shape[:2]
        sw = self.width() / w
        sh = self.height() / h
        if obs is None:
            obs = self.sim.get_obstacles()
        u_inflow = getattr(self.sim, "u_inflow", 0.15)

        for obs_spec in self.scene.obstacles:
            cx, cy, cw, ch = self._obstacle_bounds(obs_spec)
            if cw == 0 or ch == 0:
                continue
            force_x, force_y = self._estimate_obstacle_force(
                vel, obs, cx, cy, cw, ch, u_inflow
            )
            magnitude = math.sqrt(force_x**2 + force_y**2)
            if magnitude < 1e-6:
                continue
            px = (cx + cw / 2) * sw
            py = (cy + ch / 2) * sh
            max_arrow = min(self.width(), self.height()) * 0.1
            scale = min(max_arrow / max(magnitude, 1e-6), 10.0)
            dx = force_x * scale
            dy = force_y * scale
            angle = math.atan2(force_y, force_x)
            pen = QPen(QColor(255, 200, 50, 220), 2)
            painter.setPen(pen)
            brush = QBrush(QColor(255, 200, 50, 200))
            painter.setBrush(brush)
            painter.drawLine(int(px), int(py), int(px + dx), int(py + dy))
            hl = 8
            ha = 0.5
            ax1 = px + dx - math.cos(angle - ha) * hl
            ay1 = py + dy - math.sin(angle - ha) * hl
            ax2 = px + dx - math.cos(angle + ha) * hl
            ay2 = py + dy - math.sin(angle + ha) * hl
            painter.drawPolygon(
                QPointF(px + dx, py + dy),
                QPointF(ax1, ay1),
                QPointF(ax2, ay2),
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            font = QFont("monospace", 8)
            painter.setFont(font)
            painter.setPen(QColor(255, 200, 50, 200))
            painter.drawText(int(px + dx + 6), int(py + dy - 4), f"{magnitude:.3f}")

    @staticmethod
    def _obstacle_bounds(obs: ObstacleSpec) -> tuple[int, int, int, int]:
        has_rect = (
            hasattr(obs, "x")
            and hasattr(obs, "y")
            and hasattr(obs, "w")
            and hasattr(obs, "h")
        )
        if has_rect:
            return obs.x, obs.y, obs.w, obs.h
        has_circle = hasattr(obs, "x") and hasattr(obs, "y") and hasattr(obs, "radius")
        if has_circle:
            r = obs.radius
            return obs.x - r, obs.y - r, 2 * r, 2 * r
        return 0, 0, 0, 0

    @staticmethod
    def _estimate_obstacle_force(
        vel: np.ndarray,
        obs: np.ndarray,
        cx: int,
        cy: int,
        cw: int,
        ch: int,
        u_inflow: float,
    ) -> tuple[float, float]:
        h, w = vel.shape[:2]
        x1 = max(1, cx)
        x2 = min(w - 1, cx + cw)
        y1 = max(1, cy)
        y2 = min(h - 1, cy + ch)
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0
        region_u = vel[y1:y2, x1:x2, 0]
        region_v = vel[y1:y2, x1:x2, 1]
        obs_region = obs[y1:y2, x1:x2]
        interior = obs_region
        if not interior.any():
            return 0.0, 0.0
        drag_x = float(np.sum(region_u[interior]))
        drag_y = float(np.sum(region_v[interior]))
        return -drag_x, -drag_y

    # --- particles ---

    def _draw_particles(self, painter: QPainter) -> None:
        if self.sim is None:
            return
        tracer = self.sim.get_particle_tracer()
        if tracer is None or tracer.count == 0:
            return

        h, w = (
            self.scene.height,
            (
                self.scene.width
                if self.scene
                else (self.sim.grid_shape[-1], self.sim.grid_shape[-2])
            ),
        )
        sw = self.width() / w
        sh = self.height() / h

        if self._overlay_frame % 4 != 0 and self._particle_cache is not None:
            trail_lines, dot_positions = self._particle_cache
            pen = QPen(QColor(100, 220, 255, 100), 1)
            painter.setPen(pen)
            for line in trail_lines:
                painter.drawLine(int(line[0]), int(line[1]), int(line[2]), int(line[3]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 220))
            dot_r = max(2, min(4, int(min(sw, sh) * 0.3)))
            for px, py in dot_positions:
                x0 = int(px - dot_r)
                y0 = int(py - dot_r)
                painter.drawEllipse(x0, y0, dot_r * 2, dot_r * 2)
            return

        trails = tracer.get_trails()
        trail_lines = []
        if trails is not None and trails.shape[0] > 1 and trails.shape[1] > 0:
            trail_len = trails.shape[0]
            n = trails.shape[1]
            num_alpha_buckets = 5
            for bucket in range(num_alpha_buckets):
                t_min = bucket / num_alpha_buckets
                t_max = (bucket + 1) / num_alpha_buckets
                alpha = int(40 + 180 * (t_min + t_max) / 2)
                pen = QPen(QColor(100, 220, 255, alpha), 1)
                painter.setPen(pen)
                t_start = max(0, int(t_min * (trail_len - 1)))
                t_end = min(trail_len - 1, int(t_max * (trail_len - 1)))
                for t in range(t_start, t_end):
                    px_list = trails[t, :, 0] * sw
                    py_list = trails[t, :, 1] * sh
                    px_list_next = trails[t + 1, :, 0] * sw
                    py_list_next = trails[t + 1, :, 1] * sh
                    for i in range(n):
                        x0, y0 = float(px_list[i]), float(py_list[i])
                        x1, y1 = float(px_list_next[i]), float(py_list_next[i])
                        if abs(x1 - x0) < w * sw and abs(y1 - y0) < h * sh:
                            trail_lines.append((x0, y0, x1, y1))
                            painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        dot_positions = []
        positions = tracer.get_positions()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 220))
        dot_r = max(2, min(4, int(min(sw, sh) * 0.3)))
        for i in range(len(positions)):
            px = positions[i, 0] * sw
            py = positions[i, 1] * sh
            dot_positions.append((px, py))
            painter.drawEllipse(int(px - dot_r), int(py - dot_r), dot_r * 2, dot_r * 2)

        self._particle_cache = (trail_lines, dot_positions)

    # --- colorbar ---

    def _draw_colorbar(self, painter: QPainter) -> None:
        from PySide6.QtGui import QPixmap

        from resources.colormaps import FIELD_REGISTRY

        bar_w = 18
        bar_h = min(180, self.height() - 40)
        bx = self.width() - bar_w - 20
        by = 24

        cmap_name = MODE_TO_CMAP.get(self._colormap, "viridis")
        if self._colorbar_cmap != cmap_name or self._colorbar_pixmap is None:
            pm = QPixmap(bar_w, bar_h)
            pm.fill(QColor(0, 0, 0, 0))
            pm_paint = QPainter(pm)
            lut = CMAP_LUTS.get(cmap_name, CMAP_LUTS["viridis"])
            for i in range(bar_h):
                t = 1.0 - i / (bar_h - 1) if bar_h > 1 else 0.0
                idx = int(t * 255)
                r, g, b = lut[idx]
                pm_paint.setPen(QColor(int(r * 255), int(g * 255), int(b * 255)))
                pm_paint.drawLine(0, i, bar_w - 1, i)
            pm_paint.end()
            self._colorbar_pixmap = pm
            self._colorbar_cmap = cmap_name

        painter.drawPixmap(bx, by, self._colorbar_pixmap)
        painter.setPen(QPen(QColor(100, 110, 140, 180), 1))
        painter.drawRect(bx, by, bar_w, bar_h)
        font = QFont("monospace", 9)
        painter.setFont(font)
        painter.setPen(QColor(180, 190, 210))
        info = FIELD_REGISTRY.get(self._colormap)
        label = info.label if info else self._colormap.capitalize()
        painter.drawText(bx - 4, by - 6, label)

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
            pts = [(int(round(x)), int(round(y))) for x, y in self._poly_points]
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
                pts = [(int(round(x)), int(round(y))) for x, y in self._poly_points]
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
                pts = [(int(round(x)), int(round(y))) for x, y in self._poly_points]
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
