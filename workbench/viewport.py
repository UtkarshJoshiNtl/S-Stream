from __future__ import annotations

import math

import numpy as np
import OpenGL.platform as _ogl_plat

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
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
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
    glUniform1i,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader  # noqa: E402
from PySide6.QtCore import Qt, Signal  # noqa: E402
from PySide6.QtGui import QColor, QPainter, QPen  # noqa: E402
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

_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 uv;
void main() {
    uv = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_FRAG_SRC = """
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D smokeTex;
void main() {
    float s = texture(smokeTex, uv).r;
    vec3 c = mix(vec3(0.02, 0.02, 0.06), vec3(1.0, 0.75, 0.4), sqrt(s));
    fragColor = vec4(c, 1.0);
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
        self._colormap = "smoke"
        self._tex_init = False
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
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def paintGL(self) -> None:
        if self.sim is None:
            return
        glClearColor(0.02, 0.02, 0.06, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self._upload_smoke()
        glUseProgram(self.shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.shader, "smokeTex"), 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

        self._draw_overlay()

    def _upload_smoke(self) -> None:
        field = self._compute_field()
        field = np.ascontiguousarray(field.astype(np.float32))
        field = np.flipud(field)
        h, w = field.shape
        glBindTexture(GL_TEXTURE_2D, self.texture)
        if not self._tex_init:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w, h, 0, GL_RED, GL_FLOAT, field)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            self._tex_init = True
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, field)

    def _compute_field(self) -> np.ndarray:
        cmap = self._colormap
        if cmap == "smoke":
            if self.sim is None:
                shape = (
                    (self.scene.height, self.scene.width)
                    if self.scene
                    else (128, 128)
                )
                return np.zeros(shape)
            return self.sim.get_smoke()
        vel = self.sim.get_velocity()
        if cmap == "speed":
            speed = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
            mx = max(self.sim.u_inflow * 1.5, float(speed.max()), 0.001)
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
            rho = self.sim.get_density()
            p = rho - 1.0
            mx = max(float(abs(p).max()), 0.001)
            return np.clip(p / mx * 0.5 + 0.5, 0, 1)
        return self.sim.get_smoke()

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw existing obstacles
        if self.scene and self._colormap == "smoke":
            pen = QPen(QColor(255, 80, 80, 160), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 80, 80, 30))
            for obs in self.scene.obstacles:
                self._draw_obstacle_shape(painter, obs)

        # Draw emitters
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
            for emit in self.scene.emitters:
                wx, wy = self._grid_to_widget(emit.x, emit.y)
                painter.drawEllipse(
                    int(wx - er), int(wy - er), int(er * 2), int(er * 2)
                )

        # Draw drag preview
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

        # Draw probes
        for p in self.probes:
            wx, wy = self._grid_to_widget(p.spec.x, p.spec.y)
            ix, iy = int(wx), int(wy)
            pen = QPen(QColor(0, 255, 100, 220), 2)
            painter.setPen(pen)
            painter.drawLine(ix - 6, iy, ix + 6, iy)
            painter.drawLine(ix, iy - 6, ix, iy + 6)
            painter.drawText(ix + 8, iy + 4, p.spec.name)

        # Draw freehand polygon points
        if self.draw_mode == "polygon" and len(self._poly_points) > 1:
            pen = QPen(QColor(255, 255, 255, 200), 2)
            painter.setPen(pen)
            pts = [self._grid_to_widget(px, py) for px, py in self._poly_points]
            for i in range(len(pts) - 1):
                x0, y0 = int(pts[i][0]), int(pts[i][1])
                x1, y1 = int(pts[i + 1][0]), int(pts[i + 1][1])
                painter.drawLine(x0, y0, x1, y1)

        painter.end()

    def _draw_obstacle_shape(self, painter: QPainter, obs: ObstacleSpec) -> None:
        if isinstance(obs, CircleObstacle):
            cx, cy = self._grid_to_widget(obs.x, obs.y)
            if self.scene:
                sx = self.width() / self.scene.width
                sy = self.height() / self.scene.height
                s = min(sx, sy)
            else:
                s = 1
            r = obs.radius * s
            painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
        elif isinstance(obs, RectObstacle):
            x1, y1 = self._grid_to_widget(obs.x, obs.y)
            x2, y2 = self._grid_to_widget(obs.x + obs.w, obs.y + obs.h)
            painter.drawRect(self._widget_rect(x1, y1, x2, y2))
        elif isinstance(obs, PolygonObstacle) and len(obs.points) > 1:
            pts = [self._grid_to_widget(px, py) for px, py in obs.points]
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                x0, y0 = int(pts[i][0]), int(pts[i][1])
                x1, y1 = int(pts[j][0]), int(pts[j][1])
                painter.drawLine(x0, y0, x1, y1)

    @staticmethod
    def _widget_rect(x1: float, y1: float, x2: float, y2: float):
        from PySide6.QtCore import QRect
        x, y = int(min(x1, x2)), int(min(y1, y2))
        w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
        return QRect(x, y, w, h)

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
        if self.draw_mode == "polygon" and len(self._poly_points) >= 2:
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
