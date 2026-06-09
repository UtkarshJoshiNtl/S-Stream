from __future__ import annotations

import ctypes

import numpy as np

# PyOpenGL's context detection fails when GLFW manages the GL context.
# Monkey-patch GetCurrentContext to always return a sentinel (used as a dict key).
import OpenGL.platform as _ogl_plat

_ogl_plat.GetCurrentContext = lambda: object()

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_R16F,
    GL_RED,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_3D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_R,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glClear,
    glClearColor,
    glDisable,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glTexImage2D,
    glTexImage3D,
    glTexParameteri,
    glTexSubImage2D,
    glTexSubImage3D,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from engines.base import SimEngine

_VOLUME_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 uv;
void main() {
    uv = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_VOLUME_FRAG_SRC = """
#version 330 core
out vec4 fragColor;
in vec2 uv;
uniform sampler3D volume;
uniform mat4 invMVP;
uniform vec3 volExtent;

vec4 transfer(float d) {
    float a = clamp(d * 4.0, 0.0, 0.4);
    vec3 c = mix(vec3(0.02, 0.02, 0.06), vec3(1.0, 0.75, 0.4), sqrt(d));
    return vec4(c, a) * step(0.003, d);
}

void main() {
    vec4 ndcNear = vec4(uv * 2.0 - 1.0, -1.0, 1.0);
    vec4 ndcFar  = vec4(uv * 2.0 - 1.0,  1.0, 1.0);
    vec4 wn = invMVP * ndcNear;
    vec4 wf = invMVP * ndcFar;
    vec3 nearPos = wn.xyz / wn.w;
    vec3 farPos  = wf.xyz / wf.w;
    vec3 dir = normalize(farPos - nearPos);

    vec3 boxMin = -volExtent / 2.0;
    vec3 boxMax =  volExtent / 2.0;

    vec3 t0 = (boxMin - nearPos) / dir;
    vec3 t1 = (boxMax - nearPos) / dir;
    vec3 tn = min(t0, t1);
    vec3 tf = max(t0, t1);
    float tNear = max(max(tn.x, tn.y), max(tn.z, 0.0));
    float tFar  = min(min(tf.x, tf.y), tf.z);

    if (tNear >= tFar) discard;

    vec4 color = vec4(0.0);
    float t = tNear;
    float dt = (tFar - tNear) / 160.0;
    for (int i = 0; i < 160 && color.a < 0.95; i++) {
        vec3 pos = nearPos + t * dir;
        vec3 tex = (pos - boxMin) / volExtent;
        float s = texture(volume, tex).r;
        vec4 c = transfer(s);
        c.rgb *= c.a;
        color += c * (1.0 - color.a);
        t += dt;
    }
    color.rgb = pow(color.rgb, vec3(1.0 / 2.2));
    fragColor = color;
}
"""

_SLICE_FRAG_SRC = """
#version 330 core
out vec4 fragColor;
uniform sampler3D volume;
uniform float sliceZ;
in vec2 uv;
void main() {
    float s = texture(volume, vec3(uv, sliceZ)).r;
    vec3 c = mix(vec3(0.02, 0.02, 0.06), vec3(1.0, 0.75, 0.4), sqrt(s));
    fragColor = vec4(c, 1.0);
}
"""

_2D_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 uv;
void main() {
    uv = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_2D_FRAG_SRC = """
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D smokeTex;

vec4 transfer(float d) {
    float a = clamp(d * 4.0, 0.0, 0.4);
    vec3 c = mix(vec3(0.02, 0.02, 0.06), vec3(1.0, 0.75, 0.4), sqrt(d));
    return vec4(c, a);
}

void main() {
    float s = texture(smokeTex, uv).r;
    fragColor = transfer(s);
}
"""


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fov_y / 2.0)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ]
    )


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array(
        [
            [s[0], s[1], s[2], -np.dot(s, eye)],
            [u[0], u[1], u[2], -np.dot(u, eye)],
            [-f[0], -f[1], -f[2], np.dot(f, eye)],
            [0, 0, 0, 1],
        ]
    )


class Renderer:
    """Engine-agnostic OpenGL renderer for 2D and 3D smoke fields."""

    def __init__(self, sim: SimEngine, win_w: int, win_h: int) -> None:
        self.sim = sim
        self.win_w = win_w
        self.win_h = win_h

        if sim.ndim == 3:
            self._init_3d(sim)
        else:
            self._init_2d(sim)

        # Camera state (3D only)
        self.theta = np.pi * 0.25
        self.phi = np.pi * 0.35
        self.radius = 1.8 * max(sim.grid_shape)
        self._update_camera_3d()

        self.bg_color = (0.02, 0.02, 0.06)

    # --- 3D init ---

    def _init_3d(self, sim: SimEngine) -> None:
        self.vol_shader = compileProgram(
            compileShader(_VOLUME_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_VOLUME_FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        self.slice_shader = compileProgram(
            compileShader(_VOLUME_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_SLICE_FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        d, h, w = sim.grid_shape
        self.extent = np.array([w, h, d], dtype=np.float32)
        self.volume_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.volume_tex)
        glTexImage3D(
            GL_TEXTURE_3D,
            0,
            GL_R16F,
            w,
            h,
            d,
            0,
            GL_RED,
            GL_FLOAT,
            None,
        )
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        self._init_fs_quad()
        self.view_mode = "volume"
        self.slice_z = d // 2

    # --- 2D init ---

    def _init_2d(self, sim: SimEngine) -> None:
        self.tex2d_shader = compileProgram(
            compileShader(_2D_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_2D_FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        h, w = sim.grid_shape
        self.smoke_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.smoke_tex)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R16F,
            w,
            h,
            0,
            GL_RED,
            GL_FLOAT,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self._init_tex_quad()
        self._init_fs_quad()

    # --- shared GL resources ---

    def _init_fs_quad(self) -> None:
        verts = np.array(
            [
                -1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
            ],
            dtype=np.float32,
        )
        self.fs_vao = glGenVertexArrays(1)
        self.fs_vbo = glGenBuffers(1)
        glBindVertexArray(self.fs_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.fs_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def _init_tex_quad(self) -> None:
        """Quad with UV coords for 2D texture rendering."""
        verts = np.array(
            [
                -1,
                -1,
                0,
                0,
                1,
                -1,
                1,
                0,
                1,
                1,
                1,
                1,
                -1,
                -1,
                0,
                0,
                1,
                1,
                1,
                1,
                -1,
                1,
                0,
                1,
            ],
            dtype=np.float32,
        )
        self.tex_vao = glGenVertexArrays(1)
        self.tex_vbo = glGenBuffers(1)
        glBindVertexArray(self.tex_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.tex_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    # --- camera ---

    def _update_camera_3d(self) -> None:
        cp = np.array(
            [
                self.radius * np.sin(self.phi) * np.cos(self.theta),
                self.radius * np.cos(self.phi),
                self.radius * np.sin(self.phi) * np.sin(self.theta),
            ],
            dtype=np.float32,
        )
        center = np.zeros(3, dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        view = _look_at(cp, center, up)
        aspect = self.win_w / self.win_h
        proj = _perspective(np.radians(45), aspect, 0.1, self.radius * 3)
        self.mvp = proj @ view
        self.inv_mvp = np.linalg.inv(self.mvp)

    # --- per-frame upload ---

    def upload(self) -> None:
        smoke = self.sim.get_smoke()
        if self.sim.ndim == 3:
            s = np.ascontiguousarray(smoke.astype(np.float32))
            glBindTexture(GL_TEXTURE_3D, self.volume_tex)
            glTexSubImage3D(
                GL_TEXTURE_3D,
                0,
                0,
                0,
                0,
                self.extent[0],
                self.extent[1],
                self.extent[2],
                GL_RED,
                GL_FLOAT,
                s,
            )
        else:
            s = np.ascontiguousarray(smoke.astype(np.float32))
            glBindTexture(GL_TEXTURE_2D, self.smoke_tex)
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.sim.grid_shape[1],
                self.sim.grid_shape[0],
                GL_RED,
                GL_FLOAT,
                s,
            )

    # --- draw calls ---

    def render(self) -> None:
        glClearColor(*self.bg_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.upload()

        if self.sim.ndim == 3:
            self._render_3d()
        else:
            self._render_2d()

    def _render_3d(self) -> None:
        glDisable(GL_DEPTH_TEST)
        if self.view_mode == "volume":
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
            glUseProgram(self.vol_shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self.volume_tex)
            glUniform1i(glGetUniformLocation(self.vol_shader, "volume"), 0)
            glUniformMatrix4fv(
                glGetUniformLocation(self.vol_shader, "invMVP"),
                1,
                GL_TRUE,
                self.inv_mvp.flatten(),
            )
            glUniform3f(
                glGetUniformLocation(self.vol_shader, "volExtent"),
                self.extent[0],
                self.extent[1],
                self.extent[2],
            )
            glBindVertexArray(self.fs_vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)
            glUseProgram(0)
        else:
            glDisable(GL_BLEND)
            z = np.clip(self.slice_z, 0, int(self.extent[2]) - 1)
            slice_norm = z / (self.extent[2] - 1) if self.extent[2] > 1 else 0.0
            glUseProgram(self.slice_shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self.volume_tex)
            glUniform1i(glGetUniformLocation(self.slice_shader, "volume"), 0)
            glUniform1f(glGetUniformLocation(self.slice_shader, "sliceZ"), slice_norm)
            glBindVertexArray(self.fs_vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)
            glUseProgram(0)

    def _render_2d(self) -> None:
        glUseProgram(self.tex2d_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.smoke_tex)
        glUniform1i(glGetUniformLocation(self.tex2d_shader, "smokeTex"), 0)
        glBindVertexArray(self.tex_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    # --- interactive control ---

    def orbit(self, dx: float, dy: float) -> None:
        if self.sim.ndim != 3:
            return
        self.theta += dx * 0.005
        self.phi = np.clip(self.phi + dy * 0.005, 0.05, np.pi - 0.05)
        self._update_camera_3d()

    def zoom(self, factor: float) -> None:
        if self.sim.ndim != 3:
            return
        self.radius = np.clip(self.radius * factor, 1.0, 500.0)
        self._update_camera_3d()

    def resize(self, w: int, h: int) -> None:
        self.win_w = w
        self.win_h = h
        if self.sim.ndim == 3:
            self._update_camera_3d()
