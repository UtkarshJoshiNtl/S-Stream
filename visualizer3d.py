from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np
import pygame
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_R16F,
    GL_RED,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_3D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_R,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
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
    glPixelStorei,
    glTexImage2D,
    glTexImage3D,
    glTexParameteri,
    glTexSubImage3D,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    GL_UNPACK_ALIGNMENT,
    GL_RGBA,
    GL_NEAREST,
    GL_TEXTURE_2D,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    glBlendFunc,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import (
    glBufferSubData,
    glGetString,
    GL_VERSION,
    glDeleteTextures,
)

if TYPE_CHECKING:
    from cpu_lbm3d import CPULBM3D

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

_HUD_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 uv;
void main() {
    uv = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_HUD_FRAG_SRC = """
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D hudTex;
void main() {
    fragColor = texture(hudTex, uv);
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


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fov_y / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ])


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([
        [ s[0],  s[1],  s[2], -np.dot(s, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [0, 0, 0, 1],
    ])


class FluidVisualizer3D:
    """OpenGL 3.3 volume renderer for 3D LBM smoke data.
    Supports ray-marched volume rendering and 2D slice views.
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        depth: int = 64,
        scale: int = 4,
    ) -> None:
        self.vol_w = width
        self.vol_h = height
        self.vol_d = depth
        self.win_w = width * scale
        self.win_h = height * scale

        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MAJOR_VERSION, 3
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MINOR_VERSION, 3
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_CORE,
        )
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CuFloda 3D - Fluid Simulation")

        version = glGetString(GL_VERSION).decode()
        major = int(version.split('.')[0])
        if major < 3:
            raise RuntimeError(f"OpenGL 3.3+ required, got {version}")

        self.volume_shader = compileProgram(
            compileShader(_VOLUME_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_VOLUME_FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        self.hud_shader = compileProgram(
            compileShader(_HUD_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_HUD_FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        self.slice_shader = compileProgram(
            compileShader(_VOLUME_VERT_SRC, GL_VERTEX_SHADER),
            compileShader(_SLICE_FRAG_SRC, GL_FRAGMENT_SHADER),
        )

        self._init_volume_texture()
        self._init_fullscreen_quad()
        self._init_hud_quad()

        self.theta = np.pi * 0.25
        self.phi = np.pi * 0.35
        self.extent = np.array([width, height, depth], dtype=np.float32)
        self.radius = max(width, height, depth) * 1.8
        self._update_camera()

        self.font = pygame.font.Font(None, 28)
        self._hud_texture_cache: dict[str, tuple[int, int, int]] = {}
        self.bg_color = (0.02, 0.02, 0.06)

        self.paused = False
        self.running = True
        self.drawing_obstacle = False
        self.emitter_mode = False
        self.view_mode = 'volume'
        self.slice_z = depth // 2

    def _init_volume_texture(self) -> None:
        self.volume_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.volume_tex)
        glTexImage3D(
            GL_TEXTURE_3D, 0, GL_R16F,
            self.vol_w, self.vol_h, self.vol_d, 0,
            GL_RED, GL_FLOAT, None,
        )
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def _init_fullscreen_quad(self) -> None:
        verts = np.array([
            -1, -1,   1, -1,   1,  1,
            -1, -1,   1,  1,  -1,  1,
        ], dtype=np.float32)
        self.fs_vao = glGenVertexArrays(1)
        self.fs_vbo = glGenBuffers(1)
        glBindVertexArray(self.fs_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.fs_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def _init_hud_quad(self) -> None:
        verts = np.array([
            -1, -1,  0, 1,
             1, -1,  1, 1,
             1,  1,  1, 0,
            -1, -1,  0, 1,
             1,  1,  1, 0,
            -1,  1,  0, 0,
        ], dtype=np.float32)
        self.hud_vao = glGenVertexArrays(1)
        self.hud_vbo = glGenBuffers(1)
        glBindVertexArray(self.hud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def _update_camera(self) -> None:
        cp = np.array([
            self.radius * np.sin(self.phi) * np.cos(self.theta),
            self.radius * np.cos(self.phi),
            self.radius * np.sin(self.phi) * np.sin(self.theta),
        ], dtype=np.float32)
        self.cam_pos = cp
        center = np.zeros(3, dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        view = _look_at(cp, center, up)
        aspect = self.win_w / self.win_h
        proj = _perspective(np.radians(45), aspect, 0.1, self.radius * 3)
        self.mvp = proj @ view
        self.inv_mvp = np.linalg.inv(self.mvp)

    def upload_smoke(self, smoke: np.ndarray) -> None:
        s = np.ascontiguousarray(smoke.astype(np.float32))
        glBindTexture(GL_TEXTURE_3D, self.volume_tex)
        glTexSubImage3D(
            GL_TEXTURE_3D, 0, 0, 0, 0,
            self.vol_w, self.vol_h, self.vol_d,
            GL_RED, GL_FLOAT, s,
        )

    def render_volume(self) -> None:
        glUseProgram(self.volume_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.volume_tex)
        glUniform1i(glGetUniformLocation(self.volume_shader, "volume"), 0)
        glUniformMatrix4fv(
            glGetUniformLocation(self.volume_shader, "invMVP"),
            1, GL_TRUE, self.inv_mvp.flatten(),
        )
        glUniform3f(
            glGetUniformLocation(self.volume_shader, "volExtent"),
            self.extent[0], self.extent[1], self.extent[2],
        )
        glBindVertexArray(self.fs_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    def render_slice(self) -> None:
        z = np.clip(self.slice_z, 0, self.vol_d - 1)
        slice_norm = z / (self.vol_d - 1)
        glUseProgram(self.slice_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.volume_tex)
        glUniform1i(glGetUniformLocation(self.slice_shader, "volume"), 0)
        glUniform1f(glGetUniformLocation(self.slice_shader, "sliceZ"), slice_norm)
        glBindVertexArray(self.fs_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    def _surface_to_texture(self, surf: pygame.Surface) -> int:
        data = pygame.image.tostring(surf, 'RGBA', True)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            surf.get_width(), surf.get_height(), 0,
            GL_RGBA, GL_UNSIGNED_BYTE, data,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex

    def render_hud(
        self,
        fps: float,
        step_count: int,
        paused: bool,
        view_mode: str,
        slice_info: str = '',
    ) -> None:
        lines = [
            f"FPS: {fps:.1f} | Steps: {step_count} {'PAUSED' if paused else ''}",
            f"View: {view_mode}" + (f' | Z={slice_info}' if slice_info else ''),
            "O:obstacle E:emitter R:reset C:clear V:toggle W/S:slice",
            "Click+drag:rotate  Scroll:zoom",
        ]
        y_off = 10

        glUseProgram(self.hud_shader)
        glUniform1i(glGetUniformLocation(self.hud_shader, "hudTex"), 0)
        glBindVertexArray(self.hud_vao)

        for text in lines:
            if not text:
                y_off += 28 + 4
                continue

            color = (255, 200, 100) if text.startswith('View') else (255, 255, 255)

            if text in self._hud_texture_cache:
                tex_id, tw, th = self._hud_texture_cache[text]
            else:
                surf = self.font.render(text, True, color, (0, 0, 0, 0))
                tw, th = surf.get_width(), surf.get_height()
                tex_id = self._surface_to_texture(surf)
                if not text.startswith('FPS') and not text.startswith('View'):
                    self._hud_texture_cache[text] = (tex_id, tw, th)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex_id)

            nx = -1.0 + 2.0 * 10 / self.win_w
            ny = 1.0 - 2.0 * (y_off + th) / self.win_h
            nw = 2.0 * tw / self.win_w
            nh = 2.0 * th / self.win_h

            verts = np.array([
                nx, ny,   0, 1,
                nx + nw, ny,   1, 1,
                nx + nw, ny + nh, 1, 0,
                nx, ny,   0, 1,
                nx + nw, ny + nh, 1, 0,
                nx, ny + nh, 0, 0,
            ], dtype=np.float32)

            glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
            glDrawArrays(GL_TRIANGLES, 0, 6)

            if text.startswith('FPS') or text.startswith('View'):
                glDeleteTextures(1, [tex_id])

            y_off += th + 4

        glBindVertexArray(0)
        glUseProgram(0)

    def handle_events(self, sim: CPULBM3D) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    sim.initialize(rho=1.0, u=0.1, v=0.0, w=0.0)
                    sim.clear_obstacles()
                elif event.key == pygame.K_e:
                    self.emitter_mode = True
                elif event.key == pygame.K_o:
                    self.emitter_mode = False
                elif event.key == pygame.K_c:
                    sim.clear_emitters()
                elif event.key == pygame.K_v:
                    self.view_mode = 'slice' if self.view_mode == 'volume' else 'volume'
                elif event.key == pygame.K_w:
                    self.slice_z = min(self.slice_z + 1, self.vol_d - 1)
                elif event.key == pygame.K_s:
                    self.slice_z = max(self.slice_z - 1, 0)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = pygame.mouse.get_pos()
                    vw, vh = self.win_w, self.win_h
                    gx = int(x / vw * self.vol_w)
                    gy = int(y / vh * self.vol_h)
                    gz = self.slice_z if self.view_mode == 'slice' else self.vol_d // 2
                    if self.emitter_mode:
                        sim.add_emitter(gx, gy, gz, strength=0.1)
                    else:
                        self.drawing_obstacle = True
                        sim.add_obstacle_sphere(gx, gy, gz, radius=3)
                elif event.button == 4:
                    self.radius = max(self.radius * 0.9, 1.0)
                    self._update_camera()
                elif event.button == 5:
                    self.radius = min(self.radius * 1.1, 500.0)
                    self._update_camera()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing_obstacle = False

            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    dx, dy = event.rel
                    if self.drawing_obstacle and not self.emitter_mode:
                        vw, vh = self.win_w, self.win_h
                        x, y = event.pos
                        gx = int(x / vw * self.vol_w)
                        gy = int(y / vh * self.vol_h)
                        gz = (
                            self.slice_z if self.view_mode == 'slice'
                            else self.vol_d // 2
                        )
                        sim.add_obstacle_sphere(gx, gy, gz, radius=3)
                    else:
                        self.theta += dx * 0.005
                        self.phi = np.clip(
                            self.phi + dy * 0.005, 0.05, np.pi - 0.05
                        )
                        self._update_camera()

        return True

    def update(
        self,
        smoke: np.ndarray,
        fps: float,
        step_count: int,
    ) -> None:
        glClearColor(*self.bg_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.upload_smoke(smoke)

        if self.view_mode == 'volume':
            self.render_volume()
        else:
            self.render_slice()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        slice_info = str(self.slice_z) if self.view_mode == 'slice' else ''
        self.render_hud(fps, step_count, self.paused, self.view_mode, slice_info)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
