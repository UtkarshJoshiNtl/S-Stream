from __future__ import annotations

import ctypes

import glfw
from imgui_bundle import hello_imgui, imgui

from engines.base import SimEngine
from ui.renderer import Renderer
from ui.widgets import UIState, Widgets


class CuFlodaApp:
    """Application using hello_imgui runner. Engine-agnostic."""

    def __init__(self, sim: SimEngine) -> None:
        self.sim = sim

        scale = 5 if sim.ndim == 2 else 4
        if sim.ndim == 2:
            win_w, win_h = sim.grid_shape[1] * scale, sim.grid_shape[0] * scale
        else:
            win_w, win_h = sim.grid_shape[2] * scale, sim.grid_shape[1] * scale

        self.state = UIState()
        self.state.view_mode = "volume" if sim.ndim == 3 else "2d"
        if sim.ndim == 3:
            self.state.slice_z = sim.grid_shape[0] // 2

        self.widgets = Widgets(sim)
        self.renderer: Renderer | None = None
        self._renderer_ready = False
        self._fb_w = 0
        self._fb_h = 0

        self._steps_per_frame = 5 if sim.ndim == 2 else 3
        self._last_fps_time = 0.0
        self._fps_frames = 0

        self._runner_params = self._make_params(win_w, win_h)
        self._setup_callbacks()

    def _make_params(self, w: int, h: int) -> hello_imgui.RunnerParams:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "CuFloda - Fluid Simulation"
        params.app_window_params.window_geometry.size = (w, h)
        params.app_window_params.window_geometry.size_auto = False
        params.app_window_params.restore_previous_geometry = False

        params.renderer_backend_type = hello_imgui.RendererBackendType.open_gl3
        backend_opts = hello_imgui.RendererBackendOptions()
        backend_opts.open_gl_options.major_version = 3
        backend_opts.open_gl_options.minor_version = 3
        backend_opts.open_gl_options.use_core_profile = True
        backend_opts.open_gl_options.glsl_version = "#version 330"
        params.renderer_backend_options = backend_opts

        params.fps_idling.is_idling = False
        return params

    def _setup_callbacks(self) -> None:
        callbacks = hello_imgui.RunnerCallbacks()
        callbacks.post_init = self._post_init
        callbacks.pre_new_frame = self._pre_new_frame
        callbacks.custom_background = self._custom_background
        callbacks.show_gui = self._show_gui
        callbacks.any_backend_event_callback = self._handle_event
        callbacks.before_exit = self._before_exit
        self._runner_params.callbacks = callbacks

    def _post_init(self) -> None:
        win_addr = hello_imgui.get_glfw_window_address()
        win_ptr = ctypes.cast(win_addr, ctypes.POINTER(glfw._GLFWwindow))
        self._fb_w, self._fb_h = glfw.get_framebuffer_size(win_ptr)

    def _pre_new_frame(self) -> None:
        if not self._renderer_ready:
            win_addr = hello_imgui.get_glfw_window_address()
            win_ptr = ctypes.cast(win_addr, ctypes.POINTER(glfw._GLFWwindow))
            glfw.make_context_current(win_ptr)
            self.renderer = Renderer(self.sim, self._fb_w, self._fb_h)
            self._renderer_ready = True

        if not self.state.paused:
            self.sim.run(self._steps_per_frame)
            self.state.step_count += self._steps_per_frame

        self._fps_frames += 1
        now = hello_imgui.frame_rate()
        if now - self._last_fps_time >= 0.5:
            self.state.fps = self._fps_frames / (now - self._last_fps_time)
            self._last_fps_time = now
            self._fps_frames = 0

    def _custom_background(self) -> None:
        if self.renderer is None:
            return
        self._process_mouse()
        self.renderer.render()

    def _show_gui(self) -> None:
        self.widgets.draw(self.state)
        self._sync_view_mode()

    def _sync_view_mode(self) -> None:
        if self.renderer is not None and self.sim.ndim == 3:
            self.renderer.view_mode = self.state.view_mode
            self.renderer.slice_z = self.state.slice_z

    def _handle_event(self) -> None:
        pass

    def _process_mouse(self) -> None:
        if imgui.is_any_item_hovered():
            return

        win_addr = hello_imgui.get_glfw_window_address()
        win_ptr = ctypes.cast(win_addr, ctypes.POINTER(glfw._GLFWwindow))
        left = glfw.get_mouse_button(win_ptr, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        x, y = glfw.get_cursor_pos(win_ptr)

        if self.state.emitter_mode and left:
            self._place_emitter(x, y)
        elif not self.state.emitter_mode and left:
            if self.sim.ndim == 3:
                dx = x - self.state.prev_mouse_x
                dy = y - self.state.prev_mouse_y
                self.renderer.orbit(dx, dy)

        self.state.prev_mouse_x = x
        self.state.prev_mouse_y = y

    def _place_emitter(self, x: float, y: float) -> None:
        if self.sim.ndim == 3:
            d, h, w = self.sim.grid_shape
        else:
            h, w = self.sim.grid_shape
        win_addr = hello_imgui.get_glfw_window_address()
        win_ptr = ctypes.cast(win_addr, ctypes.POINTER(glfw._GLFWwindow))
        win_w, win_h = glfw.get_window_size(win_ptr)
        gx = min(int(x / max(win_w, 1) * w), w - 1)
        gy = min(int(y / max(win_h, 1) * h), h - 1)
        if self.sim.ndim == 3:
            gz = self.state.slice_z if self.state.view_mode == "slice" else d // 2
            self.sim.add_emitter(gx, gy, gz, strength=0.1)
        else:
            self.sim.add_emitter(gx, gy, strength=0.05)

    def _before_exit(self) -> None:
        pass

    def run(self) -> None:
        hello_imgui.run(self._runner_params)
