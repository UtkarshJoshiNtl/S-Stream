from __future__ import annotations

from dataclasses import dataclass, field

from imgui_bundle import imgui

from engines.base import SimEngine


@dataclass
class UIState:
    """Mutable state shared between App and Widgets."""

    paused: bool = False
    emitter_mode: bool = False
    view_mode: str = "volume"  # 'volume' or 'slice' (3D only)
    slice_z: int = 0
    show_controls: bool = True
    fps: float = 0.0
    step_count: int = 0
    prev_mouse_x: float = 0.0
    prev_mouse_y: float = 0.0
    mouse_events: list = field(default_factory=list)


class Widgets:
    """ImGui control panels — reads from and writes to UIState."""

    def __init__(self, sim: SimEngine) -> None:
        self.sim = sim

    def draw(self, state: UIState) -> None:
        self._draw_control_panel(state)
        if state.show_controls:
            self._draw_help()

    def _draw_control_panel(self, state: UIState) -> None:
        imgui.begin("Simulation Control")

        imgui.text(f"FPS: {state.fps:.1f}")
        imgui.text(f"Steps: {state.step_count}")
        imgui.text(f"Emitters: {self.sim.get_emitter_count()}")
        imgui.separator()

        if state.paused:
            if imgui.button("Resume"):
                state.paused = False
        else:
            if imgui.button("Pause"):
                state.paused = True

        imgui.same_line()

        if imgui.button("Reset"):
            kw = dict(rho=1.0, u=self.sim.u_inflow, v=0.0)
            if self.sim.ndim == 3:
                kw["w"] = 0.0
            self.sim.initialize(**kw)

        imgui.separator()

        changed, val = imgui.slider_float(
            "Viscosity", self.sim.viscosity, 0.001, 0.5, "%.3f"
        )
        if changed:
            self.sim.viscosity = val

        changed, val = imgui.slider_float("Inflow", self.sim.u_inflow, 0.0, 0.5, "%.2f")
        if changed:
            self.sim.u_inflow = val

        imgui.separator()

        changed, val = imgui.slider_float(
            "Smoke Diffusion", self.sim.smoke_diffusion, 0.0, 0.25, "%.3f"
        )
        if changed:
            self.sim.smoke_diffusion = val

        changed, val = imgui.slider_float(
            "Smoke Decay", self.sim.smoke_decay, 0.9, 1.0, "%.4f"
        )
        if changed:
            self.sim.smoke_decay = val

        imgui.separator()

        if imgui.button("Clear Obstacles"):
            self.sim.clear_obstacles()

        imgui.same_line()

        if imgui.button("Clear Emitters"):
            self.sim.clear_emitters()

        if self.sim.ndim == 3:
            imgui.separator()
            changed, idx = imgui.combo(
                "View",
                ["volume", "slice"].index(state.view_mode),
                ["Volume", "Slice"],
            )
            if changed:
                state.view_mode = ["volume", "slice"][idx]
            if state.view_mode == "slice":
                changed, val = imgui.slider_int(
                    "Slice Z", state.slice_z, 0, self.sim.grid_shape[0] - 1
                )
                if changed:
                    state.slice_z = val

        imgui.separator()
        changed, val = imgui.checkbox("Emitter Mode", state.emitter_mode)
        if changed:
            state.emitter_mode = val
        if state.emitter_mode:
            imgui.same_line()
            imgui.text_colored((0.4, 0.8, 1.0, 1.0), "Place emitters")

        imgui.end()

    def _draw_help(self) -> None:
        imgui.begin("Controls")
        imgui.text("Left drag: orbit (3D)")
        imgui.text("Left drag (obstacle): draw obstacles")
        imgui.text("Click (emitter): place emitter")
        imgui.text("Scroll: zoom (3D)")
        imgui.text("Space: toggle pause")
        imgui.end()
