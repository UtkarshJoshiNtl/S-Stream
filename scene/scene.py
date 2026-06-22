from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engines.base import SimEngine


# --- Obstacle specs ---

@dataclass
class ObstacleSpec:
    name: str

    def apply(self, sim: SimEngine) -> None:
        raise NotImplementedError


@dataclass
class CircleObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    radius: int = 5

    def apply(self, sim: SimEngine) -> None:
        obs = sim.obstacles  # type: ignore[attr-defined]
        y_grid, x_grid = np.ogrid[: sim.grid_shape[0], : sim.grid_shape[1]]
        mask = (x_grid - self.x) ** 2 + (y_grid - self.y) ** 2 <= self.radius ** 2
        obs[mask] = True


@dataclass
class RectObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    w: int = 10
    h: int = 10

    def apply(self, sim: SimEngine) -> None:
        obs = sim.obstacles  # type: ignore[attr-defined]
        x1, y1 = max(0, self.x), max(0, self.y)
        x2 = min(sim.grid_shape[1], self.x + self.w)
        y2 = min(sim.grid_shape[0], self.y + self.h)
        obs[y1:y2, x1:x2] = True


@dataclass
class PolygonObstacle(ObstacleSpec):
    points: list[tuple[int, int]] = field(default_factory=list)

    def apply(self, sim: SimEngine) -> None:
        obs = sim.obstacles  # type: ignore[attr-defined]
        h, w = obs.shape
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        xmin, xmax = max(0, min(xs)), min(w, max(xs) + 1)
        ymin, ymax = max(0, min(ys)), min(h, max(ys) + 1)
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                if _point_in_poly(x, y, self.points):
                    obs[y, x] = True


def _point_in_poly(px: int, py: int, poly: list[tuple[int, int]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# --- Emitter spec ---

@dataclass
class EmitterSpec:
    name: str
    x: int = 0
    y: int = 0
    strength: float = 0.05


# --- Probe spec ---

@dataclass
class ProbeSpec:
    name: str
    x: int = 0
    y: int = 0
    fields: list[str] = field(default_factory=lambda: ["u", "v", "speed", "pressure"])


# --- Scene ---

@dataclass
class SceneProductMeta:
    recommended_colormap: str = "smoke"
    autorun_steps: int = 0
    lesson_headline: str = ""
    expected_ranges: dict[str, list[float]] = field(default_factory=dict)
    flow_regime_labels: list[str] = field(default_factory=list)
    export_caption: str = ""
    classroom_prompts: list[str] = field(default_factory=list)
    recommended_sweep: dict = field(default_factory=dict)
    recipe: str = ""


@dataclass
class Scene:
    name: str = "Untitled"
    width: int = 128
    height: int = 128
    viscosity: float = 0.02
    u_inflow: float = 0.15
    smoke_diffusion: float = 0.05
    smoke_decay: float = 0.999
    description: str = ""
    obstacles: list[ObstacleSpec] = field(default_factory=list)
    emitters: list[EmitterSpec] = field(default_factory=list)
    probes: list[ProbeSpec] = field(default_factory=list)
    sweeps: list[dict] = field(default_factory=list)
    product: SceneProductMeta = field(default_factory=SceneProductMeta)


def apply_to_sim(scene: Scene, sim: SimEngine) -> None:
    sim.initialize(rho=1.0, u=scene.u_inflow, v=0.0)
    sim.viscosity = scene.viscosity
    sim.u_inflow = scene.u_inflow
    sim.smoke_diffusion = scene.smoke_diffusion
    sim.smoke_decay = scene.smoke_decay
    sim.clear_obstacles()
    sim.clear_emitters()
    for obs in scene.obstacles:
        obs.apply(sim)
    for emit in scene.emitters:
        sim.add_emitter(emit.x, emit.y, emit.strength)


def default_scene() -> Scene:
    return Scene(
        description="",
        obstacles=[],
        emitters=[EmitterSpec(name="Inlet", x=2, y=64, strength=0.05)],
        probes=[],
    )
