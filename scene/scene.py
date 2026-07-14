from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engines.base import SimEngine

# --- Obstacle specs ---


@dataclass
class ObstacleSpec:
    name: str
    bc_type: str = "bounce_back"

    def apply(self, sim: SimEngine) -> None:
        raise NotImplementedError


@dataclass
class CircleObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    radius: int = 5

    def apply(self, sim: SimEngine) -> None:
        obs = sim.get_obstacles_mut()
        y_grid, x_grid = np.ogrid[: sim.grid_shape[0], : sim.grid_shape[1]]
        mask = (x_grid - self.x) ** 2 + (y_grid - self.y) ** 2 <= self.radius**2
        obs[mask] = True


@dataclass
class RectObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    w: int = 10
    h: int = 10

    def apply(self, sim: SimEngine) -> None:
        obs = sim.get_obstacles_mut()
        x1, y1 = max(0, self.x), max(0, self.y)
        x2 = min(sim.grid_shape[1], self.x + self.w)
        y2 = min(sim.grid_shape[0], self.y + self.h)
        obs[y1:y2, x1:x2] = True


@dataclass
class PolygonObstacle(ObstacleSpec):
    points: list[tuple[int, int]] = field(default_factory=list)

    def apply(self, sim: SimEngine) -> None:
        obs = sim.get_obstacles_mut()
        h, w = obs.shape
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        xmin, xmax = max(0, min(xs)), min(w, max(xs) + 1)
        ymin, ymax = max(0, min(ys)), min(h, max(ys) + 1)
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                if _point_in_poly(x, y, self.points):
                    obs[y, x] = True


@dataclass
class EllipseObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    rx: int = 8
    ry: int = 5
    rotation: float = 0.0

    def apply(self, sim: SimEngine) -> None:
        obs = sim.get_obstacles_mut()
        h, w = obs.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        dx = x_grid - self.x
        dy = y_grid - self.y
        theta = np.radians(self.rotation)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rx_t = dx * cos_t + dy * sin_t
        ry_t = -dx * sin_t + dy * cos_t
        mask = (rx_t / max(self.rx, 0.1)) ** 2 + (ry_t / max(self.ry, 0.1)) ** 2 <= 1.0
        obs[mask] = True


@dataclass
class STLObstacle(ObstacleSpec):
    path: str = ""
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    filled: bool = True

    def apply(self, sim: SimEngine) -> None:
        if not self.path:
            return
        try:
            import trimesh
        except ImportError:
            return
        mesh = trimesh.load(self.path, force="mesh")
        if sim.ndim == 2:
            _rasterize_stl_2d(mesh, sim, self)
        else:
            _rasterize_stl_3d(mesh, sim, self)


def _rasterize_stl_2d(
    mesh: trimesh.Trimesh,  # noqa: F821
    sim: SimEngine,
    spec: STLObstacle,
) -> None:
    obs = sim.get_obstacles_mut()
    h, w = obs.shape
    verts = np.array(mesh.vertices)
    verts[:, 0] = verts[:, 0] * spec.scale + spec.offset_x
    verts[:, 1] = verts[:, 1] * spec.scale + spec.offset_y
    tris = np.array(mesh.faces)
    for tri in tris:
        pts = verts[tri]
        xs = pts[:, 0]
        ys = pts[:, 1]
        min_x = max(0, int(np.floor(xs.min())))
        max_x = min(w - 1, int(np.ceil(xs.max())))
        min_y = max(0, int(np.floor(ys.min())))
        max_y = min(h - 1, int(np.ceil(ys.max())))
        if min_x > max_x or min_y > max_y:
            continue
        for sy in range(min_y, max_y + 1):
            for sx in range(min_x, max_x + 1):
                if _point_in_triangle(sx, sy, pts[0, :2], pts[1, :2], pts[2, :2]):
                    obs[sy, sx] = True
        if spec.filled:
            row_mask = obs[sy, min_x : max_x + 1]
            filled_xs = np.where(row_mask)[0]
            if len(filled_xs) >= 2:
                obs[sy, min_x + filled_xs[0] : min_x + filled_xs[-1] + 1] = True


def _rasterize_stl_3d(
    mesh: trimesh.Trimesh,  # noqa: F821
    sim: SimEngine,
    spec: STLObstacle,
) -> None:
    obs = sim.get_obstacles_mut()
    d, h, w = obs.shape
    verts = np.array(mesh.vertices)
    verts[:, 0] = verts[:, 0] * spec.scale + spec.offset_x
    verts[:, 1] = verts[:, 1] * spec.scale + spec.offset_y
    tris = np.array(mesh.faces)
    for tri in tris:
        pts = verts[tri]
        for axis in range(3):
            axis_range = [0, 1, 2]
            axis_range.remove(axis)
            v0 = pts[0]
            v1 = pts[1]
            v2 = pts[2]
            coords = [v0, v1, v2]
            ax_min = int(max(0, min(c[axis] for c in coords)))
            ax_max = int(
                min(
                    [obs.shape[axis]][0] - 1,
                    max(c[axis] for c in coords),
                )
            )
            if ax_min > ax_max:
                continue
            for idx in range(ax_min, ax_max + 1):
                edges = []
                for i in range(3):
                    j = (i + 1) % 3
                    if (coords[i][axis] <= idx <= coords[j][axis]) or (
                        coords[j][axis] <= idx <= coords[i][axis]
                    ):
                        if coords[i][axis] == coords[j][axis]:
                            continue
                        t = (idx - coords[i][axis]) / (
                            coords[j][axis] - coords[i][axis]
                        )
                        other_coords = [
                            coords[i][k] + t * (coords[j][k] - coords[i][k])
                            for k in axis_range
                        ]
                        edges.append(other_coords)
                if len(edges) >= 2:
                    e0, e1 = edges[0], edges[1]
                    min_0 = max(0, int(min(e0[0], e1[0])))
                    max_0 = int(min(obs.shape[axis_range[0]] - 1, max(e0[0], e1[0])))
                    min_1 = max(0, int(min(e0[1], e1[1])))
                    max_1 = int(min(obs.shape[axis_range[1]] - 1, max(e0[1], e1[1])))
                    for a in range(min_0, max_0 + 1):
                        for b in range(min_1, max_1 + 1):
                            coords_3d = [0, 0, 0]
                            coords_3d[axis] = idx
                            coords_3d[axis_range[0]] = a
                            coords_3d[axis_range[1]] = b
                            obs[coords_3d[0], coords_3d[1], coords_3d[2]] = True


@dataclass
class ImageObstacle(ObstacleSpec):
    path: str = ""
    threshold: int = 128
    invert: bool = False
    scale_x: float = 1.0
    scale_y: float = 1.0

    def apply(self, sim: SimEngine) -> None:
        if not self.path:
            return
        try:
            from PIL import Image
        except ImportError:
            return
        img = Image.open(self.path).convert("L")
        if self.scale_x != 1.0 or self.scale_y != 1.0:
            new_w = max(1, int(img.width * self.scale_x))
            new_h = max(1, int(img.height * self.scale_y))
            img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
        arr = np.array(img)
        mask = arr > self.threshold
        if self.invert:
            mask = ~mask
        h, w = sim.grid_shape[:2]
        ih, iw = mask.shape
        y_off = min(0, (h - ih) // 2)
        x_off = min(0, (w - iw) // 2)
        obs = sim.get_obstacles_mut()
        y_src = max(0, -y_off)
        x_src = max(0, -x_off)
        y_dst = max(0, y_off)
        x_dst = max(0, x_off)
        copy_h = min(ih - y_src, h - y_dst)
        copy_w = min(iw - x_src, w - x_dst)
        if copy_h > 0 and copy_w > 0:
            obs[y_dst : y_dst + copy_h, x_dst : x_dst + copy_w] = mask[
                y_src : y_src + copy_h, x_src : x_src + copy_w
            ]


@dataclass
class AirfoilObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    chord: int = 30
    angle_of_attack: float = 0.0
    naca_code: str = "0012"

    def apply(self, sim: SimEngine) -> None:
        obs = sim.get_obstacles_mut()
        h, w = obs.shape
        naca = self.naca_code.zfill(4)
        m = int(naca[0]) / 100.0
        p = int(naca[1]) / 10.0
        t_max = int(naca[2:]) / 100.0
        c = max(self.chord, 3)
        n_pts = max(c * 4, 40)
        x_norm = np.linspace(0, 1, n_pts)
        yt = (
            5
            * t_max
            * (
                0.2969 * np.sqrt(x_norm)
                - 0.1260 * x_norm
                - 0.3516 * x_norm**2
                + 0.2843 * x_norm**3
                - 0.1015 * x_norm**4
            )
        )
        yc = np.zeros_like(x_norm)
        if p > 0 and m > 0:
            mask_lower = x_norm <= p
            mask_upper = x_norm > p
            yc[mask_lower] = (m / p**2) * (
                2 * p * x_norm[mask_lower] - x_norm[mask_lower] ** 2
            )
            yc[mask_upper] = (
                m
                / (1 - p) ** 2
                * ((1 - 2 * p) + 2 * p * x_norm[mask_upper] - x_norm[mask_upper] ** 2)
            )
        xu = x_norm * c
        yu = (yc + yt) * c
        xl = x_norm * c
        yl = (yc - yt) * c
        x_airfoil = np.concatenate([xu[::-1], xl[1:]])
        y_airfoil = np.concatenate([yu[::-1], yl[1:]])
        theta = np.radians(self.angle_of_attack)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = x_airfoil * cos_t - y_airfoil * sin_t
        y_rot = x_airfoil * sin_t + y_airfoil * cos_t
        x_grid = (x_rot + self.x).astype(int)
        y_grid = (y_rot + self.y).astype(int)
        n = len(x_grid)
        for i in range(n):
            j = (i + 1) % n
            x0, y0 = x_grid[i], y_grid[i]
            x1, y1 = x_grid[j], y_grid[j]
            dx = x1 - x0
            dy = y1 - y0
            steps = max(abs(dx), abs(dy), 1)
            for s in range(steps + 1):
                t = s / steps
                px = int(x0 + dx * t)
                py = int(y0 + dy * t)
                if 0 <= px < w and 0 <= py < h:
                    obs[py, px] = True
        x_min_c, x_max_c = int(x_grid.min()), int(x_grid.max())
        y_min_c, y_max_c = int(y_grid.min()), int(y_grid.max())
        for gy in range(max(0, y_min_c), min(h, y_max_c + 1)):
            inside = False
            j_idx = n - 1
            for i_idx in range(n):
                xi, yi = int(x_grid[i_idx]), int(y_grid[i_idx])
                _, yj = int(x_grid[j_idx]), int(y_grid[j_idx])
                if ((yi > gy) != (yj > gy)) and (
                    gy < (yj - yi) * (gy - yi) / max(yj - yi, 1e-10) + xi
                ):
                    inside = not inside
                j_idx = i_idx
            if inside:
                for gx in range(max(0, x_min_c), min(w, x_max_c + 1)):
                    if not obs[gy, gx]:
                        pts = list(
                            zip(
                                x_grid.tolist(),
                                y_grid.tolist(),
                                strict=False,
                            )
                        )
                        if _point_in_poly(gx, gy, pts):
                            obs[gy, gx] = True


@dataclass
class ChannelObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    w: int = 60
    h: int = 40
    inlet_ratio: float = 1.0
    outlet_ratio: float = 1.0

    def apply(self, sim: SimEngine) -> None:
        obs = sim.get_obstacles_mut()
        h_grid, w_grid = obs.shape
        x1 = max(0, self.x)
        x2 = min(w_grid, self.x + self.w)
        y1 = max(0, self.y)
        y2 = min(h_grid, self.y + self.h)
        in_h = max(1, int(self.h * self.inlet_ratio))
        out_h = max(1, int(self.h * self.outlet_ratio))
        for gx in range(x1, x2):
            frac = (gx - self.x) / max(self.w - 1, 1)
            wall_h = int(in_h + (out_h - in_h) * frac)
            wall_h = max(1, min(wall_h, self.h))
            wall_top = self.y + (self.h - wall_h) // 2
            wall_bot = wall_top + wall_h
            for gy in range(y1, y2):
                if gy < wall_top or gy >= wall_bot:
                    obs[gy, gx] = True


@dataclass
class LatticeObstacle(ObstacleSpec):
    x: int = 0
    y: int = 0
    w: int = 40
    h: int = 40
    cell_size: int = 8
    wall_thickness: int = 1

    def apply(self, sim: SimEngine) -> None:
        obs = sim.get_obstacles_mut()
        h_grid, w_grid = obs.shape
        cs = max(self.cell_size, 2)
        wt = max(self.wall_thickness, 1)
        x1, y1 = max(0, self.x), max(0, self.y)
        x2 = min(w_grid, self.x + self.w)
        y2 = min(h_grid, self.y + self.h)
        for gy in range(y1, y2):
            for gx in range(x1, x2):
                lx = gx - self.x
                ly = gy - self.y
                if lx % cs < wt or ly % cs < wt:
                    obs[gy, gx] = True


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


def _point_in_triangle(
    px: float,
    py: float,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> bool:
    d1 = _sign(px, py, v0[0], v0[1], v1[0], v1[1])
    d2 = _sign(px, py, v1[0], v1[1], v2[0], v2[1])
    d3 = _sign(px, py, v2[0], v2[1], v0[0], v0[1])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _sign(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)


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
