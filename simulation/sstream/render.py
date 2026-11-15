"""Layered renderer for solver snapshots: streamlines, smoke, vortices, model.

``sstream-render SNAPSHOT_DIR --out OUT`` reads the directory written by
``sstream --snapshots DIR`` (``meta.json`` + ``solid.npy`` +
``velocity_step_NNNNNNN.npz``), rebuilds the placed model deterministically
from the stored placement spec, and renders the enabled layers (at least
one of ``--streamlines``/``--smoke``/``--vortices``/``--model``) with
PyVista, off-screen.

The solver never imports this module.  Time handling: snapshot steps map to
physical time through the lattice ``dt`` in ``meta.json``; every frame lands
at a tau inside the snapshot window and the velocity field is interpolated
between the surrounding pair of snapshots, so the layers move smoothly even
though the solver wrote a handful of files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import pyvista as pv

from .fields import q_criterion, speed
from .mesh import build_model, load_surface
from .schema import MeshSpec, MeshTransform, TunnelSpec

# ---------------------------------------------------------------------------
# Snapshot loading and trilinear sampling
# ---------------------------------------------------------------------------

DTYPE = np.dtype("<f2")


def load_meta(directory: Path) -> dict:
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"{directory} does not contain a meta.json (run sstream --snapshots first)")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_snapshot_files(directory: Path) -> list[tuple[int, Path]]:
    files = []
    for path in directory.glob("velocity_step_*.npz"):
        try:
            step = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        files.append((step, path))
    if not files:
        raise ValueError(f"no velocity_step_*.npz snapshots in {directory}")
    files.sort(key=lambda item: item[0])
    return files


def load_velocity(path: Path, factor: float = 1.0) -> np.ndarray:
    """Load a stored field; ``factor`` converts lattice units to m/s (dx/dt)."""
    with np.load(path) as npz:
        field = np.asarray(npz["velocity"], dtype=np.float32)
    return field * np.float32(factor)


class Timeline:
    """Physical times of the snapshot frames, from the lattice dt."""

    def __init__(self, steps: list[int], dt: float) -> None:
        self.times = [step * dt for step in steps]

    @property
    def span(self) -> float:
        return self.times[-1] - self.times[0]

    def bracket(self, tau: float) -> tuple[int, int, float]:
        """Indices of the bracketing snapshots and the interpolation weight."""
        times = self.times
        if tau <= times[0]:
            return 0, 0, 0.0
        if tau >= times[-1]:
            return len(times) - 1, len(times) - 1, 0.0
        for i in range(len(times) - 1):
            if times[i] <= tau <= times[i + 1]:
                span = times[i + 1] - times[i]
                return i, i + 1, (tau - times[i]) / span if span > 0 else 0.0
        return len(times) - 1, len(times) - 1, 0.0


class FieldCache:
    """Holds at most two velocity fields; lerps the pair for a frame.

    ``factor`` is dx/dt of the lattice, converting the stored lattice
    velocities to m/s so advection and colour scales work in SI units.
    """

    def __init__(self, files: list[tuple[int, Path]], factor: float = 1.0) -> None:
        self.files = files
        self.factor = factor
        self.cache: dict[int, np.ndarray] = {}

    def field(self, i: int) -> np.ndarray:
        if i not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = load_velocity(self.files[i][1], self.factor)
        return self.cache[i]

    def sampled(self, i0: int, i1: int, alpha: float) -> np.ndarray:
        a = self.field(i0)
        if i1 == i0 or alpha <= 0.0:
            return a
        return (1.0 - alpha) * a + alpha * self.field(i1)


def origin_of(meta: dict) -> tuple[float, float, float]:
    """Centre of cell (0,0,0) from the stored bounds and spacing."""
    bounds = meta["tunnel"]["bounds"]
    spacing = meta["tunnel"]["spacing"]
    return (
        bounds[0] + spacing[0] * 0.5,
        bounds[2] + spacing[1] * 0.5,
        bounds[4] + spacing[2] * 0.5,
    )


class GridSampler:
    """Trilinear interpolation of a vector field at physical points."""

    def __init__(self, field: np.ndarray, spacing, origin) -> None:
        self.field = np.ascontiguousarray(field, dtype=np.float32)
        self.spacing = np.asarray(spacing, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.shape = self.field.shape[:3]

    def __call__(self, points: np.ndarray) -> np.ndarray:
        positions = (np.asarray(points, dtype=np.float64) - self.origin) / self.spacing
        positions = np.clip(positions, 0.0, np.asarray(self.shape, dtype=np.float64) - 1.0)
        i0 = np.floor(positions).astype(np.int64)
        i1 = np.minimum(i0 + 1, np.asarray(self.shape, dtype=np.int64) - 1)
        frac = positions - i0
        tx = frac[:, 0].reshape(-1, 1).astype(np.float32)
        ty = frac[:, 1].reshape(-1, 1).astype(np.float32)
        tz = frac[:, 2].reshape(-1, 1).astype(np.float32)

        def corner(cx, cy, cz):
            ix = np.where(cx == 0, i0[:, 0], i1[:, 0])
            iy = np.where(cy == 0, i0[:, 1], i1[:, 1])
            iz = np.where(cz == 0, i0[:, 2], i1[:, 2])
            return self.field[ix, iy, iz]

        c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
        c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
        c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
        c011, c111 = corner(0, 1, 1), corner(1, 1, 1)
        c00 = c000 + (c100 - c000) * tx
        c10 = c010 + (c110 - c010) * tx
        c01 = c001 + (c101 - c001) * tx
        c11 = c011 + (c111 - c011) * tx
        c0 = c00 + (c10 - c00) * ty
        c1 = c01 + (c11 - c01) * ty
        return c0 + (c1 - c0) * tz


class PointSampler:
    """Nearest-cell solid lookup against the mask."""

    def __init__(self, solid: np.ndarray, spacing, origin) -> None:
        self.solid = np.ascontiguousarray(solid, dtype=np.uint8)
        self.spacing = np.asarray(spacing, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.shape = np.asarray(self.solid.shape, dtype=np.float64)

    def inside(self, points: np.ndarray) -> np.ndarray:
        idx = np.floor((np.asarray(points, dtype=np.float64) - self.origin) / self.spacing)
        idx = idx.astype(np.int64)
        idx = np.clip(idx, 0, self.shape.astype(np.int64) - 1)
        return self.solid[idx[:, 0], idx[:, 1], idx[:, 2]] != 0


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

#: Interpolated frames per gap between consecutive snapshots in a movie.
FRAMES_PER_SNAPSHOT_GAP = 12


def frame_times(timeline: Timeline, preview: bool = False) -> list[float]:
    """Frame times for the output.

    Preview renders one frame per every 2nd snapshot; the full movie renders
    FRAMES_PER_SNAPSHOT_GAP interpolated frames per snapshot gap.
    """
    if preview:
        return [timeline.times[i] for i in range(0, len(timeline.times), 2)]
    gaps = max(len(timeline.times) - 1, 1)
    n = FRAMES_PER_SNAPSHOT_GAP * gaps + 1
    return list(np.linspace(timeline.times[0], timeline.times[-1], n))


# ---------------------------------------------------------------------------
# Smoke layer
# ---------------------------------------------------------------------------

SMOKE_MAX_PARTICLES = 3500
SMOKE_TRAIL_LENGTH = 40
SMOKE_EMIT_PER_FRAME = 80
SMOKE_SUBSTEPS = 4


class SmokePlume:
    """Continuous emitter with RK2-advected particles and fading trails."""

    def __init__(self, meta: dict, solid: np.ndarray) -> None:
        self.bounds = meta["tunnel"]["bounds"]
        self.spacing = meta["tunnel"]["spacing"]
        self.origin = origin_of(meta)
        self.mask = PointSampler(solid, self.spacing, self.origin)
        self.positions: list[np.ndarray] = []
        self.trails: list[list[np.ndarray]] = []
        self.rng = np.random.default_rng(20240515)
        # Smoke source: just upstream of the model, sized to its silhouette.
        mb = np.asarray(meta["placement"].get("placed_bounds", [[0, 1], [0, 1], [0, 1]]), dtype=float)
        model_len = max(mb[0, 1] - mb[0, 0], 0.2)
        self.emit_x = max(self.bounds[0] + 0.05, mb[0, 0] - 0.5 * model_len)
        self.emit_y = float(mb[1].mean())
        self.emit_z = float(mb[2].mean())
        self.emit_ry = max(0.35 * (mb[1, 1] - mb[1, 0]), 0.05)
        self.emit_rz = max(0.35 * (mb[2, 1] - mb[2, 0]), 0.05)

    def emit(self) -> None:
        x_emit = self.emit_x
        z_mid = self.emit_z
        half_y = self.emit_ry
        half_z = self.emit_rz
        for _ in range(SMOKE_EMIT_PER_FRAME):
            if len(self.positions) >= SMOKE_MAX_PARTICLES:
                return
            r = math.sqrt(self.rng.random())
            theta = self.rng.random() * 2.0 * math.pi
            point = np.array(
                [x_emit, half_y * r * math.cos(theta), z_mid + half_z * r * math.sin(theta)],
                dtype=np.float64,
            )
            self.positions.append(point)
            self.trails.append([point.copy()])

    def advect(self, sampler: GridSampler, dt: float) -> None:
        if dt > 0.0 and self.positions:
            pos = np.asarray(self.positions, dtype=np.float64)
            h = dt / SMOKE_SUBSTEPS
            for _ in range(SMOKE_SUBSTEPS):
                k1 = sampler(pos)
                k2 = sampler(pos + 0.5 * h * k1)
                pos = pos + h * k2
            # Kill particles that left the domain or entered the solid.
            inside = self.mask.inside(pos)
            keep = (
                ~inside
                & (pos[:, 0] <= self.bounds[1])
                & (pos[:, 1] >= self.bounds[2])
                & (pos[:, 1] <= self.bounds[3])
                & (pos[:, 2] >= self.bounds[4])
                & (pos[:, 2] <= self.bounds[5])
            )
            new_positions, new_trails = [], []
            for i, ok in enumerate(keep):
                if not ok:
                    continue
                trail = self.trails[i]
                trail.append(pos[i].copy())
                if len(trail) > SMOKE_TRAIL_LENGTH:
                    del trail[0]
                new_positions.append(pos[i])
                new_trails.append(trail)
            self.positions, self.trails = new_positions, new_trails
        # Emit after the kill-rebuild so fresh particles are never dropped.
        self.emit()

    def polydata(self) -> pv.PolyData | None:
        """All live trails as one PolyData with fading RGBA point colors."""
        n_segments = sum(max(len(t) - 1, 0) for t in self.trails)
        if n_segments == 0:
            return None
        points = np.empty((n_segments * 2, 3), dtype=np.float32)
        rgba = np.empty((n_segments * 2, 4), dtype=np.float32)
        lines = np.empty((n_segments, 3), dtype=np.int64)
        cursor = 0
        grey = 0.82
        for trail in self.trails:
            for j in range(1, len(trail)):
                # alpha rises along the trail; oldest point fades to 0
                a_lo = (j - 1) / max(len(trail) - 1, 1)
                a_hi = j / max(len(trail) - 1, 1)
                points[cursor * 2] = trail[j - 1]
                points[cursor * 2 + 1] = trail[j]
                rgba[cursor * 2] = (grey, grey, grey, a_lo)
                rgba[cursor * 2 + 1] = (grey, grey, grey, a_hi)
                lines[cursor] = (2, cursor * 2, cursor * 2 + 1)
                cursor += 1
        lines_flat = lines.ravel()
        poly = pv.PolyData(points)
        poly.lines = lines_flat
        poly.point_data["rgba"] = rgba
        return poly


# ---------------------------------------------------------------------------
# Streamlines layer (last snapshot only)
# ---------------------------------------------------------------------------

STREAMLINE_STEP_FRACTION = 0.5   # pathline step = this * cell spacing
STREAMLINE_MAX_STEPS = 400


def streamline_seeds(meta: dict) -> np.ndarray:
    """2D (y,z) rake grid upstream of the placed model."""
    bounds = meta["tunnel"]["bounds"]
    placement = meta.get("placement", {})
    mb = placement.get("placed_bounds")
    if mb is None:
        model_xmin, model_len = 0.5 * (bounds[0] + bounds[1]), 1.2
        dy = dz = 0.5
        mb = np.array(
            [[model_xmin, model_xmin + model_len], [-0.5, 0.5], [0.0, 0.3]]
        )
    else:
        model_xmin = mb[0][0]
        model_len = mb[0][1] - mb[0][0]
        dy = mb[1][1] - mb[1][0]
        dz = mb[2][1] - mb[2][0]
    x_rake = max(bounds[0] + 0.05, model_xmin - 0.3 * model_len)
    spread_y = 1.5 * max(dy, 0.25)
    spread_z = 1.5 * max(dz, 0.25)
    y_lim = max(bounds[2] + 1e-3, -spread_y * 0.5), min(bounds[3] - 1e-3, spread_y * 0.5)
    z_mid = 0.5 * (mb[2][0] + mb[2][1])
    z_lim = (
        max(bounds[4] + 0.05, z_mid - spread_z * 0.5),
        min(bounds[5] - 0.05, z_mid + spread_z * 0.5),
    )
    ys = np.linspace(y_lim[0], y_lim[1], 12)
    zs = np.linspace(z_lim[0], z_lim[1], 9)
    grid_y, grid_z = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack(
        [np.full(grid_y.size, x_rake), grid_y.reshape(-1), grid_z.reshape(-1)]
    )


def integrate_pathlines(
    seeds: np.ndarray, sampler: GridSampler, solid_mask: PointSampler, meta: dict
) -> list[np.ndarray]:
    """RK4 integration of d(p) = u_hat(p), one polyline per seed."""
    bounds = meta["tunnel"]["bounds"]
    step = STREAMLINE_STEP_FRACTION * meta["tunnel"]["spacing"][0]
    lines = []
    for seed in seeds:
        points = [seed.astype(np.float64)]
        p = seed.astype(np.float64)
        for _ in range(STREAMLINE_MAX_STEPS):
            def direction(q):
                u = sampler(np.atleast_2d(q))[0].astype(np.float64)
                norm = np.linalg.norm(u)
                if norm < 1e-9:
                    return u, norm
                return u / norm, norm

            k1, n1 = direction(p)
            k2, _ = direction(p + 0.5 * step * k1)
            k3, _ = direction(p + 0.5 * step * k2)
            k4, _ = direction(p + step * k3)
            p = p + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if (
                p[0] > bounds[1]
                or p[1] < bounds[2]
                or p[1] > bounds[3]
                or p[2] < bounds[4]
                or p[2] > bounds[5]
                or n1 < 1e-9
            ):
                break
            if solid_mask.inside(np.atleast_2d(p))[0]:
                break
            points.append(p.copy())
        if len(points) >= 2:
            lines.append(np.asarray(points))
    return lines


def build_streamlines(
    meta: dict, field: np.ndarray, solid: np.ndarray, tube_radius: float
) -> pv.PolyData | None:
    """Speed-coloured tube mesh for one field (the LAST snapshot)."""
    spacing = meta["tunnel"]["spacing"]
    origin = origin_of(meta)
    sampler = GridSampler(field, spacing, origin)
    mask = PointSampler(solid, spacing, origin)
    seeds = streamline_seeds(meta)
    lines = integrate_pathlines(seeds, sampler, mask, meta)
    if not lines:
        return None
    points = np.vstack(lines)
    connectivity = []
    cursor = 0
    for line in lines:
        connectivity.extend([len(line), *range(cursor, cursor + len(line))])
        cursor += len(line)
    poly = pv.PolyData(points)
    poly.lines = np.asarray(connectivity, dtype=np.int64)
    tube = poly.tube(radius=tube_radius, n_sides=8)
    # Colour by speed sampled at the tube surface points.
    u = sampler(np.asarray(tube.points))
    tube.point_data["speed"] = np.linalg.norm(u, axis=1).astype(np.float32)
    return tube


# ---------------------------------------------------------------------------
# Vortices layer (Q isosurface)
# ---------------------------------------------------------------------------

Q_DOWNSAMPLE_CELLS = 4_000_000


def build_vortices(
    meta: dict, field: np.ndarray, q_fraction: float
) -> pv.PolyData | None:
    """Q-isosurface coloured by speed, optionally on a 2x-downsampled grid."""
    spacing = np.asarray(meta["tunnel"]["spacing"], dtype=np.float64)
    origin = np.asarray(origin_of(meta), dtype=np.float64)
    cells = int(np.prod(field.shape[:3]))
    if cells > Q_DOWNSAMPLE_CELLS:
        field = field[::2, ::2, ::2].copy()
        spacing = spacing * 2.0

    q = q_criterion(field, tuple(float(s) for s in spacing))
    try:
        from scipy.ndimage import gaussian_filter

        # A one-cell smooth suppresses single-cell spikes (which otherwise
        # set the max and make any fraction of it nearly empty) and yields
        # coherent isosurfaces instead of dust.
        q = gaussian_filter(q, 1.0)
    except Exception:
        pass
    q_max = float(q.max())
    if not np.isfinite(q_max) or q_max <= 0.0:
        return None
    level = float(q_fraction * q_max)

    grid = pv.ImageData(
        dimensions=q.shape, spacing=tuple(spacing), origin=tuple(origin)
    )
    grid.point_data["q"] = q.ravel(order="C")  # x fastest, like VTK
    contour = grid.contour(isosurfaces=[level], scalars="q")
    if contour.n_cells == 0:
        return None
    sampler = GridSampler(field, spacing, origin)
    u = sampler(np.asarray(contour.points))
    contour.point_data["speed"] = np.linalg.norm(u, axis=1).astype(np.float32)
    return contour


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

MODEL_COLOR = "#b8c0c8"
FLOOR_COLOR = "#161a1f"
BACKGROUND_BOTTOM = (0.03, 0.04, 0.06)
BACKGROUND_TOP = (0.10, 0.14, 0.20)

CAMERA_ELEVATION_DEG = 18.0
CAMERA_AZIMUTH_START_DEG = 225.0   # three-quarter view, upstream-left
CAMERA_DISTANCE_FACTOR = 3.4       # x model length


def tunnel_from_meta(meta: dict) -> TunnelSpec:
    t = meta["tunnel"]
    return TunnelSpec(
        length=float(t["length"]),
        width=float(t["width"]),
        height=float(t["height"]),
        floor_z=float(t["floor_z"]),
        cells_streamwise=int(t["cells_streamwise"]),
    )


def rebuild_model(meta: dict, snapshot_dir: Path):
    """Rebuild the placed model from the stored placement spec."""
    placement = meta["placement"]
    raw = placement.get("model") or placement.get("path")
    if not raw:
        raise ValueError("meta.json has no model path; cannot rebuild the model")
    candidate = Path(raw)
    if not candidate.is_file():
        for base in (Path.cwd(), snapshot_dir):
            cand = base / raw
            if cand.is_file():
                candidate = cand
                break
    surface = load_surface(candidate)
    spec_dict = {
        k: v
        for k, v in placement.items()
        if k not in ("model", "placed_scale", "placed_bounds")
    }
    transform = MeshTransform(**spec_dict.pop("transform"))
    spec = MeshSpec(transform=transform, **spec_dict)
    build = build_model(surface, spec, tunnel_from_meta(meta))
    # Some exporters (including the bundled car OBJ) carry no normals;
    # compute consistent, outward-oriented ones or shading stays flat.
    if build.mesh.GetPointData().GetNormals() is None:
        build.mesh = build.mesh.compute_normals(
            consistent_normals=True, auto_orient_normals=True, point_normals=True
        )
    return build


def add_model(plotter: pv.Plotter, build) -> None:
    plotter.add_mesh(
        build.mesh,
        color=MODEL_COLOR,
        specular=0.7,
        specular_power=18.0,
        smooth_shading=True,
    )


def add_ground(plotter: pv.Plotter, tunnel: TunnelSpec) -> None:
    slab = pv.Box(
        bounds=(
            tunnel.x_min,
            tunnel.x_max,
            tunnel.y_min,
            tunnel.y_max,
            tunnel.floor_z - 0.02,
            tunnel.floor_z,
        )
    )
    plotter.add_mesh(slab, color=FLOOR_COLOR, opacity=0.95, show_edges=False)


def camera_for(model_bounds: np.ndarray, tau: float, t0: float, orbit_deg_per_sec: float):
    """[position, focal, view_up] for time tau, orbiting the model centre."""
    focal = model_bounds.mean(axis=1)
    model_len = float(model_bounds[0, 1] - model_bounds[0, 0])
    distance = CAMERA_DISTANCE_FACTOR * max(model_len, 1e-3)
    el = math.radians(CAMERA_ELEVATION_DEG)
    az = math.radians(
        CAMERA_AZIMUTH_START_DEG + orbit_deg_per_sec * (tau - t0)
    )
    pos = focal + distance * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    return [tuple(pos), tuple(focal), (0.0, 0.0, 1.0)]


def render_frames(
    snapshot_dir: Path,
    out: Path,
    layers: set[str],
    still_frame: int | None,
    preview: bool,
    orbit_deg_per_sec: float,
    tube_radius: float,
    q_fraction: float,
) -> tuple[int, float]:
    """Render every frame; returns (n_frames, seconds_per_frame)."""
    meta = load_meta(snapshot_dir)
    files = load_snapshot_files(snapshot_dir)
    solid = np.load(snapshot_dir / "solid.npy")
    lat = meta["lattice"]
    factor = float(lat["dx"]) / float(lat["dt"])   # lattice velocity -> m/s
    timeline = Timeline([step for step, _ in files], float(lat["dt"]))
    cache = FieldCache(files, factor)
    times = frame_times(timeline, preview=preview)
    if still_frame is not None:
        still_frame = min(max(still_frame, 0), len(times) - 1)
    t0 = times[0]

    build = rebuild_model(meta, snapshot_dir)
    model_bounds = np.asarray(build.mesh.bounds, dtype=float).reshape(3, 2)
    tunnel = tunnel_from_meta(meta)


    # Physical inlet speed, for a fixed colour scale across all layers.
    u_inlet = float(lat["u_lattice"]) * factor

    stream_tubes = None
    if "streamlines" in layers:
        last_field = load_velocity(files[-1][1], factor)
        stream_tubes = build_streamlines(meta, last_field, solid, tube_radius)
        if stream_tubes is None:
            print("streamlines: no pathlines survived integration", file=sys.stderr)

    smoke = SmokePlume(meta, solid) if "smoke" in layers else None

    resolution = (960, 540) if preview else (1920, 1080)
    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)

    writer = None
    if still_frame is None:
        writer = imageio.get_writer(
            str(out), fps=30, codec="libx264",
            pixelformat="yuv420p", macro_block_size=1,
        )

    started = time.perf_counter()
    n = len(times)
    for i, tau in enumerate(times):
        # Still mode must advance the stateful layers (smoke) through every
        # frame up to the target; stateless layers are skipped until then.
        if still_frame is not None and i != still_frame:
            if smoke is not None:
                i0, i1, alpha = timeline.bracket(tau)
                field = cache.sampled(i0, i1, alpha)
                smoke.advect(
                    GridSampler(field, meta["tunnel"]["spacing"], origin_of(meta)),
                    tau - (times[i - 1] if i else tau),
                )
            continue
        i0, i1, alpha = timeline.bracket(tau)
        field = cache.sampled(i0, i1, alpha)

        # clear actors only; clear() would delete the default light kit
        plotter.clear_actors()
        add_ground(plotter, tunnel)
        if "model" in layers:
            add_model(plotter, build)
        if "vortices" in layers:
            vortices = build_vortices(meta, field, q_fraction)
            if vortices is not None:
                plotter.add_mesh(
                    vortices, scalars="speed", cmap="turbo", clim=(0.5 * u_inlet, 1.5 * u_inlet),
                    opacity=0.4, smooth_shading=True, show_scalar_bar=False,
                )
        if stream_tubes is not None:
            plotter.add_mesh(
                stream_tubes, scalars="speed", cmap="turbo", clim=(0.5 * u_inlet, 1.5 * u_inlet),
                smooth_shading=True, show_scalar_bar=False,
            )
        if smoke is not None:
            smoke.advect(GridSampler(field, meta["tunnel"]["spacing"], origin_of(meta)),
                         tau - (times[i - 1] if i else tau))
            trail = smoke.polydata()
            if trail is not None:
                plotter.add_mesh(
                    trail, scalars="rgba", rgba=True, line_width=3,
                    show_scalar_bar=False,
                )

        plotter.camera_position = camera_for(
            model_bounds, tau, t0, orbit_deg_per_sec
        )
        if still_frame is not None:
            plotter.show(screenshot=str(out), auto_close=False)
        else:
            writer.append_data(plotter.screenshot(return_img=True))
        print(
            f"\r[{i + 1:>{len(str(n))}}/{n}] rendering {out.name}",
            end="", flush=True,
        )

    elapsed = time.perf_counter() - started
    print()
    if writer is not None:
        writer.close()
    plotter.close()
    return n, elapsed / max(n, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sstream-render",
        description="Layered render of sstream snapshots: streamlines, "
        "smoke, vortices, model. Reads the directory written by "
        "'sstream --snapshots DIR'.",
    )
    parser.add_argument("snapshot_dir", type=Path, help="snapshot directory")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output MP4 (movie) or PNG (with --still)",
    )
    parser.add_argument("--streamlines", action="store_true",
                        help="speed-coloured streamline tubes (last snapshot only)")
    parser.add_argument("--smoke", action="store_true",
                        help="RK2-advected smoke with 40-position fading trails")
    parser.add_argument("--vortices", action="store_true",
                        help="Q-criterion isosurface (level = fraction of max Q)")
    parser.add_argument("--model", action="store_true",
                        help="the placed model, glossy")
    parser.add_argument(
        "--still",
        type=int,
        metavar="FRAME",
        help="write a PNG of frame FRAME (0-based) instead of a movie",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="960x540, one frame per every 2nd snapshot (fast smoke test)",
    )
    parser.add_argument(
        "--orbit-deg-per-sec",
        type=float,
        default=10.0,
        help="camera orbit speed around the model (default 10 deg/s)",
    )
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=0.012,
        help="streamline tube radius in metres (default 0.012)",
    )
    parser.add_argument(
        "--q-fraction",
        type=float,
        default=0.2,
        help="vortex isosurface level as a fraction of smoothed max Q (default 0.2)",
    )
    args = parser.parse_args(argv)
    layers = {
        name
        for name in ("streamlines", "smoke", "vortices", "model")
        if getattr(args, name)
    }
    if not layers:
        parser.error("at least one of --streamlines/--smoke/--vortices/--model is required")
    if args.still is None and not args.out.suffix:
        args.out = args.out.with_suffix(".mp4")
    if args.still is not None and args.out.suffix.lower() != ".png":
        if not args.out.suffix:
            args.out = args.out.with_suffix(".png")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    layers = {
        name
        for name in ("streamlines", "smoke", "vortices", "model")
        if getattr(args, name)
    }
    try:
        n_frames, per_frame = render_frames(
            args.snapshot_dir,
            args.out,
            layers,
            still_frame=args.still,
            preview=args.preview,
            orbit_deg_per_sec=args.orbit_deg_per_sec,
            tube_radius=args.tube_radius,
            q_fraction=args.q_fraction,
        )
    except Exception as exc:  # CLI boundary: one clean message, exit code 1
        print(f"sstream-render: {exc}", file=sys.stderr)
        return 1
    if args.still is not None:
        print(
            f"Wrote {args.out} (still frame {args.still}; "
            f"timeline has {n_frames} frame(s))"
        )
    else:
        print(
            f"Wrote {args.out} ({n_frames} frames, "
            f"{per_frame * 1000.0:.0f} ms/frame, 30 fps)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
