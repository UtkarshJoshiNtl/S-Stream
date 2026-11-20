"""Shared helpers for the Re=100 sphere diagnostics.

These scripts are *diagnostic only*: they import the solver, mesh and pipeline
helpers but never modify them.  Run them from the repository root, for example::

    .venv/bin/python simulation/scripts/diagnostics/setup_audit.py

Everything is written in lattice-friendly units where density is one.  The
conversion back to SI follows :class:`sstream.lattice.LatticeUnits`; the drag
coefficient is evaluated in lattice units directly, which is algebraically the
same as the pipeline conversion because ``dt = u_lattice * dx / speed``.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[2]
ROOT = SIM_ROOT.parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from sstream.fields import frontal_area  # noqa: E402
from sstream.lattice import CS2, TAU_MIN, LatticeUnits  # noqa: E402
from sstream.lbm import (  # noqa: E402
    TUNNEL_OUTLET_MODE,
    TUNNEL_WALL_MODES,
    VALIDATION_WALL_MODES,
    LbmConfig,
    LbmSolver,
    pick_device,
)
from sstream.mesh import build_model, load_surface  # noqa: E402
from sstream.pipeline import freestream_probe_x  # noqa: E402
from sstream.schema import FluidSpec, MeshSpec, MeshTransform, TunnelSpec  # noqa: E402
from sstream.voxelize import VoxelResult, voxelize  # noqa: E402

ASSETS = SIM_ROOT / "assets" / "3d-files"
DATA_DIR = ROOT / "docs" / "diag_data"
DEVICE = pick_device()

_SURFACE = None


def sphere_surface():
    """The bundled sphere STL, read once per process."""
    global _SURFACE
    if _SURFACE is None:
        _SURFACE = load_surface(ASSETS / "sphere.stl")
    return _SURFACE


@dataclass
class SphereCase:
    """One sphere-in-a-tunnel setup, independent of how it is run."""

    label: str
    tunnel: TunnelSpec
    mask: VoxelResult
    fluid: FluidSpec
    units: LatticeUnits
    smagorinsky: float
    diameter_m: float
    extents_m: np.ndarray
    build_warnings: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.tunnel.grid_shape

    @property
    def dx(self) -> float:
        return self.tunnel.cell_size

    @property
    def diameter_cells(self) -> float:
        """Bounding-box mean diameter in cells, the quantity the tests use."""
        return self.diameter_m / self.dx

    @property
    def diameter_cells_x(self) -> float:
        return float(self.extents_m[0] / self.dx)

    @property
    def area_lattice(self) -> int:
        """Silhouette area in cell faces; the solver's frontal area in lattice units."""
        return int(np.asarray(self.mask.solid, dtype=bool).any(axis=0).sum())

    @property
    def area(self) -> float:
        return frontal_area(self.mask.solid, self.tunnel.grid_spacing)

    @property
    def blockage(self) -> float:
        return self.area / (self.tunnel.width * self.tunnel.height)

    @property
    def reynolds_lattice(self) -> float:
        return self.units.u_lattice * self.diameter_cells / self.units.nu_lattice

    @property
    def reynolds_lattice_x(self) -> float:
        return self.units.u_lattice * self.diameter_cells_x / self.units.nu_lattice

    @property
    def volume_cells(self) -> int:
        return int(np.asarray(self.mask.solid, dtype=bool).sum())

    @property
    def effective_diameter_cells(self) -> float:
        return (6.0 * self.volume_cells / math.pi) ** (1.0 / 3.0)

    @property
    def effective_diameter_ratio(self) -> float:
        return self.effective_diameter_cells / self.diameter_cells

    def cd_from_force(self, force_lattice, u: float | None = None) -> float:
        """Cd from a lattice force and the voxel silhouette area.

        ``u`` is the measured upstream freestream; when omitted the nominal
        inlet speed is used (kept for old traces and cross-checks only).
        """
        area = self.area_lattice
        if area <= 0:
            return float("nan")
        u = self.units.u_lattice if u is None else u
        return float(force_lattice[0] * 2.0 / (u * u * area))

    def lateral_from_force(self, force_lattice, u: float | None = None) -> tuple[float, float]:
        area = self.area_lattice
        if area <= 0:
            return (float("nan"), float("nan"))
        u = self.units.u_lattice if u is None else u
        scale = 2.0 / (u * u * area)
        return (float(force_lattice[1] * scale), float(force_lattice[2] * scale))


def _finish_case(
    label: str,
    tunnel: TunnelSpec,
    mask: VoxelResult,
    build_mesh,
    build_warnings: list[str],
    *,
    speed: float,
    reynolds: float,
    smagorinsky: float,
    u_lattice: float,
    note: str = "",
) -> SphereCase:
    bounds = np.asarray(build_mesh.bounds, dtype=float).reshape(3, 2)
    extents = bounds[:, 1] - bounds[:, 0]
    diameter = float(extents.mean())
    viscosity = speed * diameter / reynolds
    fluid = FluidSpec(
        speed=speed,
        kinematic_viscosity=viscosity,
        reference_length=diameter,
    )
    units = LatticeUnits.from_physical(
        speed,
        viscosity,
        tunnel.cell_size,
        u_lattice=u_lattice,
        smagorinsky=smagorinsky,
    )
    return SphereCase(
        label=label,
        tunnel=tunnel,
        mask=mask,
        fluid=fluid,
        units=units,
        smagorinsky=smagorinsky,
        diameter_m=diameter,
        extents_m=extents,
        build_warnings=list(build_warnings),
        note=note,
    )


def build_sphere_case(
    label: str,
    *,
    d_cells: float,
    smagorinsky: float,
    length_d: float = 10.0,
    width_d: float = 6.0,
    height_d: float = 6.0,
    position_x_d: float = 2.5,
    position_z_d: float = 0.0,
    speed: float = 0.05,
    reynolds: float = 100.0,
    physical_diameter: float = 1.0,
    margin: float = 0.05,
    note: str = "",
) -> SphereCase:
    """Build a sphere case with tunnel dimensions expressed in diameters.

    ``d_cells`` is the *requested* bounding-box diameter in cells.  The mesh
    is scaled to ``physical_diameter`` and the streamwise grid is
    ``round(length_d * d_cells)`` cells, so the realised diameter is within a
    fraction of a percent of the request.  ``position_z_d = 0`` rests the
    sphere on the floor, matching both pytest cases; use
    ``0.5 * (height_d - 1)`` for a free sphere.
    """
    length = length_d * physical_diameter
    tunnel = TunnelSpec(
        length=length,
        width=width_d * physical_diameter,
        height=height_d * physical_diameter,
        floor_z=0.0,
        cells_streamwise=int(round(length_d * d_cells)),
    )
    spec = MeshSpec(
        orient="none",
        fraction=physical_diameter / length,
        transform=MeshTransform(
            position_x=position_x_d * physical_diameter,
            position_z=position_z_d * physical_diameter,
        ),
        margin=margin,
    )
    build = build_model(sphere_surface(), spec, tunnel)
    mask = voxelize(build.mesh, tunnel)
    if mask.empty:
        raise RuntimeError(f"{label}: the voxelized sphere is empty")
    return _finish_case(
        label,
        tunnel,
        mask,
        build.mesh,
        build.warnings,
        speed=speed,
        reynolds=reynolds,
        smagorinsky=smagorinsky,
        u_lattice=speed,
        note=note,
    )


def build_test_sphere_case(
    label: str,
    *,
    tunnel: TunnelSpec,
    position_x: float | None,
    smagorinsky: float = 0.12,
    speed: float = 0.05,
    reynolds: float = 100.0,
    base_tunnel: TunnelSpec | None = None,
    note: str = "",
) -> SphereCase:
    """Reproduce a pytest sphere setup exactly.

    ``base_tunnel`` is the tunnel the mesh was originally placed in.  The
    outlet-reflection test builds the mesh once on the 4 m tunnel and then
    reuses it, unchanged, on the 6 m tunnel; passing ``base_tunnel`` gives the
    same absolute placement.
    """
    placement_tunnel = base_tunnel or tunnel
    transform = MeshTransform() if position_x is None else MeshTransform(position_x=position_x)
    spec = MeshSpec(orient="none", fraction=0.1, transform=transform)
    build = build_model(sphere_surface(), spec, placement_tunnel)
    mask = voxelize(build.mesh, tunnel)
    if mask.empty:
        raise RuntimeError(f"{label}: the voxelized sphere is empty")
    return _finish_case(
        label,
        tunnel,
        mask,
        build.mesh,
        build.warnings,
        speed=speed,
        reynolds=reynolds,
        smagorinsky=smagorinsky,
        u_lattice=speed,
        note=note,
    )


def exact_test_cases() -> list[tuple[SphereCase, int, str]]:
    """The two pytest sphere setups, plus the 2x-outlet reuse from test 2."""
    ref = TunnelSpec(length=4.0, width=2.4, height=2.4, floor_z=0.0, cells_streamwise=96)
    extended = TunnelSpec(length=6.0, width=2.4, height=2.4, floor_z=0.0, cells_streamwise=144)
    test1 = build_test_sphere_case(
        "test_sphere_re100 (default x=3.0)",
        tunnel=ref,
        position_x=None,
        note="MeshTransform.position_x defaults to 3.0 m",
    )
    test2_1x = build_test_sphere_case(
        "test_outlet_reflection 1x (x=1.0)",
        tunnel=ref,
        position_x=1.0,
    )
    test2_2x = build_test_sphere_case(
        "test_outlet_reflection 2x (x=1.0, L=6)",
        tunnel=extended,
        position_x=1.0,
        base_tunnel=ref,
        note="same mesh re-voxelized on the longer tunnel",
    )
    return [
        (test1, 12000, "pytest steps, Cs=0.12"),
        (test2_1x, 8000, "pytest steps, Cs=0.12"),
        (test2_2x, 8000, "pytest steps, Cs=0.12"),
    ]


@dataclass
class Trace:
    steps: np.ndarray
    cd: np.ndarray
    cy: np.ndarray
    cz: np.ndarray

    def __len__(self) -> int:
        return int(self.steps.size)

    def between(self, lo: int, hi: int) -> "Trace":
        keep = (self.steps >= lo) & (self.steps <= hi)
        return Trace(self.steps[keep], self.cd[keep], self.cy[keep], self.cz[keep])

    def window(self, n: int) -> "Trace":
        n = max(1, min(int(n), len(self)))
        return Trace(self.steps[-n:], self.cd[-n:], self.cy[-n:], self.cz[-n:])


def window_stats(trace: Trace, *, fraction: float = 0.2, min_samples: int = 20, max_samples: int = 80):
    """Last-window mean/std and the change against the previous window.

    The window is the final ``fraction`` of the sampled trace, clamped to
    ``[min_samples, max_samples]`` samples.  ``change`` compares the mean of
    the final window with the mean of the equally long window immediately
    before it, which is the same diagnostic the pipeline's convergence detector
    uses on a longer sample.
    """
    n = int(round(len(trace) * fraction))
    n = max(min_samples, min(max_samples, n))
    n = min(n, len(trace))
    last = trace.window(n)
    prev = trace.window(2 * n) if len(trace) >= 2 * n else trace
    prev = Trace(prev.steps[: len(prev) - n], prev.cd[: len(prev) - n], prev.cy[: len(prev) - n], prev.cz[: len(prev) - n])
    mean = float(np.mean(last.cd)) if len(last) else float("nan")
    std = float(np.std(last.cd, ddof=1)) if len(last) > 1 else 0.0
    prev_mean = float(np.mean(prev.cd)) if len(prev) else float("nan")
    change = abs(mean - prev_mean) / max(abs(mean), abs(prev_mean), 1e-12) if len(prev) else float("nan")
    step_lo = int(last.steps[0]) if len(last) else -1
    step_hi = int(last.steps[-1]) if len(last) else -1
    return {
        "n_samples": len(last),
        "step_lo": step_lo,
        "step_hi": step_hi,
        "window_steps": step_hi - step_lo + 1 if len(last) else 0,
        "mean_cd": mean,
        "std_cd": std,
        "prev_mean_cd": prev_mean,
        "change": change,
    }


def boundary_layer_metrics(case: SphereCase, velocity: np.ndarray) -> dict:
    """Floor-boundary-layer profile one diameter upstream of the sphere front.

    The reference speed is the local free-stream at the top of the plane, not
    the nominal ``u_lattice``: the tunnel's effective inflow decays during a
    run (see ``empty_tunnel`` in ``README``/the report), so using the nominal
    value would report a fake 6-diameter boundary layer everywhere.  The
    displacement thickness is the number that differs between the two pytest
    placements.
    """
    solid_idx = np.argwhere(np.asarray(case.mask.solid, dtype=bool))
    if solid_idx.size == 0:
        return {}
    centre = 0.5 * (solid_idx.min(axis=0) + solid_idx.max(axis=0))
    front = float(solid_idx[:, 0].min())
    nx, ny, nz = case.shape
    ix = int(np.clip(round(front - case.diameter_cells), 1, nx - 2))
    iy = int(np.clip(round(centre[1]), 2, ny - 3))
    profile = np.asarray(
        velocity[ix, iy - 2 : iy + 3, :, 0].mean(axis=0), dtype=np.float64
    )
    u_nom = float(case.units.u_lattice)
    # Local free-stream from the top quarter of the tunnel at this station.
    u_ref = float(np.mean(profile[int(0.75 * nz) :]))
    u_ref = max(u_ref, 1e-12)
    iz_centre = int(np.clip(round(centre[2]), 0, nz - 1))
    above = np.where(profile >= 0.99 * u_ref)[0]
    delta99_idx = int(above[0]) if above.size else nz
    delta99_d = (float(delta99_idx) + 0.5) * case.dx / case.diameter_m
    k = max(1, delta99_idx)
    delta_star_cells = float(np.sum(1.0 - profile[:k] / u_ref))
    delta_star_d = delta_star_cells * case.dx / case.diameter_m
    u_centre = float(profile[iz_centre])
    return {
        "bl_plane_x_cells": ix,
        "bl_u_ref": u_ref,
        "bl_u_ref_over_nominal": u_ref / max(u_nom, 1e-12),
        "bl_u_at_centre": u_centre,
        "bl_u_centre_over_ref": u_centre / u_ref,
        "bl_q_centre_over_nominal": u_centre * u_centre / max(u_nom * u_nom, 1e-12),
        "bl_delta99_d": delta99_d,
        "bl_delta_star_d": delta_star_d,
    }


def run_sphere(
    case: SphereCase,
    *,
    steps: int,
    sample_every: int | None = None,
    save_crop: Path | None = None,
    check_every: int | None = None,
    log_prefix: str = "",
    wall_modes: dict | None = None,
    outlet_mode: str | None = None,
    moving_wall_speed: float | None = None,
):
    """Run one case directly on :class:`LbmSolver` and sample the wall force.

    Returns ``(trace, info)``.  ``info`` contains the final velocity/density
    arrays and the wall time.  ``sample_every`` defaults to about 400 samples
    over the run, which keeps the CSV/PNG readable while still resolving the
    approach to steady state.  Validation runs should pass ``wall_modes=
    VALIDATION_WALL_MODES`` so every wall is slip.
    """
    steps = int(steps)
    if sample_every is None:
        sample_every = max(5, steps // 400)
    sample_every = max(1, int(sample_every))
    if check_every is None:
        check_every = max(1000, steps // 20)
    if wall_modes is None:
        wall_modes = TUNNEL_WALL_MODES
    if outlet_mode is None:
        outlet_mode = TUNNEL_OUTLET_MODE
    implies_moving = any(mode == "moving" for mode in wall_modes.values())
    if moving_wall_speed is None:
        moving_wall_speed = case.units.u_lattice if implies_moving else 0.0

    solver = LbmSolver(
        LbmConfig(
            shape=case.shape,
            tau=case.units.tau,
            smagorinsky=case.smagorinsky,
            periodic=(False, False, False),
            inlet_speed=case.units.u_lattice,
            wall_modes=wall_modes,
            outlet_mode=outlet_mode,
            moving_wall_speed=moving_wall_speed,
            device=DEVICE,
        ),
        case.mask.solid,
    )
    solver.initialize(ux=case.units.u_lattice)
    x_probe = freestream_probe_x(case.mask.solid)

    out_steps: list[int] = []
    out_cd: list[float] = []
    out_cy: list[float] = []
    out_cz: list[float] = []
    probe_speeds: list[float] = []
    max_speed_seen = 0.0
    finite = True
    completed = 0
    t0 = __import__("time").perf_counter()

    for step in range(1, steps + 1):
        solver.step()
        completed = step
        if step % sample_every == 0:
            force = solver.total_wall_force()
            u_ref = solver.plane_speed(x_probe)
            probe_speeds.append(u_ref)
            cd = case.cd_from_force(force, u_ref)
            cy, cz = case.lateral_from_force(force, u_ref)
            if not (math.isfinite(cd) and math.isfinite(cy) and math.isfinite(cz)):
                finite = False
                break
            out_steps.append(step)
            out_cd.append(cd)
            out_cy.append(cy)
            out_cz.append(cz)
        if step % check_every == 0:
            speed = solver.max_speed()
            max_speed_seen = max(max_speed_seen, speed)
            if not math.isfinite(speed) or speed > 0.5:
                finite = False
                break

    elapsed = __import__("time").perf_counter() - t0
    velocity = solver.velocity()
    density = solver.density()
    max_speed_seen = max(max_speed_seen, solver.max_speed())

    if save_crop is not None:
        _save_field_crop(save_crop, case, solver, velocity, density)

    bl = boundary_layer_metrics(case, velocity)
    info = {
        "steps_requested": steps,
        "steps_completed": completed,
        "sample_every": sample_every,
        "elapsed_s": elapsed,
        "mlups": case.shape[0] * case.shape[1] * case.shape[2] * completed / max(elapsed, 1e-9) / 1e6,
        "max_speed": max_speed_seen,
        "finite": finite,
        "velocity": velocity,
        "density": density,
        "shape": case.shape,
        "device": solver.device,
        "freestream_probe_x": x_probe,
        "freestream_lattice": float(np.mean(probe_speeds)) if probe_speeds else case.units.u_lattice,
        **bl,
    }
    trace = Trace(
        steps=np.asarray(out_steps, dtype=np.int64),
        cd=np.asarray(out_cd, dtype=np.float64),
        cy=np.asarray(out_cy, dtype=np.float64),
        cz=np.asarray(out_cz, dtype=np.float64),
    )
    if log_prefix:
        print(
            f"{log_prefix}{case.label}: {completed:,}/{steps:,} steps, "
            f"{len(trace)} samples, {elapsed:.1f} s, {info['mlups']:.1f} MLUPs, "
            f"max|u|={max_speed_seen:.4f}",
            flush=True,
        )
    return trace, info


def _save_field_crop(path: Path, case: SphereCase, solver: LbmSolver, velocity, density):
    """Save a cropped velocity/density/mask plus the floor populations.

    The crop is the control-volume box used by ``force_balance.py`` with a
    three-cell margin for finite differences.  Only the first z layer of
    populations is needed (the floor no-slip shear), so the file stays small.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = case.shape
    d = case.diameter_cells
    centre = _sphere_center_cells(case)
    x0 = max(1, int(round(centre[0] - 2.0 * d)))
    x1 = min(nx - 2, int(round(centre[0] + 4.0 * d)))
    y0 = max(1, int(round(centre[1] - 2.0 * d)))
    y1 = min(ny - 2, int(round(centre[1] + 2.0 * d)))
    floor_contact = bool(np.asarray(case.mask.solid[:, :, 0], dtype=bool).any())
    if floor_contact:
        z0 = 0
        z1 = min(nz - 2, int(round(centre[2] + 2.0 * d)))
    else:
        z0 = max(1, int(round(centre[2] - 2.0 * d)))
        z1 = min(nz - 2, int(round(centre[2] + 2.0 * d)))
    m = 3
    xs = slice(max(0, x0 - m), min(nx, x1 + 1 + m))
    ys = slice(max(0, y0 - m), min(ny, y1 + 1 + m))
    zs = slice(max(0, z0 - m), min(nz, z1 + 1 + m))
    # Slice on the device first: ``f`` is ~740 MB at D=30, and only the floor
    # layer over the control-volume footprint is needed for the wall shear.
    if floor_contact:
        f_floor = solver.f[:, x0 : x1 + 1, y0 : y1 + 1, 0].numpy()
    else:
        f_floor = np.zeros((0,), dtype=np.float32)
    force_me = solver.total_wall_force()
    np.savez_compressed(
        path,
        velocity=np.asarray(velocity[xs, ys, zs], dtype=np.float32),
        density=np.asarray(density[xs, ys, zs], dtype=np.float32),
        solid=np.asarray(case.mask.solid[xs, ys, zs], dtype=bool),
        f_floor=np.asarray(f_floor, dtype=np.float32),
        floor_contact=np.array([floor_contact], dtype=bool),
        force_lattice=np.asarray(force_me, dtype=np.float64),
        crop=np.array([xs.start, xs.stop, ys.start, ys.stop, zs.start, zs.stop], dtype=np.int64),
        cv=np.array([x0, x1, y0, y1, z0, z1], dtype=np.int64),
        diameter_cells=np.array([case.diameter_cells], dtype=np.float64),
        tau=np.array([case.units.tau], dtype=np.float64),
        u_lattice=np.array([case.units.u_lattice], dtype=np.float64),
        area_lattice=np.array([case.area_lattice], dtype=np.int64),
        smagorinsky=np.array([case.smagorinsky], dtype=np.float64),
        dx=np.array([case.dx], dtype=np.float64),
        label=np.array([case.label]),
    )
    print(f"  saved control-volume crop -> {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def _centre_x(case: SphereCase) -> float:
    """Physical x of the sphere centre, in metres."""
    solid = np.asarray(case.mask.solid, dtype=bool)
    xs = np.where(solid.any(axis=(1, 2)))[0]
    if xs.size == 0:
        return 0.0
    return float(case.tunnel.grid_origin[0] + 0.5 * (xs.min() + xs.max()) * case.dx)


def _sphere_center_cells(case: SphereCase) -> np.ndarray:
    solid = np.asarray(case.mask.solid, dtype=bool)
    idx = np.argwhere(solid)
    if idx.size == 0:
        return np.zeros(3)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    return 0.5 * (lo + hi)


def recirculation_length(case: SphereCase, velocity: np.ndarray, *, threshold: float = 0.0):
    """Centreline reverse-flow length behind the sphere, in sphere diameters.

    The line is the y centreline at the sphere-centre height.  The start is the
    last solid cell on that line; the end is the first cell downstream where
    ``u_x >= threshold``.  Returned with the raw cell count and the centreline
    minimum velocity for context.
    """
    solid = np.asarray(case.mask.solid, dtype=bool)
    ux = np.asarray(velocity[..., 0], dtype=np.float64)
    _, iy, iz = _sphere_center_cells(case).astype(int)
    iy = int(np.clip(iy, 1, case.shape[1] - 2))
    iz = int(np.clip(iz, 1, case.shape[2] - 2))
    solid_line = solid[:, iy, iz]
    if not solid_line.any():
        # The centre-height cell can be just outside the voxel surface; search z.
        for dz in (1, -1, 2, -2):
            j = int(np.clip(iz + dz, 1, case.shape[2] - 2))
            if solid[:, iy, j].any():
                iz = j
                solid_line = solid[:, iy, iz]
                break
    if not solid_line.any():
        return {"length_cells": float("nan"), "length_d": float("nan"), "min_ux": float("nan")}
    rear = int(np.where(solid_line)[0].max())
    line = ux[rear + 1 :, iy, iz]
    # Attached recirculation: the reverse region must start within the first
    # few fluid cells behind the body.  A detached pocket farther downstream
    # is not the near-wake recirculation Johnson & Patel measure.
    first = None
    for i in range(min(3, line.size)):
        if line[i] < threshold:
            first = i
            break
    if first is None:
        length_cells = 0.0
    else:
        end = first
        while end < line.size and line[end] < threshold:
            end += 1
        length_cells = float(end - first)
    min_ux = float(np.min(line)) if line.size else float("nan")
    return {
        "length_cells": length_cells,
        "length_d": length_cells / case.diameter_cells,
        "min_ux": min_ux,
        "rear_cell": rear,
        "line_y": iy,
        "line_z": iz,
        "threshold": threshold,
        "threshold_over_u": threshold / case.units.u_lattice,
    }


def write_trace_csv(path: Path, trace: Trace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cy", "cz"])
        for row in zip(trace.steps, trace.cd, trace.cy, trace.cz):
            writer.writerow([int(row[0]), f"{row[1]:.8g}", f"{row[2]:.8g}", f"{row[3]:.8g}"])
    return path


def plot_trace(path: Path, trace: Trace, case: SphereCase, stats: dict | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(trace.steps, trace.cd, lw=0.8, color="tab:blue", label="Cd")
    if stats is not None and math.isfinite(stats.get("mean_cd", float("nan"))):
        ax.axhline(stats["mean_cd"], color="tab:red", lw=0.8, ls="--", label="last-window mean")
        ax.axvspan(stats["step_lo"], stats["step_hi"], color="tab:red", alpha=0.08, label="last window")
        ax.fill_between(
            [stats["step_lo"], stats["step_hi"]],
            stats["mean_cd"] - stats["std_cd"],
            stats["mean_cd"] + stats["std_cd"],
            color="tab:red",
            alpha=0.12,
        )
    ax.set_ylabel("Cd")
    ax.set_title(case.label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax2.plot(trace.steps, trace.cy, lw=0.7, color="tab:green", label="Cy")
    ax2.plot(trace.steps, trace.cz, lw=0.7, color="tab:orange", label="Cz")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_xlabel("lattice step")
    ax2.set_ylabel("Cy, Cz")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fmt(value, digits: int = 4) -> str:
    """Compact number formatting for the audit tables."""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[list[str]], headers: list[str]):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
