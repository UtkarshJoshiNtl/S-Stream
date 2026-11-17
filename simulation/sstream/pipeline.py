"""One complete solve: mask in, fields and coefficients out.

This module is the boundary between the front end and the physics.  It
deliberately imports no GUI or plotting library, so the whole run can be
executed and tested from a script.  A front end only has to supply a
`should_stop()` callback.

What is stored in :class:`SolveResult` is the minimum needed to draw the
result and write a report: the velocity, its curl, the dye, density, and the
force history.  Nothing in here decides how the result looks.
"""

from __future__ import annotations

import time
import warnings as py_warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import warp as wp

from . import fields
from .dye import DyeField
from .lattice import (
    TAU_MIN,
    VISUAL_SMAGORINSKY,
    VISUAL_U_LATTICE,
    LatticeUnits,
    clamp_visual_tau,
)
from .lbm import (
    TUNNEL_OUTLET_MODE,
    TUNNEL_WALL_MODES,
    LbmConfig,
    LbmSolver,
)
from .preview import RunPreset
from .quality import WARMUP_STEPS, format_bytes
from .schema import FluidSpec, TunnelSpec
from .snapshots import (
    check_disk,
    snapshot_bytes,
    snapshot_steps,
    steps_per_flow_through,
    write_meta,
    write_solid_mask,
    write_velocity,
)
from .voxelize import VoxelResult


class SolveCancelled(Exception):
    """Raised when the caller asks a running solve to stop."""


@dataclass(frozen=True)
class SolvePlan:
    mask: VoxelResult
    tunnel: TunnelSpec
    fluid: FluidSpec
    preset: RunPreset
    u_lattice: float = 0.06
    smagorinsky: float = 0.12
    visual: bool = False
    convergence_tol: float | None = None
    convergence_window: int = 40
    snapshots_dir: str | None = None
    snapshot_count: int = 0
    snapshot_dtype: str = "float16"
    placement: dict | None = None

    def __post_init__(self) -> None:
        if self.mask.dimensions != self.tunnel.grid_shape:
            raise ValueError("the mask and tunnel grids do not match")
        if not 0.02 <= self.u_lattice <= 0.12:
            raise ValueError("u_lattice must stay in the weakly compressible range")
        if self.convergence_tol is not None and self.convergence_tol <= 0.0:
            raise ValueError("convergence_tol must be positive when set")
        if self.convergence_window < 5:
            raise ValueError("convergence_window must be >= 5")


@dataclass
class SolveResult:
    tunnel: TunnelSpec
    fluid: FluidSpec
    preset: RunPreset
    mask: VoxelResult
    units: LatticeUnits
    velocity: np.ndarray
    vorticity: np.ndarray
    dye: np.ndarray
    density: np.ndarray
    force: np.ndarray
    cd: float
    cl: float
    cl_amplitude: float
    cd_trace: np.ndarray
    cl_trace: np.ndarray
    steps: int
    elapsed: float
    device: str
    warmup_mlups: float = 0.0
    converged: bool | None = None
    convergence_change: float | None = None
    convergence_tol: float | None = None
    snapshot_dir: str | None = None
    snapshot_steps: list[int] = field(default_factory=list)
    snapshot_bytes: int = 0
    warnings: list[str] = field(default_factory=list)
    freestream_lattice: float = 0.0
    freestream_probe_x: int = 0

    @property
    def cells(self) -> int:
        return int(np.prod(self.tunnel.grid_shape))

    @property
    def reynolds(self) -> float:
        return self.fluid.reynolds


def freestream_probe_x(solid: np.ndarray, upstream_body_lengths: float = 1.5) -> int:
    """x index of the freestream probe plane, ahead of the body's leading edge.

    The plane sits ``upstream_body_lengths`` body lengths in front of the
    first occupied x index, clamped to the interior of the grid.  A mask with
    no solid cells has no body to measure against, so its mid-plane is used.
    """
    solid = np.asarray(solid, dtype=bool)
    nx = solid.shape[0]
    if not solid.any():
        return nx // 2
    occupied = np.where(solid.any(axis=(1, 2)))[0]
    front = int(occupied.min())
    length = int(occupied.max() - occupied.min() + 1)
    return int(np.clip(round(front - upstream_body_lengths * length), 2, nx - 3))


def run_case(
    plan: SolvePlan,
    progress: Callable[[int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SolveResult:
    """Solve one case, reporting integer progress and checking for a cancel."""
    report = progress or (lambda _percent, _message: None)
    stop = should_stop or (lambda: False)

    def check() -> None:
        if stop():
            raise SolveCancelled()

    tunnel, fluid, preset = plan.tunnel, plan.fluid, plan.preset
    shape = tunnel.grid_shape
    u_lattice = VISUAL_U_LATTICE if plan.visual else plan.u_lattice
    smagorinsky = VISUAL_SMAGORINSKY if plan.visual else plan.smagorinsky
    units = LatticeUnits.from_physical(
        fluid.speed,
        fluid.kinematic_viscosity,
        tunnel.cell_size,
        u_lattice=u_lattice,
        smagorinsky=smagorinsky,
    )
    if plan.visual:
        units = units.with_tau(clamp_visual_tau(units.tau))

    # --- Snapshot plan: decide the steps, estimate disk, refuse early. ---
    cells_early = shape[0] * shape[1] * shape[2]
    snap_steps: list[int] = []
    snap_dir = Path(plan.snapshots_dir) if plan.snapshots_dir else None
    if plan.snapshot_count and snap_dir is None:
        raise ValueError("snapshots require a --snapshots DIR")
    if snap_dir is not None and plan.snapshot_count < 1:
        raise ValueError("snapshots require --snapshot-count N >= 1")
    if snap_dir is not None:
        snap_dir.mkdir(parents=True, exist_ok=True)
        per_flow_through = steps_per_flow_through(tunnel.length, fluid.speed, units.dt)
        snap_steps = snapshot_steps(
            max(1, int(preset.steps)), per_flow_through, plan.snapshot_count
        )
        estimated = snapshot_bytes(cells_early, len(snap_steps), plan.snapshot_dtype)
        free = check_disk(snap_dir, estimated)
        report(
            0,
            f"Snapshots    {len(snap_steps)} files, est {format_bytes(estimated)} "
            f"-> {snap_dir} ({format_bytes(free)} free)",
        )
    snap_set = set(snap_steps)

    report(1, "Preparing solver")
    check()
    solver = LbmSolver(
        LbmConfig(
            shape=shape,
            tau=units.tau,
            smagorinsky=smagorinsky,
            periodic=(False, False, False),
            inlet_speed=u_lattice,
            wall_modes=TUNNEL_WALL_MODES,
            outlet_mode=TUNNEL_OUTLET_MODE,
        ),
        plan.mask.solid,
    )
    solver.initialize(ux=u_lattice)

    stripe = max(3, min(shape[1], shape[2]) // 8)
    dye = DyeField(shape, solver.device, stripe=stripe, decay=0.99995)

    steps = max(1, int(preset.steps))
    start = steps // 2
    sample_every = max(1, (steps - start) // 160)
    x_probe = freestream_probe_x(plan.mask.solid)
    drag, lift, lateral, probe_speeds = [], [], [], []
    clock = time.perf_counter()
    warm = min(WARMUP_STEPS, steps)
    warmup_mlups = 0.0
    cells = cells_early
    snap_bytes_written = 0
    snap_files_written = 0
    converged: bool | None = None
    convergence_change: float | None = None
    converged_step = steps

    # Lattice Mach guard: past ~0.3 the weakly compressible model's errors
    # dominate, so the run is flagged instead of silently reporting garbage.
    mach_limit = 0.3
    mach_tripped = False

    def _check_convergence() -> bool:
        nonlocal converged, convergence_change
        if plan.convergence_tol is None:
            return False
        w = plan.convergence_window
        if len(drag) < 2 * w:
            return False
        window = np.asarray(drag[-2 * w :], dtype=np.float64)
        prev = window[:w].mean()
        curr = window[w:].mean()
        change = abs(float(curr) - float(prev)) / max(abs(float(curr)), abs(float(prev)), 1e-12)
        convergence_change = change
        if change < float(plan.convergence_tol):
            converged = True
            return True
        return False

    def do_step(step: int) -> bool:
        nonlocal mach_tripped, snap_bytes_written, snap_files_written, converged_step
        solver.step()
        dye.step(solver.u, solver.solid)
        if snap_dir is not None and step in snap_set:
            snap_bytes_written += write_velocity(
                snap_dir, step, solver.u.numpy(), plan.snapshot_dtype
            )
            snap_files_written += 1
        if step >= start and (step - start) % sample_every == 0:
            force = solver.total_wall_force()
            drag.append(force[0])
            lateral.append(force[1])
            lift.append(force[2])
            probe_speeds.append(solver.plane_speed(x_probe))
            if _check_convergence():
                converged_step = step
                check()
                report(
                    1 + int(94 * step / steps),
                    f"Converged · step {step:,} / {steps:,} "
                    f"(ΔCd window {convergence_change:.3%} < {plan.convergence_tol:.3%})",
                )
                return True
        if step % 64 == 0 and not mach_tripped:
            if solver.max_speed() > mach_limit:
                mach_tripped = True
                py_warnings.warn(
                    f"max lattice speed exceeded {mach_limit} (Mach guard); "
                    "the low-Mach assumption is invalid and the force "
                    "history is not meaningful",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if step % 16 == 0:
            check()
            report(1 + int(94 * step / steps), f"Solving · step {step:,} / {steps:,}")
        return False

    report(1, "Warming up")
    warm_start = time.perf_counter()
    for step in range(1, warm + 1):
        if do_step(step):
            break
    wp.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    warm_done = min(warm, converged_step)
    if warm_done and warm_seconds > 0.0:
        warmup_mlups = cells * warm_done / warm_seconds / 1e6
        report(
            2,
            f"Warm-up {warm_done} steps at {warmup_mlups:.1f} MLUPs · "
            f"est {steps:,} steps ≈ {steps / warmup_mlups:.1f} s",
        )

    final_step = converged_step
    if converged_step == steps:
        for step in range(warm + 1, steps + 1):
            if do_step(step):
                final_step = converged_step
                break
        else:
            final_step = steps

    check()
    report(96, "Deriving fields")
    velocity = solver.velocity()
    density = solver.density()
    curl = fields.vorticity(velocity, tunnel.grid_spacing)
    dye_field = dye.numpy()

    area = fields.frontal_area(plan.mask.solid, tunnel.grid_spacing)
    # Cd uses the *measured* freestream at the upstream probe, averaged over
    # the same force window as the force samples.  The nominal inlet speed is
    # only a setup parameter: a boundary layer or an unsettled tunnel makes it
    # differ from the flow the body actually sees.
    u_ref = float(np.mean(probe_speeds)) if probe_speeds else u_lattice
    if not np.isfinite(u_ref) or u_ref <= 0.0:
        u_ref = u_lattice
    u_ref_physical = u_ref * units.dx / units.dt
    dynamic_pressure_ref = 0.5 * fluid.density * u_ref_physical**2
    scale = fluid.density * units.dx**4 / units.dt**2 / (dynamic_pressure_ref * area)
    drag = np.asarray(drag, dtype=np.float64)
    lift = np.asarray(lift, dtype=np.float64)
    # A body in a +x-going flow is pushed downstream: force on the body in
    # +x is positive drag, so Cd uses +force[0] without any sign flip.
    cd_trace = drag * scale
    cl_trace = lift * scale
    mean_force = np.array(
        [
            drag.mean() if drag.size else 0.0,
            np.mean(lateral) if lateral else 0.0,
            lift.mean() if lift.size else 0.0,
        ]
    )

    if plan.convergence_tol is not None and converged is not True:
        converged = False

    warnings: list[str] = list(plan.mask.warnings)
    if plan.mask.solid_count / cells > 0.10:
        warnings.append("The model blocks more than 10% of the tunnel volume; Cd is confinement-biased.")
    blockage = area / (tunnel.width * tunnel.height)
    if blockage > 0.10:
        warnings.append(f"The model blocks {blockage:.0%} of the tunnel cross-section; Cd is confinement-biased.")
    if abs(units.tau - TAU_MIN) < 1e-3:
        warnings.append("The eddy-viscosity floor is active; treat Cd as a relative LES value.")
    if mach_tripped:
        warnings.append(
            "max lattice speed exceeded 0.3; the low-Mach assumption is invalid, "
            "Cd/Cl are not meaningful."
        )
    if not plan.mask.watertight:
        warnings.append("The surface is not watertight; the solid mask is approximate.")
    if plan.visual:
        warnings.append(
            "Visual mode: tau pinned and parameters chosen for stability/visuals; "
            "Cd not calibrated."
        )
    if plan.convergence_tol is not None:
        if converged:
            warnings.append(
                f"Convergence: reached at step {final_step} "
                f"(drag-window change {convergence_change:.3%} < tolerance {plan.convergence_tol:.3%})."
            )
        else:
            if convergence_change is None:
                warnings.append(
                    "Convergence: never converged (insufficient drag samples to evaluate "
                    f"tolerance {plan.convergence_tol:.3%} within {final_step} steps)."
                )
            else:
                warnings.append(
                    f"Convergence: never converged within {final_step} steps "
                    f"(drag-window change {convergence_change:.3%} >= tolerance {plan.convergence_tol:.3%})."
                )

    if snap_dir is not None:
        write_solid_mask(snap_dir, plan.mask.solid)
        write_meta(
            snap_dir, tunnel, plan.placement or {}, snap_steps,
            units, plan.smagorinsky,
        )
        report(
            99,
            f"Snapshots    wrote {snap_files_written} files "
            f"({format_bytes(snap_bytes_written)}) -> {snap_dir}",
        )

    report(100, "Done")
    return SolveResult(
        tunnel=tunnel,
        fluid=fluid,
        preset=preset,
        mask=plan.mask,
        units=units,
        velocity=velocity,
        vorticity=curl,
        dye=dye_field,
        density=density,
        force=mean_force,
        cd=float(cd_trace.mean()) if cd_trace.size else 0.0,
        cl=float(cl_trace.mean()) if cl_trace.size else 0.0,
        cl_amplitude=float(0.5 * (cl_trace.max() - cl_trace.min())) if cl_trace.size else 0.0,
        freestream_lattice=u_ref,
        freestream_probe_x=x_probe,
        cd_trace=cd_trace,
        cl_trace=cl_trace,
        steps=final_step,
        elapsed=time.perf_counter() - clock,
        device=solver.device,
        warmup_mlups=warmup_mlups,
        converged=converged,
        convergence_change=convergence_change,
        convergence_tol=plan.convergence_tol,
        snapshot_dir=str(snap_dir) if snap_dir else None,
        snapshot_steps=snap_steps,
        snapshot_bytes=snap_bytes_written,
        warnings=warnings,
    )
