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
from typing import Callable

import numpy as np
import warp as wp

from . import fields
from .dye import DyeField
from .lattice import TAU_MIN, LatticeUnits
from .lbm import LbmConfig, LbmSolver
from .preview import RunPreset
from .quality import WARMUP_STEPS
from .schema import FluidSpec, TunnelSpec
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

    def __post_init__(self) -> None:
        if self.mask.dimensions != self.tunnel.grid_shape:
            raise ValueError("the mask and tunnel grids do not match")
        if not 0.02 <= self.u_lattice <= 0.12:
            raise ValueError("u_lattice must stay in the weakly compressible range")


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
    warnings: list[str] = field(default_factory=list)

    @property
    def cells(self) -> int:
        return int(np.prod(self.tunnel.grid_shape))

    @property
    def reynolds(self) -> float:
        return self.fluid.reynolds


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
    units = LatticeUnits.from_physical(
        fluid.speed,
        fluid.kinematic_viscosity,
        tunnel.cell_size,
        u_lattice=plan.u_lattice,
        smagorinsky=plan.smagorinsky,
    )

    report(1, "Preparing solver")
    check()
    solver = LbmSolver(
        LbmConfig(
            shape=shape,
            tau=units.tau,
            smagorinsky=plan.smagorinsky,
            periodic=(False, False, False),
            inlet_speed=units.u_lattice,
        ),
        plan.mask.solid,
    )
    solver.initialize(ux=units.u_lattice)

    stripe = max(3, min(shape[1], shape[2]) // 8)
    dye = DyeField(shape, solver.device, stripe=stripe, decay=0.99995)

    steps = max(1, int(preset.steps))
    start = steps // 2
    sample_every = max(1, (steps - start) // 160)
    drag, lift, lateral = [], [], []
    clock = time.perf_counter()
    warm = min(WARMUP_STEPS, steps)
    warmup_mlups = 0.0
    cells = shape[0] * shape[1] * shape[2]

    # Lattice Mach guard: past ~0.3 the weakly compressible model's errors
    # dominate, so the run is flagged instead of silently reporting garbage.
    mach_limit = 0.3
    mach_tripped = False

    def do_step(step: int) -> None:
        nonlocal mach_tripped
        solver.step()
        dye.step(solver.u, solver.solid)
        if step >= start and (step - start) % sample_every == 0:
            force = solver.total_wall_force()
            drag.append(force[0])
            lateral.append(force[1])
            lift.append(force[2])
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

    report(1, "Warming up")
    warm_start = time.perf_counter()
    for step in range(1, warm + 1):
        do_step(step)
    wp.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    if warm and warm_seconds > 0.0:
        warmup_mlups = cells * warm / warm_seconds / 1e6
        report(
            2,
            f"Warm-up {warm} steps at {warmup_mlups:.1f} MLUPs · "
            f"est {steps:,} steps ≈ {steps / warmup_mlups:.1f} s",
        )

    for step in range(warm + 1, steps + 1):
        do_step(step)

    check()
    report(96, "Deriving fields")
    velocity = solver.velocity()
    density = solver.density()
    curl = fields.vorticity(velocity, tunnel.grid_spacing)
    dye_field = dye.numpy()

    area = fields.frontal_area(plan.mask.solid, tunnel.grid_spacing)
    scale = fluid.density * units.dx**4 / units.dt**2 / (fluid.dynamic_pressure * area)
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
        cd_trace=cd_trace,
        cl_trace=cl_trace,
        steps=steps,
        elapsed=time.perf_counter() - clock,
        device=solver.device,
        warmup_mlups=warmup_mlups,
        warnings=warnings,
    )
