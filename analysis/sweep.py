from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from analysis.physics import drag_coefficient, reynolds_number
from engines.lbm2d import LBM2D
from scene.scene import Scene, apply_to_sim


@dataclass
class SweepResult:
    parameter: str
    values: list[float]
    measurements: list[str]
    data: dict[str, list[float]] = field(default_factory=dict)
    elapsed: float = 0.0
    # A cancelled sweep returns the values it finished, and says so. Returning the
    # partial data *without* this flag would hand the plot a short line that looks
    # exactly like a complete one -- the data is shorter than `values`, which is the
    # only thing that distinguishes them, and a reader of the saved JSON has no way to
    # tell that apart from a sweep that really was that short.
    cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "values": self.values,
            "measurements": self.measurements,
            "data": self.data,
            "elapsed": self.elapsed,
            "cancelled": self.cancelled,
        }

    @staticmethod
    def from_dict(d: dict) -> SweepResult:
        return SweepResult(
            parameter=d["parameter"],
            values=d["values"],
            measurements=d["measurements"],
            data=d["data"],
            elapsed=d.get("elapsed", 0.0),
            # Absent in documents written before this field existed, which are
            # necessarily complete rather than cancelled.
            cancelled=d.get("cancelled", False),
        )


_MEASUREMENT_FUNCS: dict[str, Callable] = {
    "reynolds_number": lambda sim, obs: reynolds_number(sim),
    "drag_coefficient": lambda sim, obs: drag_coefficient(sim),
    "max_speed": lambda sim, obs: float(
        np.sqrt(
            sim.get_velocity()[:, :, 0] ** 2 + sim.get_velocity()[:, :, 1] ** 2
        ).max()
    ),
    "max_vorticity": lambda sim, obs: _max_vorticity(sim),
}


# Steps between cancellation checks. Small enough that a cancel lands promptly,
# large enough that the per-call overhead stays irrelevant next to the stepping.
_CANCEL_CHECK_INTERVAL = 250


def _max_vorticity(sim) -> float:
    vel = sim.get_velocity()
    u = vel[:, :, 0]
    v = vel[:, :, 1]
    dvdx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
    return float(np.abs(dvdx - dudy).max())


def _resolve_parameter(scene: Scene, param: str) -> tuple[str, float]:
    """Get the attribute path and current value for a parameter name.

    Supports: 'viscosity', 'u_inflow', 'smoke_diffusion', 'smoke_decay',
    or obstacle fields like 'cylinder.radius'.
    """
    if "." in param:
        parts = param.split(".")
        for obs in scene.obstacles:
            if obs.name == parts[0] and hasattr(obs, parts[1]):
                return parts[1], float(getattr(obs, parts[1]))
        msg = f"Obstacle '{parts[0]}' not found or has no field '{parts[1]}'"
        raise ValueError(msg)
    if hasattr(scene, param):
        return param, float(getattr(scene, param))
    msg = f"Unknown parameter: {param}"
    raise ValueError(msg)


def _set_parameter(scene: Scene, param: str, value: float) -> None:
    if "." in param:
        parts = param.split(".")
        for obs in scene.obstacles:
            if obs.name == parts[0] and hasattr(obs, parts[1]):
                setattr(obs, parts[1], type(getattr(obs, parts[1]))(value))
                return
        msg = f"Obstacle '{parts[0]}' not found"
        raise ValueError(msg)
    if hasattr(scene, param):
        setattr(scene, param, type(getattr(scene, param))(value))
        return
    msg = f"Unknown parameter: {param}"
    raise ValueError(msg)


def run_sweep(
    scene: Scene,
    parameter: str,
    values: list[float],
    measurements: list[str] | None = None,
    steps_per_run: int = 5000,
    progress_callback: Callable[[int, int], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> SweepResult:
    if measurements is None:
        measurements = ["reynolds_number", "drag_coefficient"]
    for m in measurements:
        if m not in _MEASUREMENT_FUNCS:
            msg = f"Unknown measurement: {m}"
            raise ValueError(msg)

    result = SweepResult(
        parameter=parameter,
        values=values,
        measurements=measurements,
        data={m: [] for m in measurements},
    )

    t0 = time.time()
    completed = 0
    cancelled = False
    for idx, val in enumerate(values):
        if should_continue is not None and not should_continue():
            cancelled = True
            break
        run_scene = copy.deepcopy(scene)
        _set_parameter(run_scene, parameter, val)

        sim = LBM2D(width=run_scene.width, height=run_scene.height)
        apply_to_sim(run_scene, sim)

        # Stepped in chunks rather than in one call so a cancel is honoured mid-run.
        # The dialog only ticks its progress bar once per value, so with the minimum
        # two values a long run would sit at 0% and refuse to stop: cancellation has
        # to be visible in the simulation loop, not only in the loop around it.
        remaining = steps_per_run
        stopped = False
        while remaining > 0:
            chunk = min(_CANCEL_CHECK_INTERVAL, remaining)
            sim.run(chunk, physics_only=True)
            remaining -= chunk
            if should_continue is not None and not should_continue():
                stopped = True
                break

        if stopped:
            # This value is only part-run, so its measurement would be meaningless. It
            # is dropped rather than plotted: a drag coefficient from half the steps
            # is not a worse answer, it is a different quantity.
            cancelled = True
            break

        for m in measurements:
            func = _MEASUREMENT_FUNCS[m]
            result.data[m].append(float(func(sim, run_scene.obstacles)))
        completed += 1

        if progress_callback:
            progress_callback(idx + 1, len(values))

    # Only values that produced a measurement stay on the axis, so the plot cannot
    # pair a measurement with the wrong x. Done once at the end against the caller's
    # list: truncating inside the loop re-truncates an already-shortened list, which
    # silently pinned the axis at its first length however many values completed.
    result.values = values[:completed]
    result.cancelled = cancelled
    result.elapsed = time.time() - t0
    return result
