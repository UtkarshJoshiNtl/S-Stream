from __future__ import annotations

import numpy as np

from engines.base import SimEngine
from engines.lbm_common import LATTICE_2D
from scene.scene import CircleObstacle, RectObstacle, Scene


def characteristic_length(scene: Scene) -> float:
    for obs in scene.obstacles:
        if isinstance(obs, CircleObstacle):
            return float(obs.radius * 2)
        if isinstance(obs, RectObstacle):
            return float(max(obs.w, obs.h))
    return float(scene.width)


def drive_velocity(sim: SimEngine) -> float:
    """Resolve the velocity that actually drives the flow.

    ``reynolds_number`` and ``drag_coefficient`` already know that a lid-driven
    cavity is driven by ``lid_velocity`` rather than by the inlet. Regime
    classification needs the same resolution, otherwise a cavity preset is
    reported as having no driven flow at all.
    """
    inlet = float(getattr(sim, "u_inflow", 0.0) or 0.0)
    if inlet > 0:
        return inlet
    if getattr(sim, "domain_mode", None) == "cavity":
        return float(getattr(sim, "lid_velocity", 0.0) or 0.0)
    return inlet


def reynolds_number(sim: SimEngine, obstacle_diameter: float | None = None) -> float:
    L = obstacle_diameter or float(sim.grid_shape[1])
    U = float(sim.u_inflow)
    if getattr(sim, "domain_mode", None) == "cavity":
        U = float(getattr(sim, "lid_velocity", 0.0) or 0.0)
        if obstacle_diameter is None:
            L = float(sim.grid_shape[0])
    nu = sim.viscosity
    if nu <= 0 or U <= 0:
        return 0.0
    return U * L / nu


def drag_coefficient(sim: SimEngine, diameter: float | None = None) -> float:
    """Momentum-exchange drag coefficient for 2D obstacles.

    For each fluid→solid lattice link along direction ``i``, the force
    contribution is ``2 * f_i * c_i`` (stationary halfway/fullway BB).
    ``Cd = 2 |Fx| / (ρ U² D)``.
    """
    lattice = LATTICE_2D
    obs = sim.get_obstacles_mut()
    if not np.any(obs):
        return 0.0

    f = sim.get_f()
    height, width = obs.shape
    Fx = 0.0
    for i in range(1, 9):
        cx = int(lattice.cx[i])
        cy = int(lattice.cy[i])
        ys, xs = np.where(~obs)
        if len(ys) == 0:
            continue
        nx = xs + cx
        ny = ys + cy
        valid = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        ny_c = np.clip(ny, 0, height - 1)
        nx_c = np.clip(nx, 0, width - 1)
        neighbor_obs = obs[ny_c, nx_c]
        hit = valid & neighbor_obs
        Fx += float(np.sum(f[i, ys[hit], xs[hit]]) * cx)

    U = float(sim.u_inflow)
    if getattr(sim, "domain_mode", None) == "cavity":
        U = float(getattr(sim, "lid_velocity", 0.0) or 0.0)
    rho0 = 1.0
    if diameter is None:
        A = float(np.sum(obs))
        D = 2.0 * np.sqrt(A / np.pi) if A > 0 else 1.0
    else:
        D = float(diameter)
    if U <= 0 or D <= 0:
        return 0.0
    Cd = 2.0 * abs(Fx) / (rho0 * U * U * D)
    return float(Cd)


def strouhal_number(
    v_history: list[float],
    dt: float = 1.0,
    diameter: float = 1.0,
    velocity: float = 1.0,
    times: list[float] | None = None,
) -> float | None:
    """Strouhal number of a transverse-velocity trace, St = f * D / U.

    `dt` is the number of simulation steps *between consecutive samples*, and it is
    not a free parameter. A probe in the analysis panel samples on every second
    tick, so consecutive samples are two steps apart; passing dt=1 there declares a
    signal that oscillates twice as fast as it does and reports St exactly twice as
    high. `times` is the recorded step index per sample (see `Probe.times`), which
    removes the assumption entirely -- the cadence is then measured rather than
    declared.

    The FFT assumes uniform sampling, which the median gap approximates: a probe
    that missed samples would report the typical interval, not every one.
    """
    if len(v_history) < 32:
        return None
    if times is not None and len(times) == len(v_history) and len(times) >= 2:
        gaps = np.diff(np.asarray(times, dtype=float))
        measured = float(np.median(gaps))
        if measured > 0:
            dt = measured
    recent = list(v_history)[-256:]
    arr = np.array(recent)
    arr = arr - np.mean(arr)
    n = len(arr)
    if n < 4:
        return None
    window = np.hanning(n)
    fft = np.fft.rfft(arr * window)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(fft)
    low = max(1, int(0.02 * n))
    mag[:low] = 0.0
    peak_idx = int(np.argmax(mag))
    peak_mag = mag[peak_idx]
    total = float(np.sum(mag))
    if total < 1e-10 or peak_mag / total < 0.1:
        return None
    f_peak = freqs[peak_idx]
    if velocity <= 0:
        return None
    return f_peak * diameter / velocity
