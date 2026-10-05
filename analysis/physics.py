from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PressureDrop:
    """An inlet-to-outlet gauge-pressure difference, and what it was measured from.

    `value` is `mean(gauge p at the inlet column) - mean(gauge p at the outlet
    column)`, in lattice pressure units. The scatter of each column is carried
    alongside it because a mean difference smaller than the columns' own spread is
    not a measurement, and `domain_lengths` records how many obstacle diameters the
    domain is, which is what decides whether the outlet pressure has developed at
    all.
    """

    value: float
    inlet_mean: float
    outlet_mean: float
    inlet_scatter: float
    outlet_scatter: float
    domain_lengths: float

    @property
    def resolved(self) -> bool:
        """Whether the difference stands out against the columns' own scatter."""
        return abs(self.value) > (self.inlet_scatter + self.outlet_scatter)


@dataclass(frozen=True)
class WakeStrength:
    """A normalized velocity deficit across the wake, and what it was measured over.

    `value` is `1 - mean(speed in the wake region) / mean(speed upstream of the
    body)`, clamped at zero: the fraction of the incoming speed the wake takes
    out. Being a ratio, it is dimensionless and comparable between runs at
    different inlet velocities, which an absolute speed spread is not.
    """

    value: float
    upstream_mean: float
    wake_mean: float
    wake_cells: int
    upstream_cells: int
    body_span: int

    @property
    def resolved(self) -> bool:
        """Whether the deficit is large enough to be more than boundary scatter."""
        return abs(self.value) > 0.01


def wake_strength(sim: SimEngine, scene: Scene | None = None) -> WakeStrength | None:
    """Velocity deficit in the body's wake, relative to undisturbed flow upstream.

    The scorecard used `np.std(speed)` over the whole domain. Three things were
    wrong with that, in the order they matter:

    1. **It is not dimensionless.** An absolute speed spread grows with the inlet
       velocity, so the number ranks a fast run above a slow one even when the two
       wakes are identical. Measured on a 96x48 channel with the same body at two
       inlets: 0.0167 at u=0.05 and 0.0332 at u=0.10 -- a factor of two, from
       nothing but the inlet. As a *ranking* of wake strength that is close to
       meaningless.

    2. **It is a whole-domain statistic.** The inflow region and the body's own
       neighbourhood set the spread; the wake behind it is one part of a number
       dominated by the rest of the domain. It also never subtracts a baseline, so
       it cannot distinguish "strong wake" from "flow is fast here".

    3. **It was broken in 3D.** `vel[:, :, 0]` on a `(depth, height, width, 3)`
       velocity is the z=0 plane, and adding `vel[:, :, 1] ** 2` adds a second
       *component* axis: the expression produced a `(depth, width, 3)` array and
       reported 0.0667 on a flow whose actual speed spread is 0.0.

    The regions are read from the obstacle mask rather than from the scene
    geometry, so a polygon body gets the same treatment as a circle, and both
    windows use the same lateral extent so the comparison is like for like. Returns
    None when there is no body, no room downstream, or no upstream flow to divide
    by -- rather than a number with no referent.

    This is the "region-defined wake strength with an upstream baseline" of
    §6.2. It measures *how much of the flow the body takes out of the wake*, which
    is the quantity a design comparison needs; it does not attempt to measure the
    wake's width or its oscillatory content (which is what `strouhal_number` and
    the scorecard's shedding confidence are for).

    **Scope, because the number is easy to over-read.** It is an indicator for
    comparing bodies *within one configuration* — same domain, same inlet, same
    settling time. It is dimensionless, so it does not drift with the inlet the way
    an absolute spread does, but it is not inlet-invariant either: the baseline is
    the *bulk* upstream speed, and the bulk speed depends on the body (a body
    blocking 42% of the channel height makes the domain run at 0.027 against a
    prescribed inlet of 0.05). Measured on one 96x48 channel, the same r=10 body
    reads 0.00 at u=0.05 and 0.26 at u=0.10 — Re=100 where the wake is steady
    against Re=200 where it has begun to shed. A lid-driven cavity is measurable
    too, but there the "wake" is a recirculation region and the "upstream" band is
    reverse flow; that is a different comparison from a channel's and should not be
    mixed with one.
    """
    mask = _obstacle_mask(sim)
    if mask is None or not mask.any():
        return None

    try:
        speed = np.asarray(sim.speed_magnitude(), dtype=np.float64)
    except (KeyError, ValueError):
        return None

    if speed.ndim != mask.ndim or speed.shape != mask.shape:
        return None

    # Fold any leading axes into one, so the two window slices below mean the same
    # thing in 2D and 3D. They do not otherwise: for a 2D array `a[..., s0, s1]`
    # binds the slices to axes (0, 1), while for a 3D array it binds them to
    # (1, 2) and the ellipsis absorbs the depth axis. Identical text, different
    # axes -- which silently put the column window on the row axis for every 2D
    # run and produced an empty slice.
    solid = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
    speed = speed.reshape(-1, speed.shape[-2], speed.shape[-1])
    height = solid.shape[-2]
    width = solid.shape[-1]
    # Project away the leading axes as well as the in-plane one, so this reads the
    # same columns whatever the rank. `any(axis=-2)[0]` would return indices into the
    # leading axis instead, which is length 1 after the reshape above.
    cols = np.flatnonzero(solid.any(axis=-2).reshape(-1, width).any(axis=0))
    rows = np.flatnonzero(solid.any(axis=-1).reshape(-1, height).any(axis=0))
    if cols.size == 0 or rows.size == 0:
        return None

    lead, trail = int(cols[0]), int(cols[-1])
    span = max(trail - lead + 1, 1)
    # The body's *own* lateral extent, not a wider band: a window widened past the
    # body fills with the accelerated shear layers, whose mean exceeds the upstream
    # flow, and the deficit collapses to zero for a small body. Measured on a 96x48
    # channel with a radius-6 body, a band of +/-1 diameter gave upstream 0.0359 and
    # "wake" 0.0368 -- a negative deficit from averaging in the acceleration.
    row_lo = max(int(rows[0]), 0)
    row_hi = min(int(rows[-1]) + 1, height)
    row_hi = max(row_hi, row_lo + 1)

    def window(col_lo: int, col_hi: int) -> np.ndarray | None:
        col_lo = max(col_lo, 0)
        col_hi = min(col_hi, width)
        if col_hi - col_lo < 1:
            return None
        # Row slice second-to-last, column slice last: numpy binds explicit slices
        # from the right, so `[..., col, row]` would put the column window on the
        # row axis and silently read an empty or wrong block.
        return speed[..., row_lo:row_hi, col_lo:col_hi]

    # Upstream: immediately ahead of the body, same lateral extent. The band is
    # narrowed to whatever room exists ahead of the body rather than refused, so a
    # body that fills a third of the domain is still measurable; its width is
    # reported in `upstream_cells` rather than assumed.
    band = max(1, min(span, lead))
    wake_from = trail + max(span // 2, 1)
    wake = window(wake_from, wake_from + 2 * span)
    upstream = window(lead - band, lead)
    if wake is None or upstream is None or wake.size == 0 or upstream.size == 0:
        return None

    wake_mean = float(wake.mean())
    upstream_mean = float(upstream.mean())
    # The reference has to be a real flow, not a field that has not started yet. A
    # lid-driven cavity 200 steps in is still spinning up and reads 5.3e-4 against
    # a prescribed 0.05 -- 1% of the drive -- which makes the ratio a comparison of
    # two rounding errors. Measured, not guessed: that case returned a confident
    # deficit of 0.0.
    if upstream_mean < 0.05 * drive_velocity(sim):
        return None

    value = 1.0 - wake_mean / upstream_mean
    return WakeStrength(
        value=float(max(value, 0.0)),
        upstream_mean=upstream_mean,
        wake_mean=wake_mean,
        wake_cells=int(wake.size),
        upstream_cells=int(upstream.size),
        body_span=span,
    )


def _obstacle_mask(sim: SimEngine) -> np.ndarray | None:
    try:
        return np.asarray(sim.get_obstacles())
    except (AttributeError, ValueError):
        return None


def pressure_drop(sim: SimEngine, scene: Scene | None = None) -> PressureDrop | None:
    """Gauge-pressure difference across the domain, or None when it is not defined.

    This was inlined in `scorecard.py` as `abs(mean(p[:, 0]) - mean(p[:, -1]))` on
    the *absolute* pressure field. Three things were wrong with that, in ascending
    order of how much they mattered:

    1. **On a 3D engine it measured the wrong pair of faces.** `p[:, 0]` on a
       `(depth, height, width)` array is the *bottom wall plane*, not the inlet
       column, so the "pressure drop" was a top-to-bottom wall difference. The
       x axis is last in both 2D and 3D, so this indexes `[..., 0]` and `[..., -1]`.

    2. **It reported a number where no gradient exists.** A lid-driven cavity has
       no streamwise through-flow and no inlet/outlet pressure gradient by
       construction, yet it produced 1.4e-4 -- a plausible-looking figure with no
       physical referent. Refused.

    3. **It never said what it was measuring.** Returned as `float` and rendered as
       "Δp" next to Re, Cd and St, three quantities with established definitions.

    The guards are deliberately narrow -- only the cases where the quantity is
    undefined are refused:

    * no driven through-flow (`drive_velocity <= 0`);
    * an obstacle touching the inlet or outlet column, whose column mean would
      include solid cells;
    * a gauge pressure this engine cannot produce.

    What this does **not** do is make the number a real pressure-loss
    coefficient. The outlet is a *copy* of its neighbour rather than a pressure
    boundary, so the difference includes the inlet transient. Measured on a 256x128
    channel after 500 steps: 9.0e-3 with no obstacle, 1.35e-2 with one at the
    centre -- the obstacle's contribution is about a third of the reading. Use it
    to compare two runs of the *same* geometry, never as a loss coefficient.
    """
    # The *inlet*, not `drive_velocity`: a lid-driven cavity has a nonzero drive
    # velocity but no streamwise through-flow, so no inlet/outlet pressure gradient
    # can exist and any difference between the two walls is a stagnation artifact.
    if float(getattr(sim, "u_inflow", 0.0) or 0.0) <= 0:
        return None

    try:
        p = np.asarray(sim.get_pressure_gauge(), dtype=np.float64)
    except (KeyError, ValueError):
        return None

    if _touches_boundary(sim, p.shape[-1]):
        return None

    inlet, outlet = p[..., 0], p[..., -1]
    value = float(inlet.mean() - outlet.mean())

    length = characteristic_length(scene) if scene is not None else 1.0
    return PressureDrop(
        value=value,
        inlet_mean=float(inlet.mean()),
        outlet_mean=float(outlet.mean()),
        inlet_scatter=float(inlet.std()),
        outlet_scatter=float(outlet.std()),
        domain_lengths=float(p.shape[-1] / length) if length > 0 else 0.0,
    )


def _touches_boundary(sim: SimEngine, width: int) -> bool:
    """Whether any obstacle occupies the inlet or outlet column.

    Only the mask is consulted, not the scene's obstacle geometry, so this holds
    however the solid was drawn: the question is what is in the column that gets
    averaged, and a polygon clipping the edge puts solid cells there just as surely
    as a circle does.
    """
    try:
        mask = np.asarray(sim.get_obstacles())
    except (AttributeError, ValueError):
        return False
    if mask.ndim < 2 or mask.shape[-1] != width:
        return False
    return bool(mask[..., 0].any() or mask[..., -1].any())


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
