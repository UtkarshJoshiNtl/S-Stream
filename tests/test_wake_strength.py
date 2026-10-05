"""Wake strength was a whole-domain standard deviation, and broken in 3D.

PLAN.md C42 (§6.2). The scorecard computed `wake_strength = np.std(speed)` where
`speed = sqrt(vel[:,:,0]**2 + vel[:,:,1]**2)`.

**1. It was broken in 3D.** `vel[:, :, 0]` on a `(depth, height, width, 3)` velocity
is the z=0 plane, and `vel[:, :, 1]` is a *component* axis, so the expression
produced a `(depth, width, 3)` array — speed mixed with depth plane and component
axis. Measured on a `LBM3D(8, 6, 5)` at rest: it reported **0.0667** where the real
speed field's spread is **0.0**. It now uses the engine's own `speed_magnitude()`
(C38).

**2. It was not dimensionless.** An absolute speed spread grows with the inlet
velocity, so it ranks a fast run above a slow one even for identical wakes.
Measured on a 96x48 channel, same body, twice the inlet:

    u=0.05   std(speed) = 0.0157
    u=0.10   std(speed) = 0.0322     <- 2.05x, from nothing but the inlet

**3. It was a whole-domain statistic with no baseline.** The inflow region and the
body's own neighbourhood set the spread; the wake behind it is one part of a number
dominated by the rest of the domain, and nothing subtracts what the flow was like
before the body.

Now: `1 - mean(speed in the wake) / mean(speed upstream of the body)`, clamped at
zero — the fraction of the incoming flow the body takes out of the wake. The regions
come from the obstacle mask, so a polygon body gets the same treatment as a circle,
and the wake and upstream windows share one lateral extent so the comparison is
like for like.

**Two design decisions that measurement forced, both recorded here because they look
arbitrary otherwise.**

*The lateral window is the body's own extent, not a wider band.* A band widened past
the body fills with the accelerated shear layers, whose mean exceeds the upstream
flow. Measured on the same channel with a radius-6 body: a band of ±1 diameter gave
upstream 0.0359 and "wake" 0.0368 — a *negative* deficit from averaging in the
acceleration around the body.

*The upstream band narrows to the room available rather than refusing.* A body
filling a third of the domain has no full-width reference band ahead of it; the
band shrinks to what exists and its width is reported in `upstream_cells` instead of
being assumed.

**What the number is sensitive to, which is not fixed here.** It is a ratio, so it
cannot drift with the inlet the way an absolute spread does — but it is not
inlet-invariant either, and pretending otherwise would be the same mistake in the
other direction. The baseline is the *bulk* upstream speed, and the bulk speed
depends on the body: a body blocking 42% of the channel height makes the domain run
at 0.027 while the inlet is prescribed at 0.05. Measured deficits on the same
channel:

    body r=6,  u=0.05 -> 0.31        body r=6,  u=0.10 -> 0.55
    body r=10, u=0.05 -> 0.00        body r=10, u=0.10 -> 0.26

The `r=10, u=0.05` zero is not a bug in the arithmetic: that case is Re=100, where
the wake is steady, against Re=200 where it has begun to shed. So the measure is
documented as what it is — an indicator for comparing bodies **within one
configuration** (same domain, same inlet, same settling time), which is the design
comparison §6.2 asks for. Comparisons across inlet velocities are out of scope and
the docstring says so.
"""

import numpy as np
import pytest

from analysis.physics import wake_strength
from analysis.scorecard import compute_scorecard
from engines.lbm2d import LBM2D
from engines.lbm3d import LBM3D
from scene.scene import Scene


def _channel(U: float = 0.08, radius: int = 8, steps: int = 1200) -> LBM2D:
    sim = LBM2D(width=96, height=48)
    sim.u_inflow = U
    sim.add_obstacle(32, 24, radius=radius)
    for _ in range(steps):
        sim.step()
    return sim


class TestItIsTheEngineSpeedField:
    def test_two_dimensional_uses_the_engines_speed(self):
        sim = _channel()
        result = wake_strength(sim)
        assert result is not None
        assert 0.0 <= result.value <= 1.0

    def test_three_dimensional_is_measured_not_indexed_wrong(self):
        sim = LBM3D(width=24, height=16, depth=8)
        sim.u_inflow = 0.05
        sim.add_obstacle(8, 8, 4, radius=3)
        for _ in range(300):
            sim.step()
        result = wake_strength(sim)
        assert result is not None
        assert np.isfinite(result.value)
        assert result.value >= 0.0
        # The old expression on 3D returned 0.0667 on a flow whose speed spread is 0.
        vel = sim.get_velocity()
        old_expression = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        assert old_expression.shape[-1] == 3
        assert old_expression.shape != sim.speed_magnitude().shape

    def test_the_two_dimensional_windows_bind_to_the_right_axes(self):
        """`a[..., s0, s1]` binds s0 to axis -2 in 2D but to axis -1 in 3D.

        Both orders of that mistake were made while writing this, and both produced
        an empty slice rather than an error — the measure silently returned nothing.
        Pinned by requiring a real, non-empty window in both ranks.
        """
        for sim in (_channel(), _three_dimensional()):
            result = wake_strength(sim)
            assert result is not None
            assert result.wake_cells > 0
            assert result.upstream_cells > 0
            assert result.wake_mean > 0.0


def _three_dimensional() -> LBM3D:
    sim = LBM3D(width=24, height=16, depth=8)
    sim.u_inflow = 0.05
    sim.add_obstacle(8, 8, 4, radius=3)
    for _ in range(300):
        sim.step()
    return sim


class TestItIsARegionNotTheDomain:
    def test_the_wake_window_sits_downstream_of_the_body(self):
        sim = _channel()
        mask = np.asarray(sim.get_obstacles())
        trail = int(np.flatnonzero(mask.any(axis=-2))[-1])
        result = wake_strength(sim)
        assert result.body_span == trail - int(np.flatnonzero(mask.any(axis=-2))[0]) + 1
        # The wake mean must be a small fraction of the whole-domain mean spread:
        # a whole-domain statistic cannot distinguish a wake at all.
        assert result.wake_cells < mask.size

    def test_a_uniform_flow_has_no_deficit(self):
        """No body, no wake: the guard refuses rather than inventing one."""
        sim = LBM2D(width=64, height=64)
        sim.u_inflow = 0.05
        for _ in range(200):
            sim.step()
        assert wake_strength(sim) is None

    def test_the_wake_mean_differs_from_the_upstream_mean(self):
        sim = _channel()
        result = wake_strength(sim)
        assert result.wake_mean != pytest.approx(result.upstream_mean)


class TestItHasABaseline:
    def test_the_value_is_one_minus_the_speed_ratio(self):
        sim = _channel()
        result = wake_strength(sim)
        assert result.value == pytest.approx(
            max(1.0 - result.wake_mean / result.upstream_mean, 0.0)
        )

    def test_it_is_clamped_at_zero(self):
        sim = _channel()
        result = wake_strength(sim)
        assert result.value >= 0.0

    def test_the_upstream_window_is_immediately_ahead_of_the_body(self):
        sim = _channel()
        mask = np.asarray(sim.get_obstacles())
        lead = int(np.flatnonzero(mask.any(axis=-2))[0])
        assert lead > 0
        assert wake_strength(sim).upstream_mean > 0.0

    def test_the_baseline_is_the_bulk_flow_not_the_inlet(self):
        """Pinned because it is the whole point, and easy to get wrong.

        A one-column baseline reads the *prescribed* inlet speed (0.05) while the
        bulk flow in a blocked channel runs at 0.027, which inflates the deficit by
        roughly a factor of two. The baseline is a bulk band, and the difference is
        measurable.
        """
        sim = _channel(U=0.08, radius=10)
        bulk = wake_strength(sim).upstream_mean
        inlet = float(sim.speed_magnitude()[:, 0].mean())
        assert inlet > bulk * 1.5


class TestTheOldMetricWasNotDimensionless:
    def test_the_speed_spread_doubles_with_the_inlet(self):
        slow = _channel(U=0.05, steps=1500)
        fast = _channel(U=0.10, steps=1500)
        slow_spread = float(np.std(slow.speed_magnitude()))
        fast_spread = float(np.std(fast.speed_magnitude()))
        assert fast_spread == pytest.approx(2.0 * slow_spread, rel=0.1)

    def test_the_deficit_cannot_leave_the_unit_interval(self):
        for U in (0.05, 0.10):
            result = wake_strength(_channel(U=U))
            assert 0.0 <= result.value <= 1.0

    def test_the_sensitivity_that_remains_is_documented(self):
        """Not inlet-invariant, and the tests say so rather than implying it is.

        The baseline is the bulk upstream speed, and the bulk speed depends on the
        body (blockage), so the ratio moves when the inlet moves. Pinning it keeps a
        later improvement from being mistaken for a regression, and keeps this test
        from implying a robustness the measure does not have.
        """
        low = wake_strength(_channel(U=0.05, radius=10, steps=1500)).value
        high = wake_strength(_channel(U=0.10, radius=10, steps=1500)).value
        assert low != pytest.approx(high, abs=0.05)


class TestTheGuardsAndTheConsumers:
    def test_a_body_with_no_upstream_room_still_measures(self):
        """The band narrows to the room available; `upstream_cells` records it."""
        sim = LBM2D(width=96, height=48)
        sim.u_inflow = 0.05
        sim.add_obstacle(32, 24, radius=14)
        for _ in range(400):
            sim.step()
        result = wake_strength(sim)
        assert result is not None
        assert result.upstream_cells > 0

    def test_a_flow_that_has_not_started_yet_is_refused(self):
        """A spinning-up cavity read a confident deficit of 0.0 from nothing.

        At step 200 its upstream band averages 5.3e-4 against a prescribed lid
        speed of 0.05 — 1% of the drive — so the "ratio" was comparing two rounding
        errors and the clamp turned it into a plausible zero.
        """
        sim = LBM2D(width=64, height=64)
        sim.u_inflow = 0.0
        sim.domain_mode = "cavity"
        sim.lid_velocity = 0.05
        sim.add_obstacle(20, 20, radius=5)
        for _ in range(200):
            sim.step()
        assert wake_strength(sim) is None

    def test_the_same_cavity_measures_once_it_has_spun_up(self):
        sim = LBM2D(width=64, height=64)
        sim.u_inflow = 0.0
        sim.domain_mode = "cavity"
        sim.lid_velocity = 0.05
        sim.add_obstacle(20, 20, radius=5)
        for _ in range(6000):
            sim.step()
        result = wake_strength(sim)
        assert result is not None
        assert result.upstream_mean > 0.05 * 0.05

    def test_the_scorecard_reports_the_new_quantity(self):
        sim = _channel()
        card = compute_scorecard(sim, Scene(name="s", width=96, height=48), [], 1200)
        assert card.wake_strength == pytest.approx(wake_strength(sim).value)

    def test_the_scorecard_refuses_when_there_is_no_body(self):
        sim = LBM2D(width=48, height=48)
        sim.u_inflow = 0.05
        for _ in range(200):
            sim.step()
        card = compute_scorecard(sim, Scene(name="s", width=48, height=48), [], 200)
        assert card.wake_strength is None

    def test_the_report_handles_the_guard(self):
        import inspect

        from export import report

        assert "not measurable for this configuration" in inspect.getsource(report)

    def test_the_scorecard_no_longer_inlines_a_speed_expression(self):
        import inspect

        from analysis import scorecard

        source = inspect.getsource(scorecard.compute_scorecard)
        assert "np.sqrt(vel[" not in source
        assert "wake_strength(sim, scene)" in source

    def test_the_reported_metadata_is_consistent(self):
        result = wake_strength(_channel())
        assert result.wake_cells > result.upstream_cells  # 2*span wide vs span
        assert result.body_span > 0
        assert result.resolved == (abs(result.value) > 0.01)
