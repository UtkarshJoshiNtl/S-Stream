"""Pressure drop: promoted out of the scorecard, with the cases it cannot define.

PLAN.md C41 (P1-2). The scorecard inlined `abs(mean(p[:, 0]) - mean(p[:, -1]))` on
the *absolute* pressure field. Three problems, in ascending order of importance.

**1. On a 3D engine it measured the wrong pair of faces.** `p[:, 0]` on a
`(depth, height, width)` array is the bottom wall plane, not the inlet column — so
on every 3D backend the "pressure drop" was a top-to-bottom wall difference
labelled as a streamwise one. The x axis is last in both 2D and 3D.

**2. It printed a number where no gradient exists.** A lid-driven cavity has no
streamwise through-flow and therefore no inlet/outlet pressure gradient by
construction, yet the inlined expression returned 1.4e-4 — plausible-looking, with
no physical referent. An obstacle sitting in the inlet or outlet column is the same
problem: the column mean includes solid cells. Both are now `None`, and both
surfaces that render the value say "not measurable" instead of a number.

**3. It was returned as a bare `float`.** Re, Cd and St sit next to it in the same
row, three quantities with established definitions, while this one is an
unnormalized difference between two column means. It is now a named dataclass that
also carries each column's scatter and the domain length in obstacle diameters.

**What the surviving number is, measured.** The outlet is a *copy* of its
neighbour, not a pressure boundary, so the difference includes the inlet
transient:

    256x128 driven channel, 500 steps      |dp| = 9.03e-3    (no obstacle)
    256x128 driven channel, 500 steps      |dp| = 1.35e-2    (obstacle at centre)

The obstacle contributes about a third of the reading. That is why the outcome
panel now labels the row "Δp (proxy)" rather than "Δp", and why the function's
docstring tells callers to compare runs of the same geometry rather than read it as
a loss coefficient. A guard that would have caught this — "reject the difference
when it is smaller than the columns' scatter" — was implemented, measured, and
**rejected**: it classifies the real driven-cylinder case as unresolved
(ratio 0.44) while calling a no-obstacle channel resolved (ratio 9.8), because the
scatter is dominated by the same inlet transient. It is kept only as a reported
property (`PressureDrop.resolved`), not as a gate.
"""

import numpy as np
import pytest

from analysis.physics import pressure_drop
from analysis.scorecard import compute_scorecard
from engines.lbm2d import LBM2D
from engines.lbm3d import LBM3D
from scene.scene import Scene, apply_to_sim
from scene.serializer import load

_CHANNEL_CACHE: dict[tuple, tuple] = {}


def _channel(steps: int = 500, **kwargs) -> tuple[LBM2D, Scene]:
    """The preset cylinder, settled once and cloned per caller.

    Five call sites want the same 500-step channel and read it without writing to
    it, so each was rebuilding it (500 steps of a 256x128 domain, five times over).
    The settled engine is cached and cloned through `sim.snapshot()` (~2 ms against
    ~2.7 s for the steps); the `Scene` is immutable here and shared as-is. Handing
    out a clone rather than the cached object keeps the tests order-independent.
    """
    key = (steps, tuple(sorted(kwargs.items())))
    cached = _CHANNEL_CACHE.get(key)
    if cached is None:
        scene = load("presets/scenes/cylinder.json")
        settled = LBM2D(width=scene.width, height=scene.height, **kwargs)
        apply_to_sim(scene, settled)
        for _ in range(steps):
            settled.step()
        cached = (settled.snapshot(), scene)
        _CHANNEL_CACHE[key] = cached
    state, scene = cached
    sim = LBM2D(width=scene.width, height=scene.height, **kwargs)
    apply_to_sim(scene, sim)
    sim.restore(state)
    return sim, scene


class TestItReadsTheRightColumns:
    def test_two_dimensional_reads_the_inlet_and_outlet_columns(self):
        sim, scene = _channel()
        drop = pressure_drop(sim, scene)
        p = sim.get_pressure_gauge()
        assert drop.value == pytest.approx(p[:, 0].mean() - p[:, -1].mean())

    def test_three_dimensional_reads_the_inlet_and_outlet_columns(self):
        """The defect: `p[:, 0]` on 3D is the bottom wall plane."""
        sim = LBM3D(width=16, height=8, depth=6)
        sim.u_inflow = 0.05
        for _ in range(30):
            sim.step()
        drop = pressure_drop(sim)
        p = sim.get_pressure_gauge()
        assert p.ndim == 3
        assert drop.value == pytest.approx(p[..., 0].mean() - p[..., -1].mean())
        # The old expression averaged a y-plane and compared the two walls.
        assert drop.value != pytest.approx(p[:, 0].mean() - p[:, -1].mean())

    def test_it_uses_the_gauge_field_not_the_absolute_one(self):
        """A difference is offset-invariant: the numbers match, the field does not."""
        sim, scene = _channel()
        drop = pressure_drop(sim, scene)
        assert drop.inlet_mean == pytest.approx(sim.get_pressure_gauge()[:, 0].mean())


class TestTheGuards:
    def test_a_lid_driven_cavity_refuses(self):
        sim = LBM2D(width=48, height=48)
        sim.domain_mode = "cavity"
        sim.lid_velocity = 0.05
        sim.u_inflow = 0.0
        for _ in range(300):
            sim.step()
        # Unguarded, this expression returned 1.4e-4 for a flow with no through-flow.
        assert sim.u_inflow == 0.0
        assert pressure_drop(sim) is None

    def test_an_obstacle_in_the_outlet_column_refuses(self):
        sim = LBM2D(width=64, height=64)
        sim.add_obstacle(62, 32, radius=8)
        for _ in range(200):
            sim.step()
        assert pressure_drop(sim) is None

    def test_an_obstacle_in_the_inlet_column_refuses(self):
        sim = LBM2D(width=64, height=64)
        sim.add_obstacle(1, 32, radius=4)
        for _ in range(200):
            sim.step()
        assert pressure_drop(sim) is None

    def test_a_clear_channel_still_measures(self):
        """The guards must not swallow the case that is actually measurable."""
        sim = LBM2D(width=64, height=64)
        sim.u_inflow = 0.08
        for _ in range(300):
            sim.step()
        assert pressure_drop(sim) is not None

    def test_the_scorecard_carries_the_guard_through(self):
        sim = LBM2D(width=48, height=48)
        scene = Scene(name="cavity", width=48, height=48)
        sim.domain_mode = "cavity"
        sim.lid_velocity = 0.05
        sim.u_inflow = 0.0
        card = compute_scorecard(sim, scene, [], 400)
        assert card.pressure_drop is None


class TestWhatTheNumberIs:
    def test_it_carries_the_columns_scatter_and_the_domain_length(self):
        sim, scene = _channel()
        drop = pressure_drop(sim, scene)
        p = sim.get_pressure_gauge()
        assert drop.inlet_scatter == pytest.approx(p[:, 0].std())
        assert drop.outlet_scatter == pytest.approx(p[:, -1].std())
        assert drop.domain_lengths == pytest.approx(scene.width / 24.0)

    def test_resolved_is_reported_but_is_not_used_as_a_gate(self):
        """The rejected guard, kept as information.

        It classifies the real driven-cylinder case as unresolved (0.44) while
        calling a no-obstacle channel resolved (9.8), because the scatter is
        dominated by the same inlet transient as the difference. So `resolved` is
        reported and deliberately not used to refuse.
        """
        sim, scene = _channel()
        assert pressure_drop(sim, scene).resolved is False

    def test_the_obstacle_is_a_minor_part_of_the_reading(self):
        """The number is a proxy, and this is why."""
        clear = LBM2D(width=96, height=48)
        clear.u_inflow = 0.08
        with_body = LBM2D(width=96, height=48)
        with_body.u_inflow = 0.08
        with_body.add_obstacle(48, 24, radius=6)
        for _ in range(500):
            clear.step()
            with_body.step()
        plain = pressure_drop(clear).value
        body = pressure_drop(with_body).value
        # Both are negative (the outlet pressure sits above the inlet's), so the
        # comparison is on magnitude: adding the body deepens the difference.
        assert abs(body) > abs(plain)
        # The obstacle contributes a minority of the reading; the rest is the
        # inlet transient, which is why this is labelled a proxy and not a loss.
        assert abs(body - plain) < abs(plain)


class TestConsumersRenderTheGuard:
    def test_the_panel_says_not_measurable(self):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from workbench.panels.outcome_panel import OutcomePanel

        sim = LBM2D(width=48, height=48)
        scene = Scene(name="cavity", width=48, height=48)
        sim.domain_mode = "cavity"
        sim.lid_velocity = 0.05
        sim.u_inflow = 0.0
        panel = OutcomePanel(sim, scene)
        panel.update_outcome(400, force=True)
        assert panel.pdrop_value.text() == "not measurable"
        assert app is not None

    def test_the_panel_labels_the_row_a_proxy(self):
        import inspect

        from workbench.panels.outcome_panel import OutcomePanel

        assert "Δp (proxy)" in inspect.getsource(OutcomePanel.__init__)

    def test_the_report_does_not_format_a_none(self):
        import inspect

        from export import report

        assert "not measurable for this configuration" in inspect.getsource(report)

    def test_the_scorecard_to_dict_keeps_the_guard(self):
        sim = LBM2D(width=48, height=48)
        scene = Scene(name="cavity", width=48, height=48)
        sim.domain_mode = "cavity"
        sim.lid_velocity = 0.05
        sim.u_inflow = 0.0
        assert compute_scorecard(sim, scene, [], 400).to_dict()["pressure_drop"] is None

    def test_the_function_is_not_inlined_in_the_scorecard_anymore(self):
        import inspect

        from analysis import scorecard

        source = inspect.getsource(scorecard.compute_scorecard)
        assert "get_pressure()" not in source
        assert "np.mean(p[" not in source
        assert "pressure_drop(sim, scene)" in source

    def test_the_value_is_finite_for_a_measured_case(self):
        sim, scene = _channel()
        drop = pressure_drop(sim, scene)
        assert np.isfinite(drop.value)
        assert isinstance(drop.value, float)
