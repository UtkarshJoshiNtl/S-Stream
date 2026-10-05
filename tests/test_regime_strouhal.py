"""The regime classifier's "Strouhal number" was not a Strouhal number.

PLAN.md C40 (P1-1), following C39. `analysis/regimes._probe_oscillation` called
`strouhal_number(v_history, dt=1.0)` and left `diameter` and `velocity` at their
defaults of 1.0. With D = U = 1 the expression `f * D / U` collapses to `f` -- the
raw oscillation frequency in cycles per step. Three consumers then treated that
number as a Strouhal number:

    workbench/panels/analysis_panel.py   f * D / U   correct
    analysis/regimes.py                  f * 1 / 1   = f
    analysis/scorecard.py                compares against expected St ranges

The disagreement between the two panels is a factor of D/U. For the cylinder
preset that is 24 / 0.08 = 300. And because `scorecard` compares the value against
that preset's `expected_ranges["St"] = [0.12, 0.28]`, the check **could not pass
for any scene at any setting** -- a guaranteed-fail verdict reported next to a
regime the classifier was confidently calling "Periodic vortex shedding".

Measured on `presets/scenes/cylinder.json` at step 4000, identical probe data:

    regime.strouhal        0.00977      <- cycles per step
    scorecard St verdict   pass=False   against [0.12, 0.28]
    panel's St             2.93         <- f * D / U

**What this fix does and does not do.** It makes all three consumers agree on the
quantity, which is the item's scope. It does *not* make the scorecard's St check
pass: 2.93 is still ~18x the literature value for a Re=300 cylinder, because the
FFT window (256 samples = 512 steps at cadence 2) is far shorter than the shedding
period. That is a separate defect, flagged in C39 and left visible here: the check
moves from *wrong for definitional reasons* to *wrong for a stated reason*. The
test asserts the agreement and pins the residual gap, so the remaining fix is
measurable rather than a matter of opinion.
"""

import numpy as np
import pytest

from analysis.physics import characteristic_length, drive_velocity, strouhal_number
from analysis.regimes import detect_flow_regime
from analysis.scorecard import compute_scorecard
from engines.lbm2d import LBM2D
from scene.probe import Probe
from scene.scene import ProbeSpec, apply_to_sim
from scene.serializer import load

CADENCE = 2
STEPS = 1200  # long enough for a real oscillatory trace, short enough to be quick


@pytest.fixture(scope="module")
def cylinder():
    """The cylinder preset, driven and sampled at the panel's real cadence."""
    scene = load("presets/scenes/cylinder.json")
    sim = LBM2D(width=96, height=48)
    apply_to_sim(scene, sim)
    sim.u_inflow = 0.08
    probe = Probe(scene.probes[0])
    for step in range(1, STEPS + 1):
        sim.step()
        if step % CADENCE == 0:
            probe.record(sim, None, step=step)
    return scene, sim, probe


class TestAllThreeConsumersAgreeOnTheQuantity:
    def test_regime_strouhal_matches_the_panels_call(self, cylinder):
        scene, sim, probe = cylinder
        D, U = characteristic_length(scene), drive_velocity(sim)
        regime = detect_flow_regime(sim, scene, [probe], STEPS)
        panel_value = strouhal_number(
            list(probe.history["v"]), diameter=D, velocity=U, times=probe.times()
        )
        assert regime.strouhal == pytest.approx(panel_value, rel=1e-9)

    def test_the_value_is_not_the_bare_frequency_any_more(self, cylinder):
        scene, sim, probe = cylinder
        D, U = characteristic_length(scene), drive_velocity(sim)
        regime = detect_flow_regime(sim, scene, [probe], STEPS)
        # f = St * U / D. Before the fix the classifier returned f itself.
        assert regime.strouhal != pytest.approx(regime.strouhal * U / D, rel=1e-6)
        assert regime.strouhal == pytest.approx(
            regime.strouhal * D / U * U / D, rel=1e-6
        )

    def test_the_default_arguments_would_be_wrong_for_this_scene(self, cylinder):
        """The regression is scene-dependent, so pin the scene that triggers it."""
        scene, sim, probe = cylinder
        D, U = characteristic_length(scene), drive_velocity(sim)
        with_defaults = strouhal_number(list(probe.history["v"]), times=probe.times())
        with_reals = strouhal_number(
            list(probe.history["v"]), diameter=D, velocity=U, times=probe.times()
        )
        assert with_defaults != pytest.approx(with_reals, rel=0.01)
        # The factor is exactly D/U, which is what the two panels disagreed by.
        assert with_defaults * D / U == pytest.approx(with_reals, rel=0.01)


class TestTheScorecardCheckIsNoLongerGuaranteedToFail:
    def test_it_compares_a_dimensional_st(self, cylinder):
        scene, sim, probe = cylinder
        card = compute_scorecard(sim, scene, [probe], STEPS)
        check = card.range_checks["St"]
        assert check["value"] == pytest.approx(
            detect_flow_regime(sim, scene, [probe], STEPS).strouhal
        )
        # D/U = 300 for this preset, so the value is now 300x what it was.
        assert check["value"] > 0.1

    def test_the_residual_gap_is_the_window_not_the_units(self, cylinder):
        """Still ~18x the literature St for Re=300, for the stated reason.

        256 samples at cadence 2 is 512 steps, against a shedding period of roughly
        D/(St*U) = 24/(0.16*0.08) = 1875 steps. The FFT is resolving a fraction of a
        cycle and picking a harmonic. Pinned so that lengthening the window later
        is measurable.
        """
        scene, sim, probe = cylinder
        D, U = characteristic_length(scene), drive_velocity(sim)
        value = strouhal_number(
            list(probe.history["v"]), diameter=D, velocity=U, times=probe.times()
        )
        expected_period_steps = D / (0.16 * U)
        window_steps = 256 * CADENCE
        assert window_steps < expected_period_steps
        assert value > 0.16 * 5


class TestTheVelocityHasOneDefinition:
    def test_regimes_uses_drive_velocity(self):
        import inspect

        from analysis import regimes

        assert "velocity=drive_velocity(sim)" in inspect.getsource(
            regimes.detect_flow_regime
        )

    def test_the_panel_uses_drive_velocity_not_the_inlet(self):
        import inspect

        from workbench.panels.analysis_panel import AnalysisPanel

        source = inspect.getsource(AnalysisPanel._update_physics)
        assert "drive_velocity(self.sim)" in source
        assert "self.sim.u_inflow" not in source

    def test_a_lid_driven_cavity_is_not_reported_as_having_no_flow(self):
        """`drive_velocity` exists for exactly this; both panels must use it."""
        sim = LBM2D(width=32, height=32)
        sim.u_inflow = 0.0
        sim.domain_mode = "cavity"
        sim.lid_velocity = 0.05
        assert drive_velocity(sim) == 0.05
        assert sim.u_inflow == 0.0


class TestTheFunctionTakesItsInputs:
    def test_diameter_and_velocity_reach_strouhal_number(self):
        import inspect

        from analysis import regimes

        assert "diameter=diameter" in inspect.getsource(regimes._probe_oscillation)
        assert "velocity=velocity" in inspect.getsource(regimes._probe_oscillation)

    def test_no_probe_still_yields_nothing(self):
        from analysis.regimes import _probe_oscillation

        assert _probe_oscillation([], diameter=10.0, velocity=0.1) == (0.0, None)

    def test_a_short_trace_yields_no_strouhal(self):
        from analysis.regimes import _probe_oscillation

        sim = LBM2D(width=16, height=16)
        probe = Probe(ProbeSpec(name="p", x=8, y=8, fields=["v"]))
        probe.record(sim, None, step=1)
        amplitude, st = _probe_oscillation([probe], diameter=10.0, velocity=0.1)
        assert st is None
        assert amplitude == pytest.approx(0.0)

    def test_the_unavailable_field_default_does_not_raise(self):
        """An engine with no usable probe must not break classification."""
        scene = load("presets/scenes/cylinder.json")
        sim = LBM2D(width=48, height=24)
        apply_to_sim(scene, sim)
        sim.u_inflow = 0.05
        regime = detect_flow_regime(sim, scene, [], STEPS)
        assert regime.strouhal is None
        assert isinstance(regime.label, str) and regime.label

    def test_the_numpy_value_is_a_float(self, cylinder):
        _, sim, probe = cylinder
        value = detect_flow_regime(sim, cylinder[0], [probe], STEPS).strouhal
        assert np.isscalar(value)
        assert np.isfinite(value)
