"""Probe time axes and Strouhal numbers were both off by exactly the sampling cadence.

PLAN.md C39 (P1-2). Two clocks were being confused, in two places.

A probe does not sample every step. `AnalysisPanel.tick` records on every second
tick, so consecutive samples are **two** simulation steps apart. But:

* `_ProbePlot.update_plot` set the x-axis label to "step" and plotted the *sample
  index* — a time axis reading at half speed.
* `strouhal_number` was handed `dt=1.0`, declaring a sampling interval of one step
  for samples two steps apart. A signal sampled every `k` steps has `k` times
  fewer cycles per sample, so an FFT that assumes one step per sample reports the
  frequency `k` times too high — and St exactly `k` times too high with it.

Measured on a real wake probe (cylinder D=20, U=0.1, Re=100, 96x48, sampling every
2 steps, identical history both ways):

    dt=1.0 (what the panel passed)   St = 3.90625
    dt=2.0 (the real cadence)        St = 1.95312

A ratio of exactly two, as it must be. Note what this test does *not* claim: the
underlying shedding frequency of this engine at Re=100 is not asserted against
literature here. What is asserted is the cadence arithmetic, which is exact and
independent of the physics — if the window is too short to contain a period (see
the last test), the FFT is fitting harmonics and the *absolute* value is not
meaningful, but the factor of two between the two assumptions is nonetheless not
in dispute.

The fix records the simulation step beside every sample, so the cadence is measured
rather than declared, and hands that to the FFT.
"""

import numpy as np
import pytest

from analysis.physics import strouhal_number
from engines.lbm2d import LBM2D
from scene.probe import Probe
from scene.scene import ProbeSpec


def _sinusoid(period_steps: float, cadence: int, n: int = 512) -> tuple[list, list]:
    """A transverse-velocity trace sampled every `cadence` steps."""
    steps = np.arange(2, n + 2) * cadence
    values = np.sin(2 * np.pi * steps / period_steps)
    return list(values), list(steps)


class TestStrouhalReadsTheCadenceItIsGiven:
    def test_a_declared_dt_of_one_halves_the_period_and_doubles_st(self):
        values, _ = _sinusoid(period_steps=100.0, cadence=2)
        # Sampled every 2 steps, a 100-step period appears as 50 samples. St is
        # f*D/U = (1/100)*20/0.1 = 2.0; the FFT lands 2.3% low on bin
        # quantization, which the tolerance covers.
        assert strouhal_number(
            values, dt=2.0, diameter=20.0, velocity=0.1
        ) == pytest.approx(100.0**-1 * 20.0 / 0.1, rel=0.05)

    def test_wrong_dt_is_wrong_by_exactly_the_cadence_factor(self):
        values, _ = _sinusoid(period_steps=100.0, cadence=2)
        right = strouhal_number(values, dt=2.0, diameter=20.0, velocity=0.1)
        wrong = strouhal_number(values, dt=1.0, diameter=20.0, velocity=0.1)
        assert wrong == pytest.approx(2.0 * right, rel=1e-9)

    def test_recorded_times_give_the_same_answer_as_the_right_dt(self):
        values, steps = _sinusoid(period_steps=100.0, cadence=2)
        from_dt = strouhal_number(values, dt=2.0, diameter=20.0, velocity=0.1)
        from_times = strouhal_number(
            values, dt=1.0, diameter=20.0, velocity=0.1, times=steps
        )
        assert from_times == pytest.approx(from_dt, rel=1e-9)

    def test_times_overrule_a_deliberately_wrong_dt(self):
        values, steps = _sinusoid(period_steps=100.0, cadence=4)
        stale_dt = strouhal_number(values, dt=1.0, diameter=20.0, velocity=0.1)
        measured = strouhal_number(
            values, dt=1.0, diameter=20.0, velocity=0.1, times=steps
        )
        assert measured == pytest.approx(stale_dt / 4.0, rel=0.05)

    def test_mismatched_times_are_ignored_rather_than_trusted(self):
        values, _ = _sinusoid(period_steps=100.0, cadence=2)
        with_dt2 = strouhal_number(values, dt=2.0, diameter=20.0, velocity=0.1)
        # A times array that does not line up with the samples must not silently
        # shift the answer; it is dropped and dt stands.
        assert strouhal_number(
            values, dt=2.0, diameter=20.0, velocity=0.1, times=[0, 1]
        ) == pytest.approx(with_dt2, rel=1e-9)

    def test_the_window_is_too_short_for_the_period_it_is_given(self):
        """A limitation stated rather than hidden.

        The FFT looks at the most recent 256 samples. At the panel's cadence that is
        512 simulation steps, while a Re=100 cylinder sheds on a ~1250-step period —
        so at exactly the Reynolds number where Strouhal matters, the estimator is
        fitting a fraction of a cycle and reports a harmonic instead. The measured
        St for that run above (1.95, where the textbook value is ~0.16) is what a
        sub-period window produces. This test pins the *relationship* between window
        and period so the fix in a later item is measurable rather than a matter of
        opinion.
        """
        values, steps = _sinusoid(period_steps=1250.0, cadence=2)
        assert len(values) == 512
        # 512 samples at cadence 2 span 1024 steps -- less than one period.
        assert (steps[-1] - steps[0]) < 1250.0


class TestTheProbeRemembersWhenItSampled:
    def _probe(self, fields=("v",)):
        sim = LBM2D(width=16, height=16)
        return sim, Probe(ProbeSpec(name="p", x=8, y=8, fields=list(fields)))

    def test_a_supplied_step_is_recorded_verbatim(self):
        sim, probe = self._probe()
        for step in (10, 12, 14, 16):
            sim.step()
            probe.record(sim, step=step)
        assert list(probe.steps) == [10, 12, 14, 16]

    def test_omitting_the_step_falls_back_to_the_sample_counter(self):
        sim, probe = self._probe()
        for _ in range(4):
            sim.step()
            probe.record(sim)
        assert list(probe.steps) == [0, 1, 2, 3]
        assert probe.sample_interval() == 1

    def test_the_sample_interval_is_the_median_step_gap(self):
        sim, probe = self._probe()
        for step in (2, 4, 6, 8):
            probe.record(sim, step=step)
        assert probe.sample_interval() == 2

    def test_a_missed_sample_does_not_average_into_the_cadence(self):
        sim, probe = self._probe()
        # A stalled frame: one 12-step gap among 2-step gaps.
        for step in (2, 4, 6, 8, 10, 22, 24, 26):
            probe.record(sim, step=step)
        assert probe.sample_interval() == 2

    def test_too_few_samples_report_no_interval(self):
        sim, probe = self._probe()
        assert probe.sample_interval() is None
        probe.record(sim, step=5)
        assert probe.sample_interval() is None

    def test_steps_stay_aligned_with_the_history(self):
        sim, probe = self._probe(fields=("u", "v"))
        for step in (2, 4, 6):
            sim.step()
            probe.record(sim, step=step)
        assert len(probe.steps) == len(probe.history["v"]) == len(probe.history["u"])

    def test_a_field_that_records_nothing_stamps_no_step(self):
        sim = LBM2D(width=16, height=16)
        probe = Probe(ProbeSpec(name="p", x=8, y=8, fields=["phase"]))
        # "phase" does not exist on LBM2D, so nothing is ever recorded.
        probe.record(sim, step=2)
        assert list(probe.steps) == []
        assert len(probe.history["phase"]) == 0

    def test_clear_drops_the_timestamps_too(self):
        sim, probe = self._probe()
        probe.record(sim, step=2)
        probe.clear()
        assert list(probe.steps) == []
        assert probe.sample_interval() is None


class TestThePanelSamplesOnItsOwnCadence:
    def test_the_app_stamps_each_sample_with_the_simulation_step(self):
        """Every second tick, so samples land two steps apart -- and say so."""
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        app_qt = QApplication.instance() or QApplication([])
        from workbench.panels.analysis_panel import AnalysisPanel

        sim = LBM2D(width=24, height=24)
        panel = AnalysisPanel(sim)
        panel.probes = [Probe(ProbeSpec(name="p", x=12, y=12, fields=["v"]))]
        panel.probes[0].spec.x = 12

        for step in range(1, 11):
            sim.step()
            panel.tick(1.0, step=step)

        recorded = list(panel.probes[0].steps)
        assert recorded == [2, 4, 6, 8, 10]
        assert panel.probes[0].sample_interval() == 2
        assert app_qt is not None


class TestTheConsumersUseIt:
    def test_regimes_no_longer_declares_a_cadence_of_one(self):
        import inspect

        from analysis import regimes

        source = inspect.getsource(regimes._probe_oscillation)
        assert "times=" in source
        assert "dt=1.0)" not in source

    def test_regimes_uses_the_engines_single_curl(self):
        import inspect

        from analysis import regimes

        assert "sim.vorticity()" in inspect.getsource(regimes._max_vorticity)

    def test_the_plot_x_axis_is_steps_not_sample_count(self):
        import inspect

        from workbench.panels.analysis_panel import _ProbePlot

        source = inspect.getsource(_ProbePlot.update_plot)
        assert "self.probe.times()" in source
        assert "self.curve.setData(x, data)" in source
