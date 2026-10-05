"""The Strouhal number must mean the same thing in every consumer.

PLAN.md C43 (P1-2). C40 made the panel, the regime classifier and the scorecard agree
on the *quantity*; this file covers the consumers that were not in that list -- the
report, the outcome panel, the probe CSV, and the scoring rule itself -- and the two
properties that make the agreement meaningful rather than merely uniform.

What C40 already pinned: panel == regime == scorecard, and the value is dimensional
(`f * D / U`) rather than a bare frequency. What it did not pin, and what is easy to
get wrong, is whether those consumers agree on something *correct*. Agreement between
three functions that are all wrong in the same way is not consistency, it is a
convention -- which is exactly how the window defect survived: panel, regime and
scorecard all reported 2.93 on the cylinder preset, all consistently, ~18x the
literature St of 0.16 for a Re=300 cylinder. So the tests below come in two kinds:

1. Agreement, extended to the consumers C40 did not cover.
2. Properties that hold regardless of implementation -- `St` scales as `D/U`, and
   the time axis a consumer publishes is the time axis the simulation ran on.

**The exporter defect this found.** `export_probe_csv` wrote the sample index into a
column headed `step`. A panel-cadence probe records steps 2, 4, 6 ... 40, and the
export wrote `0, 1, 2 ... 19`:

    recorded steps : [2, 4, 6, 8, ..., 40]   cadence 2
    csv step column: ['0', '1', '2', ..., '19']

Every exported figure therefore had a time axis halved, offset to zero, and
unlabelled as to what it was -- and the file could not reproduce the Strouhal number
the report prints a few lines above it. C39 gave probes real timestamps; this was
the last consumer still reading indices.
"""

import ast
import inspect

import numpy as np
import pytest

from analysis.physics import strouhal_number
from analysis.regimes import detect_flow_regime
from analysis.scorecard import compute_scorecard
from engines.lbm2d import LBM2D
from export.data import export_probe_csv
from export.report import export_markdown_report
from scene.probe import Probe
from scene.scene import ProbeSpec
from scene.serializer import load

CADENCE = 2
STEPS = 1200


def _oscillatory_trace(n: int = 512, period_steps: float = 125.0) -> tuple[list, list]:
    """A clean transverse-velocity signal: period 125 steps, sampled every 2."""
    times = [i * CADENCE for i in range(n)]
    values = [0.03 * np.sin(2 * np.pi * t / period_steps) for t in times]
    return values, times


def _probe_with_timestamps(steps: int, x: int = 24, y: int = 24) -> Probe:
    """A probe carrying the panel's real timestamps; the values are the caller's."""
    sim = LBM2D(width=48, height=48)
    probe = Probe(ProbeSpec(name="p", x=x, y=y, fields=["v"]))
    for step in range(1, steps + 1):
        sim.step()
        if step % CADENCE == 0:
            probe.record(sim, None, step=step)
    return probe


def _strouhal_calls(module) -> list[str]:
    """Every `strouhal_number(...)` call site in a module, as source text."""
    tree = ast.parse(inspect.getsource(module))
    return [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "strouhal_number"
    ]


class TestTheStrouhalNumberIsDimensionless:
    def test_it_scales_with_the_length_scale(self):
        """St = f D / U, so doubling D must double St. This is definitional."""
        values, times = _oscillatory_trace()
        base = strouhal_number(values, diameter=10.0, velocity=0.1, times=times)
        doubled = strouhal_number(values, diameter=20.0, velocity=0.1, times=times)
        assert doubled == pytest.approx(2 * base)

    def test_it_scales_against_the_velocity(self):
        values, times = _oscillatory_trace()
        base = strouhal_number(values, diameter=10.0, velocity=0.1, times=times)
        faster = strouhal_number(values, diameter=10.0, velocity=0.2, times=times)
        assert base == pytest.approx(2 * faster)

    def test_the_default_arguments_recover_the_raw_frequency(self):
        """D = U = 1 collapses St to f. Pinned so the scaling tests above have teeth.

        Asserted to within one FFT bin, which is the most the estimator can support
        and is itself worth recording. With 256 samples at cadence 2 the frequency
        resolution is 1/(256*2) = 1/512 = 0.00195 cycles/step, while a 125-step
        period falls at bin 4.096. The peak is therefore *ambiguous between bins 4
        and 5*, and the estimate came out 0.00977 (bin 5) against a true 0.008 -- a
        22% error that is resolution, not bias. A tolerance tighter than one bin
        would be asserting the FFT resolves frequencies it cannot.
        """
        values, times = _oscillatory_trace()
        measured = strouhal_number(values, diameter=1.0, velocity=1.0, times=times)
        resolution = 1.0 / (256 * CADENCE)
        assert abs(measured - 1.0 / 125.0) <= resolution
        assert resolution / (1.0 / 125.0) > 0.1  # i.e. the window really is this coarse


class TestTheCadenceIsMeasuredNotAssumed:
    def test_the_median_gap_is_the_true_cadence(self):
        probe = _probe_with_timestamps(40)
        assert probe.sample_interval() == 2
        values = [np.sin(2 * np.pi * t / 125.0) for t in probe.times()]
        measured = strouhal_number(
            values, diameter=1.0, velocity=1.0, times=probe.times()
        )
        declared = strouhal_number(values, diameter=1.0, velocity=1.0, dt=CADENCE)
        assert measured == pytest.approx(declared)

    def test_declaring_dt_of_one_would_double_the_frequency(self):
        values, times = _oscillatory_trace()
        wrong = strouhal_number(values, diameter=1.0, velocity=1.0, dt=1.0)
        right = strouhal_number(values, diameter=1.0, velocity=1.0, times=times)
        assert right == pytest.approx(wrong / 2.0)


class TestEveryConsumerPassesTheRecordedTimes:
    def test_the_regime_classifier_forwards_the_probe_timestamps(self):
        source = inspect.getsource(detect_flow_regime)
        assert "_probe_oscillation" in source
        call = inspect.getsource(
            inspect.getmodule(detect_flow_regime)._probe_oscillation
        )
        assert "times=" in call

    def test_no_consumer_calls_strouhal_number_without_times(self):
        """Walked as AST, not matched by regex.

        A regex over the module source matches the *signature* too -- its text also
        contains `v_history` -- so it flagged the definition of the function whose
        whole point is that `times` is optional. `ast` only sees real call sites.
        """
        import analysis.physics as physics
        import analysis.regimes as regimes
        import workbench.panels.analysis_panel as panel

        found = 0
        for module in (physics, regimes, panel):
            for call in _strouhal_calls(module):
                found += 1
                assert "times=" in call, f"{module.__name__}: {call}"
        assert found >= 2, "no call sites found -- the walker is not looking"


@pytest.fixture(scope="module")
def driven():
    """The cylinder preset, driven and sampled at the panel's real cadence."""
    scene = load("presets/scenes/cylinder.json")
    sim = LBM2D(width=96, height=48)
    sim.u_inflow = 0.08
    probe = Probe(scene.probes[0])
    for step in range(1, STEPS + 1):
        sim.step()
        if step % CADENCE == 0:
            probe.record(sim, None, step=step)
    return sim, scene, probe


class TestTheReportAndThePanelAgreeWithTheClassifier:
    def test_the_report_prints_the_classifiers_value(self, driven, tmp_path):
        sim, scene, probe = driven
        regime = detect_flow_regime(sim, scene, [probe], STEPS)
        path = tmp_path / "report.md"
        export_markdown_report(path, scene, sim, STEPS, regime=regime)
        text = path.read_text()
        assert f"{regime.strouhal:.3f}" in text

    def test_the_report_says_not_settled_rather_than_printing_zero(
        self, driven, tmp_path
    ):
        sim, scene, _probe = driven
        quiet = Probe(ProbeSpec(name="quiet", x=2, y=2, fields=["v"]))
        quiet.history["v"] = [0.0] * 100
        regime = detect_flow_regime(sim, scene, [quiet], STEPS)
        assert regime.strouhal is None
        path = tmp_path / "report.md"
        export_markdown_report(path, scene, sim, STEPS, regime=regime)
        assert "not settled" in path.read_text()

    def test_the_scorecard_reports_the_same_value_it_checks(self, driven):
        sim, scene, probe = driven
        regime = detect_flow_regime(sim, scene, [probe], STEPS)
        card = compute_scorecard(sim, scene, [probe], STEPS)
        checks = card.to_dict()["range_checks"]
        assert checks["St"]["value"] == pytest.approx(regime.strouhal)


class TestTheExportedTraceCanReproduceThePublishedNumber:
    def test_the_step_column_holds_simulation_steps_not_indices(self, driven, tmp_path):
        sim, _scene, probe = driven
        path = tmp_path / "probe.csv"
        export_probe_csv([probe], path)
        rows = [
            line.split(",")
            for line in path.read_text().splitlines()
            if line and not line.startswith("#") and not line.startswith("step")
        ]
        assert [int(r[0]) for r in rows] == list(probe.times())

    def test_the_exported_trace_yields_the_published_strouhal(self, driven, tmp_path):
        """The point of the fix: a reader can recompute St from the shipped file."""
        sim, _scene, probe = driven
        path = tmp_path / "probe.csv"
        export_probe_csv([probe], path)
        lines = path.read_text().splitlines()
        header = next(line for line in lines if line.startswith("step"))
        columns = header.split(",")
        v_col = columns.index("v")
        steps, values = [], []
        for line in lines:
            if not line or line.startswith("#") or line.startswith("step"):
                continue
            parts = line.split(",")
            steps.append(float(parts[0]))
            values.append(float(parts[v_col]))
        from_csv = strouhal_number(
            values,
            diameter=24.0,
            velocity=sim.u_inflow,
            times=steps,
        )
        published = detect_flow_regime(sim, _scene, [probe], STEPS).strouhal
        assert from_csv == pytest.approx(published)

    def test_a_cadence_of_one_is_unchanged(self, driven, tmp_path):
        """No regression for probes that do sample every step."""
        sim, _scene, probe = driven
        per_step = Probe(ProbeSpec(name="p", x=34, y=24, fields=["v"]))
        for step in range(1, 41):
            sim.step()
            per_step.record(sim, None, step=step)
        path = tmp_path / "one.csv"
        export_probe_csv([per_step], path)
        rows = [
            int(line.split(",")[0])
            for line in path.read_text().splitlines()
            if line and not line.startswith("#") and not line.startswith("step")
        ]
        assert rows == list(range(1, 41))


class TestTheWindowLimitationIsPinnedRatherThanHidden:
    def test_the_window_is_still_too_short_for_the_period(self):
        """C40 left the 256-sample window in place; this keeps it visible.

        At cadence 2 the analysis window spans 512 steps. The cylinder preset's
        shedding period is ~1250-1875 steps, so the peak sits at bin 4-8 of 128 and
        the estimate is not a resolved frequency.
        """
        source = inspect.getsource(strouhal_number)
        assert "[-256:]" in source

    def test_a_synthetic_period_at_the_window_edge_is_recovered(self):
        """A signal that does fit the window is still measured correctly."""
        times = [i * CADENCE for i in range(128)]
        period = 512.0 / 4  # four cycles inside the 128-sample window
        values = [np.sin(2 * np.pi * t / period) for t in times]
        st = strouhal_number(values, diameter=1.0, velocity=1.0, times=times)
        assert st == pytest.approx(1.0 / period, rel=0.1)
