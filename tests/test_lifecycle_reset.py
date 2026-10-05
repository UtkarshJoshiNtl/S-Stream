"""One reset path for all derived runtime state.

Reset, preset load, saved-file open and engine swap all invalidate the same
derived state, and each used to clear a different subset of it. The residue was
user-visible:

  * the stability trace kept drawing the previous run's convergence curve beside
    the new one,
  * probe plots kept carrying the previous run's samples,
  * `_tick_counter` stayed offset, so probe sampling no longer lined up with the
    step count it is plotted against,
  * a reset triggered mid-drag left `_scrubbing` True, and `tick()` then skips
    `sim.step()` forever -- the app looks frozen and nothing is wrong,
  * an active video recorder kept appending the new run's frames to the old
    video.
"""

from __future__ import annotations

import pytest

from engines.lbm2d import LBM2D
from scene.probe import Probe
from scene.scene import ProbeSpec, Scene


class _StubStatus:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, text: str, *_args) -> None:  # noqa: N802 - Qt naming
        self.messages.append(text)


class _StubTimeline:
    def __init__(self) -> None:
        self.range = (5, 5)
        self.value = 5
        self.blocked = False

    def blockSignals(self, flag: bool) -> None:  # noqa: N802 - Qt naming
        self.blocked = flag

    def setRange(self, lo: int, hi: int) -> None:  # noqa: N802 - Qt naming
        self.range = (lo, hi)

    def setValue(self, value: int) -> None:  # noqa: N802 - Qt naming
        self.value = value


class _StubLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self.text = text


class _StubAnalysisPanel:
    def __init__(self, probes) -> None:
        self.probes = probes
        self.reset_calls = 0

    def reset_runtime(self) -> None:
        self.reset_calls += 1
        for probe in self.probes:
            probe.clear()


class _StubViewport:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate_caches(self) -> None:
        self.invalidated += 1


class _StubButton:
    def __init__(self) -> None:
        self.text = ""
        self.checked = False

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self.text = text

    def setChecked(self, flag: bool) -> None:  # noqa: N802 - Qt naming
        self.checked = flag


class _StubRecorder:
    def __init__(self) -> None:
        self.recording = True
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.recording = False


def _window():
    from workbench.app import MainWindow

    window = MainWindow.__new__(MainWindow)
    window.sim = LBM2D(64, 64)
    window.scene = Scene(
        width=64,
        height=64,
        probes=[ProbeSpec(name="p", x=10, y=10, fields=["u", "density"])],
    )
    probes = [Probe(spec) for spec in window.scene.probes]
    window.analysis_panel = _StubAnalysisPanel(probes)
    window.viewport = _StubViewport()
    window.status = _StubStatus()
    window.timeline_slider = _StubTimeline()
    window.timeline_label = _StubLabel()
    window.record_btn = _StubButton()
    window.step_count = 137
    window._state_history = [(1, "f", "rho")] * 9
    window._scrubbing = True
    window._demo_running = True
    window._demo_target = 500
    window._recorder = _StubRecorder()
    return window


class TestResetClearsDerivedState:
    def test_step_count(self):
        window = _window()
        window._reset_runtime_state()
        assert window.step_count == 0

    def test_scrubbing_flag(self):
        """Left True, tick() skips sim.step() forever and the app looks hung."""
        window = _window()
        window._reset_runtime_state()
        assert window._scrubbing is False

    def test_state_history(self):
        window = _window()
        window._reset_runtime_state()
        assert window._state_history == []

    def test_timeline_widget(self):
        window = _window()
        window._reset_runtime_state()
        assert window.timeline_slider.range == (0, 0)
        assert window.timeline_label.text == "Step 0"

    def test_timeline_signals_stay_blocked_off(self):
        window = _window()
        window._reset_runtime_state()
        assert window.timeline_slider.blocked is False

    def test_demo_is_stopped(self):
        window = _window()
        window._reset_runtime_state()
        assert window._demo_running is False

    def test_recorder_is_closed(self):
        window = _window()
        recorder = window._recorder
        window._reset_runtime_state()
        assert recorder.closed is True
        assert window._recorder is None

    def test_panel_state_is_reset(self):
        window = _window()
        window._reset_runtime_state()
        assert window.analysis_panel.reset_calls == 1

    def test_viewport_caches_are_invalidated(self):
        window = _window()
        window._reset_runtime_state()
        assert window.viewport.invalidated == 1

    def test_no_recorder_is_fine(self):
        window = _window()
        window._recorder = None
        window._reset_runtime_state()
        assert window.step_count == 0

    def test_probe_histories_are_emptied(self):
        window = _window()
        sim = window.sim
        for probe in window.analysis_panel.probes:
            for _ in range(3):
                probe.record(sim)
        assert all(len(p.history["u"]) == 3 for p in window.analysis_panel.probes)
        window._reset_runtime_state()
        assert all(len(p.history["u"]) == 0 for p in window.analysis_panel.probes)


class TestBothPathsShareOneReset:
    """The point of C26: reset() and _apply_and_refresh() must not drift."""

    def _callable_names(self, method_name: str) -> set[str]:
        import inspect

        from workbench.app import MainWindow

        source = inspect.getsource(getattr(MainWindow, method_name))
        return {token for token in source.split() if token.startswith("self._")}

    def test_reset_delegates_to_the_shared_path(self):
        import inspect

        from workbench.app import MainWindow

        assert "_reset_runtime_state()" in inspect.getsource(MainWindow.reset)

    def test_apply_and_refresh_delegates_to_the_shared_path(self):
        import inspect

        from workbench.app import MainWindow

        assert "_reset_runtime_state()" in inspect.getsource(
            MainWindow._apply_and_refresh
        )

    def test_reset_no_longer_hand_rolls_its_own_clearing(self):
        """If both paths touch the same fields directly they will drift again."""
        import inspect

        from workbench.app import MainWindow

        source = inspect.getsource(MainWindow.reset)
        for field in ("_state_history", "timeline_slider", "timeline_label"):
            assert field not in source, (
                f"reset() still clears {field} directly; it belongs in "
                "_reset_runtime_state"
            )


class _StubLabel:
    def __init__(self) -> None:
        self.text = "stale"

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self.text = text


class TestAnalysisPanelReset:
    def _panel(self):
        from workbench.panels.analysis_panel import AnalysisPanel

        panel = AnalysisPanel.__new__(AnalysisPanel)
        from collections import deque

        panel._stability_data = deque([1.0, 2.0], maxlen=256)
        panel._tick_counter = 42
        panel.probes = [Probe(ProbeSpec(name="p", x=1, y=1, fields=["u"]))]
        panel.stability_curve = None
        # C33: reset_runtime also blanks the on-screen readouts and the probe
        # y-ranges. This stub has to register them, and registering them by hand is
        # precisely the weakness C33's tests exist to remove -- a stub cannot
        # expose a field nobody remembered, which is how the residue survived C26.
        # See tests/test_state_reset.py for the same behaviour asserted against the
        # real widgets.
        panel.re_label = _StubLabel()
        panel.cd_label = _StubLabel()
        panel.st_label = _StubLabel()
        panel.min_label = _StubLabel()
        panel.max_label = _StubLabel()
        panel.mean_label = _StubLabel()
        panel._probe_widgets = []
        return panel

    def test_stability_trace_is_cleared(self):
        panel = self._panel()
        panel.reset_runtime()
        assert len(panel._stability_data) == 0

    def test_tick_counter_is_zeroed(self):
        """Otherwise probe sampling stays offset from the plotted step count."""
        panel = self._panel()
        panel.reset_runtime()
        assert panel._tick_counter == 0

    def test_probes_are_cleared(self):
        panel = self._panel()
        sim = LBM2D(32, 32)
        panel.probes[0].record(sim)
        panel.reset_runtime()
        assert len(panel.probes[0].history["u"]) == 0

    def test_readouts_are_blanked(self):
        panel = self._panel()
        panel.reset_runtime()
        for name in ("re_label", "cd_label", "st_label", "min_label", "max_label"):
            assert getattr(panel, name).text == "—"

    def test_probe_plot_ranges_are_cleared(self):
        class _PW:
            _y_range = (0.0, 1.0)

        panel = self._panel()
        panel._probe_widgets = [_PW()]
        panel.reset_runtime()
        assert panel._probe_widgets[0]._y_range is None

    def test_tolerates_a_missing_curve(self):
        panel = self._panel()
        panel.stability_curve = None
        panel.reset_runtime()

    def test_tolerates_no_probes(self):
        panel = self._panel()
        panel.probes = []
        panel.reset_runtime()

    def test_is_idempotent(self):
        panel = self._panel()
        panel.reset_runtime()
        panel.reset_runtime()
        assert panel._tick_counter == 0
        assert panel.min_label.text == "—"


class TestViewportInvalidate:
    def _viewport(self):
        from workbench.viewport import Viewport

        viewport = Viewport.__new__(Viewport)
        viewport._tex_init = True
        return viewport

    def test_texture_flag_is_cleared(self):
        viewport = self._viewport()
        viewport.invalidate_caches()
        assert viewport._tex_init is False


class TestGuidedDemoOrdering:
    def test_demo_starts_after_the_reset_that_clears_it(self):
        """_run_guided_demo calls reset(silent=True), which now clears
        _demo_running. It must set the flag afterwards or the demo never runs."""
        import inspect

        from workbench.app import MainWindow

        source = inspect.getsource(MainWindow._run_guided_demo)
        assert source.index("self.reset(silent=True)") < source.index(
            "self._demo_running = True"
        )


@pytest.mark.parametrize("field", ["step_count", "_scrubbing", "_state_history"])
def test_window_stub_exposes_the_fields_under_test(field):
    assert field in vars(_window())
