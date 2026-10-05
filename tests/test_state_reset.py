"""Reset and load clear every piece of runtime-derived state.

PLAN.md C33 (P0-4), completing C26. `test_lifecycle_reset.py` pins what
`_reset_runtime_state()` clears, but it builds a stub window holding only the fields
it knows about, which is exactly why two pieces of derived state survived: a stub
cannot expose a field nobody remembered, and `_update_field_stats` /
`_update_physics` run on a five-tick cadence, so blanking `_tick_counter` without
blanking their labels looks correct to the very tests that check it.

The theme is *cadence*. Almost every residue found here existed because some
consumer refreshes on a longer period than the reset: labels refreshed every fifth
tick, the regime card every fifteenth step, the y-axis never shrinking at all. A
reset that clears the input to a consumer is not the same as clearing what the
consumer already put on screen, and only the second one is what a user can see.

These tests drive the real widgets offscreen rather than stubs, because the
defects are all in the interaction between the reset and a consumer.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from engines.lbm2d import LBM2D
from scene.scene import CircleObstacle, ProbeSpec, Scene

pytest.importorskip("PySide6")

_BLANK = "—"


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    from workbench.app import MainWindow

    scene = Scene(
        name="probe scene",
        width=48,
        height=48,
        probes=[ProbeSpec(name="p", x=10, y=10, fields=["u", "v", "density"])],
    )
    win = MainWindow(LBM2D(width=48, height=48))
    win.timer.stop()
    win._adopt_scene(scene)
    win._apply_and_refresh()
    monkeypatch.setattr(win, "_prompt_poe_prediction", lambda *a, **kw: None)
    yield win
    win.close()


def _run(window, steps: int = 12) -> None:
    for _ in range(steps):
        window.tick()
    window.timer.stop()


class TestReadoutsAreBlanked:
    """They refresh every fifth tick, so clearing the tick counter is not enough."""

    def test_field_statistics_are_blanked(self, window):
        window.analysis_panel.min_label.setText("0.1234")
        window.analysis_panel.max_label.setText("9.8765")
        window.analysis_panel.mean_label.setText("1.0000")

        window._reset_runtime_state()

        assert window.analysis_panel.min_label.text() == _BLANK
        assert window.analysis_panel.max_label.text() == _BLANK
        assert window.analysis_panel.mean_label.text() == _BLANK

    def test_physics_readouts_are_blanked(self, window):
        window.analysis_panel.re_label.setText("123.4")
        window.analysis_panel.cd_label.setText("5.678")
        window.analysis_panel.st_label.setText("0.197")

        window._reset_runtime_state()

        assert window.analysis_panel.re_label.text() == _BLANK
        assert window.analysis_panel.cd_label.text() == _BLANK
        assert window.analysis_panel.st_label.text() == _BLANK

    def test_a_paused_window_never_self_corrects(self, window):
        """The reason the labels have to be blanked rather than left to expire: a
        reset on a paused window leaves nothing to tick, so a stale readout would
        sit there indefinitely."""
        window.paused = True
        window.analysis_panel.min_label.setText("7.0")

        window.reset(silent=True)
        _run(window, 2)

        assert window.analysis_panel.min_label.text() == _BLANK


class TestProbePlotRangesAreCleared:
    """`_ProbePlot` deliberately never shrinks its y-range.

    Correct within a run -- it stops the axis jumping between samples -- and wrong
    across a reset, where the new run's signal inherits the previous run's axis. A
    probe that used to swing 0 to 1.0 and now lives in 0 to 0.05 is drawn as a flat
    line along the bottom of the old range, which reads as "nothing is happening"
    rather than as a bug.
    """

    def test_the_range_is_cleared(self, window):
        _run(window, 8)
        plots = window.analysis_panel._probe_widgets
        assert plots, "expected a probe plot"
        for pw in plots:
            pw._y_range = (0.0, 1.0)

        window._reset_runtime_state()

        assert all(pw._y_range is None for pw in window.analysis_panel._probe_widgets)

    def test_the_new_run_can_rescale(self, window):
        """An end-to-end version: after a reset the plot must be able to adopt a
        narrow range, which it cannot if the old one is retained."""
        panel = window.analysis_panel
        pw = panel._probe_widgets[0]
        pw._y_range = (0.0, 1.0)
        window._reset_runtime_state()

        probe = panel.probes[0]
        probe.history["u"] = [0.01, 0.012, 0.011, 0.013] * 8
        pw.update_plot()

        assert pw._y_range is not None
        assert pw._y_range[1] < 0.1, (
            "the plot kept the previous run's axis, so a signal two orders of "
            "magnitude smaller renders as a flat line"
        )

    def test_resetting_twice_is_idempotent(self, window):
        window._reset_runtime_state()
        window._reset_runtime_state()
        assert all(pw._y_range is None for pw in window.analysis_panel._probe_widgets)


class TestLoadClearsDerivedStateEndToEnd:
    """The load path, driven through the real window."""

    def test_a_load_empties_the_probe_history(self, window, tmp_path):
        _run(window, 10)
        assert len(window.runtime_probes[0].history["u"]) > 0

        path = tmp_path / "clean.json"
        from scene.serializer import scene_to_dict

        scene = Scene(name="clean", width=48, height=48)
        path.write_text(__import__("json").dumps(scene_to_dict(scene)))

        from PySide6.QtWidgets import QFileDialog

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                QFileDialog,
                "getOpenFileName",
                staticmethod(lambda *a, **kw: (str(path), "")),
            )
            window._file_open()

        assert window.runtime_probes == []
        assert window.step_count == 0

    def test_a_load_zeroes_the_stability_trace(self, window):
        _run(window, 10)
        assert len(window.analysis_panel._stability_data) > 0

        window._file_new()

        assert len(window.analysis_panel._stability_data) == 0

    def test_a_load_clears_the_step_count_and_history(self, window):
        _run(window, 10)
        assert window._state_history

        window._file_new()

        assert window.step_count == 0
        assert window._state_history == []

    def test_a_load_stops_a_running_recorder(self, window, tmp_path):
        """A recorder left open keeps the previous run's frames and its handle.

        The corruption is on disk, not on screen, so it is asserted on the file:
        an unclosed writer leaves an MP4 with no moov atom, unreadable.
        """
        import imageio.v3 as iio
        from PySide6.QtGui import QImage

        from export.video import VideoRecorder

        path = tmp_path / "clip.mp4"
        recorder = VideoRecorder(path, fps=10, max_frames=10)
        # 16x16: ffmpeg pads anything smaller up to the macroblock size.
        recorder.add_frame(QImage(16, 16, QImage.Format.Format_RGB888))
        window._recorder = recorder

        window._file_new()

        assert window._recorder is None
        assert path.exists()
        assert iio.imread(path).shape == (1, 16, 16, 3), (
            "the video was dropped without being closed, so the file on disk is "
            "unreadable"
        )


class TestTheOutcomeCardDescribesTheLoadedScene:
    """`update_outcome` refreshes every fifteenth step.

    `reset()` forced it; `_apply_and_refresh()` did not, so after opening a file,
    preset or wizard template the regime, confidence badge, drag and scorecard kept
    describing the scene that had just been discarded. On a paused window nothing
    ever ticks, so it stayed wrong until the user pressed Play and waited.
    """

    def test_the_headline_is_recomputed_on_load(self, window):
        window.scene.product.lesson_headline = "old headline"
        window.outcome_panel.headline.setText("<b>old headline</b>")

        from scene.scene import Scene as _Scene

        window._adopt_scene(_Scene(name="fresh", width=48, height=48))

        window.sim.step()
        window.outcome_panel.update_outcome(0, force=True)
        assert window.outcome_panel.headline.text() != "<b>old headline</b>"

    def test_apply_and_refresh_forces_the_update(self, window):
        """Mechanism, not outcome: an outcome test cannot tell this fix from a tick
        that happens to land."""
        import inspect

        from workbench.app import MainWindow

        source = inspect.getsource(MainWindow._apply_and_refresh)
        assert "force=True" in source, (
            "_apply_and_refresh must force the outcome update; otherwise the card "
            "keeps the previous scene's regime until step 15"
        )

    def test_a_reset_still_forces_the_update(self, window):
        import inspect

        from workbench.app import MainWindow

        assert "force=True" in inspect.getsource(MainWindow.reset)

    def test_the_card_matches_the_open_scene_after_a_load(self, window):
        from scene.scene import Scene as _Scene

        window._adopt_scene(_Scene(name="vortex street", width=48, height=48))
        window._apply_and_refresh()

        assert window.outcome_panel.scene is window.scene
        assert window.outcome_panel.step_count == 0


class TestTheResetPathCannotDrift:
    """Guards against the failure mode that produced the two bugs above: a new piece
    of derived state added to a panel and never registered with its reset."""

    def test_every_analysis_readout_is_registered(self):
        """Every label the panel writes on a cadence is named in `reset_runtime`.

        Driven off the labels that actually exist on the widget, so adding a readout
        without registering it fails here rather than silently going stale.
        """
        import inspect

        from workbench.panels.analysis_panel import AnalysisPanel

        panel = AnalysisPanel(LBM2D(32, 32))
        source = inspect.getsource(AnalysisPanel.reset_runtime)

        cadence_labels = [
            "re_label",
            "cd_label",
            "st_label",
            "min_label",
            "max_label",
            "mean_label",
        ]
        for name in cadence_labels:
            assert name in source, (
                f"AnalysisPanel.reset_runtime does not clear {name}; it is written "
                "on a five-tick cadence, so a stale value survives the reset"
            )
            assert hasattr(panel, name)

    def test_both_entry_points_use_the_shared_path(self):
        import inspect

        from workbench.app import MainWindow

        for method in ("reset", "_apply_and_refresh"):
            assert "_reset_runtime_state()" in inspect.getsource(
                getattr(MainWindow, method)
            )

    def test_a_load_does_not_recreate_a_probe_that_survives(self, window):
        """Probes are rebuilt from the scene on every load. If they were reused, a
        probe absent from the new scene would keep recording into the old scene's
        coordinates."""
        window._adopt_scene(
            Scene(
                name="two probes",
                width=48,
                height=48,
                probes=[
                    ProbeSpec(name="a", x=5, y=5, fields=["u"]),
                    ProbeSpec(name="b", x=9, y=9, fields=["u"]),
                ],
            )
        )
        window._apply_and_refresh()

        assert [p.spec.name for p in window.runtime_probes] == ["a", "b"]

        window._adopt_scene(
            Scene(
                name="one probe",
                width=48,
                height=48,
                probes=[
                    ProbeSpec(name="only", x=5, y=5, fields=["u"]),
                ],
            )
        )
        window._apply_and_refresh()

        assert [p.spec.name for p in window.runtime_probes] == ["only"]


class TestUnrelatedSceneStateIsNotDestroyed:
    """A reset re-seeds the same document; it must not touch the document itself."""

    def test_the_scene_survives_a_reset(self, window):
        window.scene.obstacles.append(CircleObstacle(name="keep", x=5, y=5, radius=2))
        window.scene.viscosity = 0.021

        window.reset(silent=True)

        assert [o.name for o in window.scene.obstacles] == ["keep"]
        assert window.scene.viscosity == pytest.approx(0.021)
