"""Closing the window must not silently discard work (PLAN.md C47).

There was no `closeEvent` at all. Closing the window threw away unsaved scene edits,
and a recording in progress was abandoned without ever being finalized.
"""

import ast
import inspect
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from engines.lbm2d import LBM2D  # noqa: E402
from scene.scene import CircleObstacle, default_scene  # noqa: E402
from workbench.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    # Building a QWidget before a QApplication exists aborts the process outright
    # ("Current thread ... line 70 in __init__"), so this fixture cannot be skipped.
    win = MainWindow(LBM2D(width=48, height=48))
    try:
        yield win
    finally:
        win.timer.stop()
        win.deleteLater()


def _edit(win: MainWindow) -> None:
    win.scene.obstacles = [
        CircleObstacle(name="probe", x=10, y=10, radius=4),
    ]
    win._on_scene_changed()


def _click(win: MainWindow, name: str) -> bool:
    """Call `win.close()` and report whether the window accepted the close."""
    event = QEvent(QEvent.Type.Close)
    win.closeEvent(event)
    assert event.isAccepted() is (name != "cancel")
    return event.isAccepted()


@pytest.fixture
def click_button(monkeypatch):
    """Make the next modal box report `label` as the button the user pressed.

    `QMessageBox.exec` is patched rather than the dialog class: Shiboken types ignore
    `__new__` substitution, so a fake subclass silently fell through to the real one.
    """

    def _install(label):
        def fake_exec(box):
            if label is not None:
                matches = [b for b in box.buttons() if b.text() == label]
                assert matches, f"no button labelled {label!r}"
                matches[0].click()
            return 0

        monkeypatch.setattr(QMessageBox, "exec", fake_exec, raising=False)

    return _install


class TestTheWindowHasACloseGuard:
    def test_there_is_a_close_event(self):
        assert hasattr(MainWindow, "closeEvent")

    def test_the_title_marker_alone_would_not_have_caught_this(self):
        """C46 made the state visible; nothing acted on it."""
        source = Path("workbench/app.py").read_text()
        tree = ast.parse(source)
        handlers = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert "closeEvent" in handlers
        assert "_confirm_close" in handlers


class TestUnsavedWorkPrompts:
    def test_a_clean_document_closes_without_asking(self, window, monkeypatch):
        def explode(*a, **k):
            raise AssertionError("a clean document must not prompt")

        monkeypatch.setattr(window, "_ask_to_save_changes", explode)
        assert _click(window, "close") is True

    def test_an_edited_document_prompts(self, window, monkeypatch):
        asked = []
        monkeypatch.setattr(
            window,
            "_ask_to_save_changes",
            lambda: (asked.append(1), "discard")[1],
        )
        _edit(window)
        assert _click(window, "close") is True
        assert asked == [1], "edits were discarded without asking"

    def test_cancel_keeps_the_window_open(self, window, monkeypatch):
        monkeypatch.setattr(window, "_ask_to_save_changes", lambda: "cancel")
        _edit(window)
        assert _click(window, "cancel") is False

    def test_save_is_attempted_and_closes_when_it_succeeds(
        self, window, monkeypatch, tmp_path
    ):
        path = tmp_path / "scene.json"
        saved = []
        monkeypatch.setattr(window, "_file_path", path, raising=False)
        monkeypatch.setattr(
            window,
            "_file_save",
            lambda: (window._mark_document_clean(), saved.append(1)),
        )
        monkeypatch.setattr(window, "_ask_to_save_changes", lambda: "save")
        _edit(window)
        assert _click(window, "close") is True
        assert saved == [1]

    def test_a_cancelled_save_as_keeps_the_window_open(self, window, monkeypatch):
        """Save opens Save As; cancelling it must not be read as 'saved'."""
        monkeypatch.setattr(window, "_ask_to_save_changes", lambda: "save")
        monkeypatch.setattr(window, "_file_save", lambda: None)
        _edit(window)
        assert _click(window, "cancel") is False


class TestThePromptItself:
    """The dialog, tested by clicking its real buttons."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Save", "save"),
            ("Discard", "discard"),
            ("Cancel", "cancel"),
        ],
    )
    def test_each_button_maps_to_its_answer(
        self, window, click_button, label, expected
    ):
        click_button(label)
        assert window._ask_to_save_changes() == expected

    def test_dismissing_with_the_window_chrome_is_a_cancel(self, window, click_button):
        """The X button must never be the destructive choice."""
        click_button(None)
        assert window._ask_to_save_changes() == "cancel"

    def test_it_offers_exactly_three_choices(self, window):
        box, buttons = window._save_prompt()
        assert sorted(buttons.values()) == ["cancel", "discard", "save"]
        assert sorted(b.text() for b in buttons) == ["Cancel", "Discard", "Save"]

    def test_only_save_is_the_default(self, window):
        """Enter saves. It must not discard."""
        box, buttons = window._save_prompt()
        defaults = [answer for button, answer in buttons.items() if button.isDefault()]
        assert defaults == ["save"]

    def test_discard_carries_the_destructive_role(self, window):
        """Roles are not readable back off the widget in this Qt version, so this is
        a source check; `buttonRole()` does not exist on QPushButton here."""
        source = inspect.getsource(MainWindow._save_prompt)
        assert '"Discard": QMessageBox.ButtonRole.DestructiveRole' in source
        assert '"Save": QMessageBox.ButtonRole.AcceptRole' in source
        assert '"Cancel": QMessageBox.ButtonRole.RejectRole' in source

    def test_it_names_the_file_being_saved(self, window, tmp_path):
        window._file_path = tmp_path / "my scene.json"
        box, _ = window._save_prompt()
        assert "my scene" in box.text()
        assert "unsaved changes" in box.text()

    def test_an_unnamed_document_is_named_by_its_scene(self, window):
        window._file_path = None
        window.scene.name = "flow study"
        box, _ = window._save_prompt()
        assert "flow study" in box.text()


class TestALiveRecordingIsFinalized:
    def test_a_recording_in_progress_is_stopped_on_close(self, window, monkeypatch):
        class FakeRecorder:
            recording = True

            def __init__(self):
                self.closed = 0

            def close(self):
                self.closed += 1

        recorder = FakeRecorder()
        monkeypatch.setattr(window, "_recorder", recorder, raising=False)
        stopped = []
        monkeypatch.setattr(
            window,
            "_stop_video_recording",
            lambda: (recorder.close(), stopped.append(1)),
        )
        assert _click(window, "close") is True
        assert stopped == [1], "the writer was dropped without being finalized"

    def test_an_idle_recorder_is_left_alone(self, window, monkeypatch):
        class Idle:
            recording = False

            def close(self):
                raise AssertionError("nothing to finalize")

        monkeypatch.setattr(window, "_recorder", Idle(), raising=False)
        assert _click(window, "close") is True

    def test_no_recorder_at_all_is_fine(self, window):
        assert window._recorder is None
        assert _click(window, "close") is True


class TestARunningSweepBlocksTheClose:
    def test_a_running_worker_stops_the_window_closing(self, window, monkeypatch):
        class FakeWorker:
            def isRunning(self):
                return True

        class FakeDialog:
            _worker = FakeWorker()

        monkeypatch.setattr(window, "_sweep_dialog", FakeDialog(), raising=False)
        monkeypatch.setattr(
            QMessageBox, "information", staticmethod(lambda *a, **k: None)
        )
        assert _click(window, "cancel") is False

    def test_a_finished_worker_does_not(self, window, monkeypatch):
        class FakeWorker:
            def isRunning(self):
                return False

        class FakeDialog:
            _worker = FakeWorker()

        monkeypatch.setattr(window, "_sweep_dialog", FakeDialog(), raising=False)
        assert _click(window, "close") is True

    def test_no_dialog_at_all_does_not(self, window):
        assert window._sweep_dialog is None
        assert _click(window, "close") is True

    def test_the_window_knows_which_dialog_it_opened(self):
        """It held no reference, so it could not even ask the question."""
        source = inspect.getsource(MainWindow._open_sweep_dialog)
        assert "_sweep_dialog" in source


class TestTheRunStateSurvivesACancelledClose:
    def test_a_cancelled_close_leaves_the_timer_running(self, window, monkeypatch):
        """The user asked to keep working; stopping the run would be a second loss."""
        monkeypatch.setattr(window, "_ask_to_save_changes", lambda: "cancel")
        _edit(window)
        was_running = window.timer.isActive()
        assert _click(window, "cancel") is False
        assert window.timer.isActive() is was_running

    def test_an_accepted_close_stops_the_timer(self, window):
        assert window.timer.isActive()
        _click(window, "close")
        assert not window.timer.isActive()

    def test_an_accepted_close_does_not_restart_it(self, window, monkeypatch):
        monkeypatch.setattr(window, "_ask_to_save_changes", lambda: "cancel")
        _edit(window)
        _click(window, "cancel")
        assert window.timer.isActive()


class TestTheOrderingIsDeliberate:
    def test_a_live_job_is_checked_before_unsaved_work(self):
        """A job cannot be finalized once the window is gone."""
        source = inspect.getsource(MainWindow._confirm_close)
        assert source.index("_sweep_in_progress") < source.index("is_dirty")

    def test_the_recorder_is_finalized_before_the_prompt(self):
        """Finalizing first means a cancelled prompt has already preserved the file."""
        source = inspect.getsource(MainWindow._confirm_close)
        assert source.index("_stop_video_recording") < source.index(
            "_ask_to_save_changes"
        )


class TestDefaultsStillWork:
    def test_a_default_document_is_clean(self):
        scene = default_scene()
        assert scene.name

    def test_the_fixture_window_starts_clean(self, window):
        assert window.is_dirty is False

    def test_probing_the_default_scene_does_not_dirt_it(self, window):
        window.scene.width = 64
        window._on_scene_changed()
        assert window.is_dirty is True
