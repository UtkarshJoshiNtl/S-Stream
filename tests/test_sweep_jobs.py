"""Long-running jobs must be cancellable and must fail visibly (PLAN.md C48).

`_SweepWorker.run` called `run_sweep` and emitted `finished` only on success. An
exception therefore unwound out of `run()`, PySide6 printed it to stderr -- which a
GUI user never sees -- `finished` was never emitted, and the dialog's Run button
stayed disabled with no explanation. There was also no way to stop a sweep once
started, and the one button the dialog offered dismissed it, which destroys the
`QThread` mid-`run` and aborts the process.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from analysis.sweep import (  # noqa: E402
    _CANCEL_CHECK_INTERVAL,
    SweepResult,
    run_sweep,
)
from scene.scene import CircleObstacle, Scene  # noqa: E402
from workbench.dialogs.sweep_dialog import SweepDialog, _SweepWorker  # noqa: E402

CHUNKS_PER_5K = 5000 // _CANCEL_CHECK_INTERVAL  # 20 chunks, plus 1 top-of-loop check


def scene(width: int = 64) -> Scene:
    return Scene(
        name="s",
        width=width,
        height=width,
        obstacles=[CircleObstacle(name="cyl", x=30, y=30, radius=6)],
    )


def allow(n: int):
    """A `should_continue` that permits `n` checks and then refuses, with its own
    counter. Sharing one across cases measures the *other* cases' budget."""
    state = {"i": 0}

    def predicate() -> bool:
        state["i"] += 1
        return state["i"] <= n

    predicate.checks = state
    return predicate


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(app):
    dlg = SweepDialog(scene())
    yield dlg
    dlg.deleteLater()


class TestTheResultSaysWhetherItFinished:
    def test_a_complete_sweep_is_not_cancelled(self):
        r = run_sweep(scene(), "u_inflow", [0.05, 0.06], ["reynolds_number"], 200)
        assert r.cancelled is False

    def test_cancelling_keeps_the_values_that_finished(self):
        # Enough checks for value 0 (1 + 20 chunks) and part of value 1.
        r = run_sweep(
            scene(),
            "u_inflow",
            [0.05, 0.06, 0.07, 0.08],
            ["reynolds_number"],
            5000,
            should_continue=allow(CHUNKS_PER_5K + 2),
        )
        assert r.cancelled is True
        assert len(r.values) == 1
        assert len(r.data["reynolds_number"]) == 1

    def test_a_half_run_value_is_dropped_not_plotted(self):
        """Half the steps is a different quantity, not a worse answer."""
        r = run_sweep(
            scene(),
            "u_inflow",
            [0.05, 0.06],
            ["reynolds_number"],
            5000,
            should_continue=allow(CHUNKS_PER_5K // 2),
        )
        assert r.cancelled is True
        assert r.values == []
        assert r.data["reynolds_number"] == []

    def test_the_axis_always_matches_the_data(self):
        """The plot pairs x with y, so the two lists cannot disagree in length."""
        for budget in (0, 1, CHUNKS_PER_5K + 1, CHUNKS_PER_5K * 4):
            r = run_sweep(
                scene(),
                "u_inflow",
                [0.05, 0.06, 0.07],
                ["reynolds_number"],
                300,
                should_continue=allow(budget),
            )
            assert len(r.values) == len(r.data["reynolds_number"])

    def test_every_value_survives_an_uncancelled_run(self):
        """Regression: truncating inside the loop re-truncates an already-shortened
        list, which pinned the axis at its first length however many values ran."""
        r = run_sweep(
            scene(),
            "u_inflow",
            [0.05, 0.06, 0.07, 0.08],
            ["reynolds_number"],
            200,
            should_continue=lambda: True,
        )
        assert len(r.values) == 4
        assert len(r.data["reynolds_number"]) == 4

    def test_the_callers_list_is_not_mutated(self):
        values = [0.05, 0.06, 0.07]
        run_sweep(
            scene(),
            "u_inflow",
            values,
            ["reynolds_number"],
            200,
            should_continue=allow(0),
        )
        assert values == [0.05, 0.06, 0.07]

    def test_progress_never_exceeds_the_total(self):
        seen = []
        run_sweep(
            scene(),
            "u_inflow",
            [0.05, 0.06, 0.07],
            ["reynolds_number"],
            200,
            progress_callback=lambda c, t: seen.append((c, t)),
            should_continue=allow(0),
        )
        assert all(1 <= c <= t for c, t in seen)


class TestCancellationIsCheckedInsideTheSimulation:
    def test_a_long_run_stops_promptly(self):
        """Cancellation only at the value boundary would leave the user waiting out
        a whole run after they asked to stop."""
        checks = allow(_CANCEL_CHECK_INTERVAL // _CANCEL_CHECK_INTERVAL)  # 1 chunk
        r = run_sweep(
            scene(128),
            "u_inflow",
            [0.05, 0.06],
            ["reynolds_number"],
            50000,
            should_continue=checks,
        )
        assert r.cancelled is True
        # 50000 steps at 128x128 is minutes of work; one chunk is well under a second.
        assert checks.checks["i"] == 2

    def test_the_interval_is_short_enough_to_matter(self):
        assert _CANCEL_CHECK_INTERVAL <= 500


class TestTheResultSurvivesSerialization:
    def test_cancelled_round_trips(self):
        r = run_sweep(
            scene(),
            "u_inflow",
            [0.05, 0.06],
            ["reynolds_number"],
            200,
            should_continue=allow(0),
        )
        assert SweepResult.from_dict(r.to_dict()).cancelled is True

    def test_a_complete_round_trips(self):
        r = run_sweep(scene(), "u_inflow", [0.05], ["reynolds_number"], 200)
        assert SweepResult.from_dict(r.to_dict()).cancelled is False

    def test_documents_written_before_the_field_are_complete(self):
        old = {
            "parameter": "u_inflow",
            "values": [0.05],
            "measurements": ["reynolds_number"],
            "data": {"reynolds_number": [1.0]},
            "elapsed": 1.0,
        }
        assert SweepResult.from_dict(old).cancelled is False


class TestTheWorkerReportsFailure:
    def _stale_scene(self):
        """The scene loses the obstacle after the parameter was offered."""
        import copy

        stale = copy.deepcopy(scene())
        stale.obstacles = []
        return stale

    def _drain(self, app, worker, limit: int = 600):
        import time

        for _ in range(limit):
            app.processEvents()
            if not worker.isRunning():
                break
            time.sleep(0.005)
        app.processEvents()
        worker.wait(3000)

    def test_a_failure_emits_the_message(self, app):
        worker = _SweepWorker(
            self._stale_scene(), "cyl.radius", [0.05], ["reynolds_number"], 200
        )
        failures = []
        worker.failed.connect(failures.append)
        worker.start()
        self._drain(app, worker)
        assert failures, "the exception was swallowed instead of reported"
        assert "cyl" in failures[0]

    def test_a_failure_does_not_also_claim_success(self, app):
        worker = _SweepWorker(
            self._stale_scene(), "cyl.radius", [0.05], ["reynolds_number"], 200
        )
        results, failures = [], []
        worker.finished.connect(results.append)
        worker.failed.connect(failures.append)
        worker.start()
        self._drain(app, worker)
        assert results == []
        assert len(failures) == 1

    def test_a_success_emits_the_result_and_not_a_failure(self, app):
        worker = _SweepWorker(scene(), "u_inflow", [0.05], ["reynolds_number"], 200)
        results, failures = [], []
        worker.finished.connect(results.append)
        worker.failed.connect(failures.append)
        worker.start()
        self._drain(app, worker)
        assert failures == []
        assert len(results) == 1
        assert results[0].cancelled is False

    def test_interruption_produces_a_cancelled_result_not_a_failure(self, app):
        worker = _SweepWorker(
            scene(128), "u_inflow", [0.05] * 8, ["reynolds_number"], 50000
        )
        results, failures = [], []
        worker.finished.connect(results.append)
        worker.failed.connect(failures.append)
        worker.start()
        worker.requestInterruption()
        self._drain(app, worker)
        assert failures == []
        assert results and results[0].cancelled is True


class TestTheDialogOffersACancel:
    def test_cancel_is_only_available_while_running(self, dialog):
        assert dialog._cancel_btn.isEnabled() is False
        dialog._set_running(True)
        assert dialog._cancel_btn.isEnabled() is True
        assert dialog._run_btn.isEnabled() is False

    def test_close_is_disabled_while_a_thread_runs(self, dialog):
        """Rejecting the dialog would destroy a running QThread and abort."""
        dialog._set_running(True)
        assert dialog._close_btn.isEnabled() is False
        dialog._set_running(False)
        assert dialog._close_btn.isEnabled() is True

    def test_cancelling_asks_the_worker_to_stop(self, dialog):
        class Worker:
            def __init__(self):
                self.asked = 0

            def requestInterruption(self):
                self.asked += 1

        worker = Worker()
        dialog._worker = worker
        dialog._cancel_sweep()
        assert worker.asked == 1
        assert "Cancelling" in dialog._est_label.text()

    def test_cancelling_with_no_worker_is_harmless(self, dialog):
        dialog._worker = None
        dialog._cancel_sweep()

    def test_a_second_run_cannot_start_while_one_is_live(self, dialog):
        class Worker:
            def isRunning(self):
                return True

        dialog._worker = Worker()
        dialog._run_btn.setEnabled(True)
        dialog._run_sweep()
        assert dialog._worker.__class__ is Worker


class TestTheDialogShowsWhatHappened:
    @pytest.fixture
    def shown(self, monkeypatch):
        """The dialog's notifications are modal by construction; capture them.

        Returns the capture rather than stashing it on the fixture: an autouse fixture
        writing to `self` puts the lists on the *fixture*, and a test reading them off
        itself finds nothing there.
        """
        box = SimpleNamespace(information=[], warning=[])
        monkeypatch.setattr(
            QMessageBox,
            "information",
            staticmethod(lambda *a, **k: box.information.append(a[1:3])),
        )
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            staticmethod(lambda *a, **k: box.warning.append(a[1:3])),
        )
        return box

    def test_a_failure_is_shown_and_unlocks_the_dialog(self, dialog, shown):
        dialog._set_running(True)
        dialog._on_failure("ValueError: Obstacle 'cyl' not found")
        assert dialog._run_btn.isEnabled() is True
        assert dialog._cancel_btn.isEnabled() is False
        assert dialog._close_btn.isEnabled() is True
        assert dialog._error == "ValueError: Obstacle 'cyl' not found"
        assert shown.warning and "cyl" in shown.warning[0][1]
        assert "failed" in dialog._est_label.text().lower()

    def test_a_failure_clears_stale_results(self, dialog, shown):
        dialog._result = None
        dialog._on_failure("boom")
        assert dialog.result is None

    def test_a_cancel_keeps_the_finished_values_and_says_so(self, dialog, shown):
        dialog._requested = 8
        result = SweepResult(
            parameter="u_inflow",
            values=[0.05, 0.06],
            measurements=["reynolds_number"],
            data={"reynolds_number": [1.0, 2.0]},
            elapsed=1.0,
            cancelled=True,
        )
        dialog._on_result(result)
        assert dialog.result is result
        assert dialog._run_btn.isEnabled() is True
        assert "2/8" in dialog._est_label.text()
        assert shown.information and "2 of 8" in shown.information[0][1]

    def test_the_plot_still_shows_the_partial_values(self, dialog, shown):
        result = SweepResult(
            parameter="u_inflow",
            values=[0.05, 0.06],
            measurements=["reynolds_number"],
            data={"reynolds_number": [1.0, 2.0]},
            elapsed=1.0,
            cancelled=True,
        )
        dialog._on_result(result)
        xs, ys = dialog._curve.getData()
        assert list(xs) == [0.05, 0.06]
        assert list(ys) == [1.0, 2.0]

    def test_a_complete_sweep_is_not_announced_as_cancelled(self, dialog, shown):
        dialog._requested = 2
        dialog._on_result(
            SweepResult(
                parameter="u_inflow",
                values=[0.05, 0.06],
                measurements=["reynolds_number"],
                data={"reynolds_number": [1.0, 2.0]},
                elapsed=2.0,
            )
        )
        assert shown.information == []


class TestTheEstimateIsNotAFixedLie:
    def test_it_seeds_before_anything_has_run(self, dialog):
        dialog._num_steps.setValue(8)
        dialog._sim_steps.setValue(5000)
        dialog._values_done = 0
        dialog._elapsed = 0.0
        dialog._update_estimate()
        text = dialog._est_label.text()
        # 8 * 5000 steps measured 13.1s; the old formula claimed 0.4s.
        assert "estimated" in text
        assert float(text.split("~")[1].split("s")[0]) > 1.0

    def test_it_uses_the_measured_rate_once_there_is_one(self, dialog):
        dialog._num_steps.setValue(8)
        dialog._values_done = 2
        dialog._elapsed = 4.0  # 2 s/value
        dialog._update_estimate()
        assert "12" in dialog._est_label.text()  # 6 values left at 2 s each
        assert "remaining" in dialog._est_label.text()

    def test_it_shows_progress(self, dialog):
        dialog._num_steps.setValue(8)
        dialog._values_done = 2
        dialog._elapsed = 4.0
        dialog._update_estimate()
        assert "2/8" in dialog._est_label.text()

    def test_a_finished_run_refreshes_it(self, dialog):
        dialog._num_steps.setValue(4)
        dialog._sim_steps.setValue(300)
        dialog._on_result(
            SweepResult(
                parameter="u_inflow",
                values=[0.05, 0.06],
                measurements=["reynolds_number"],
                data={"reynolds_number": [1.0, 2.0]},
                elapsed=1.0,
            )
        )
        assert "remaining" in dialog._est_label.text()
