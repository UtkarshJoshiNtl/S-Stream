"""The guided demo's progress bar was configured and then never driven.

PLAN.md C45 (P1-3), the second half of P1-3 after C44. `OutcomePanel` creates a
`QProgressBar`, `set_demo_target` sets its maximum and its visibility -- and
`setValue()` is never called anywhere in the codebase. Measured on the cylinder
preset (autorun_steps = 5000), stepping the whole demo:

    after start   value = -1   minimum = (unset)
    after 120     value = -1
    after 5000    value = -1   (demo_running = False, button back to "Run Demo")

**-1, not 0.** A `QProgressBar` with no minimum set reports its value as -1, which
Qt draws as the *indeterminate* busy indicator. So the widget configured to show how
far a guided run had got spent the entire run as a spinner: not a wrong number, but
the one value that means "this control cannot tell you anything". A test asserting
`value == 0` would have passed against the broken code.

Two more things the bar's absence hid. It never reached its target, so there was no
visual confirmation that the demo had finished -- the only signal was the toolbar
button's label flipping back. And the format was the bare `%p%`, so even once driven
it would have shown a percentage of a step count rather than the step count a user
deciding whether to wait longer needs.

**The toolbar label was the second half.** `demo_btn.setText("Running...")` was
written at demo start and cleared when the target was reached, but `toggle_pause` never
touched it: a paused demo still read "Running...". It cannot be fixed at the pause
site either, because pausing also happens implicitly when the timeline is grabbed for
scrubbing. The label is now derived from `_demo_running and not paused` in one helper,
so every path that changes the state updates it.

The bar is driven *before* the panel's 15-step cadence gate, not after: it is the one
widget in the panel that is expected to look continuous, and one `setValue` per frame
is not worth a second throttle.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from engines.lbm2d import LBM2D
from scene.scene import SceneProductMeta
from scene.serializer import load
from tests.conftest import dispose_window

pytest.importorskip("PySide6")

CYLINDER = "presets/scenes/cylinder.json"
SHORT_TARGET = 120


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    from workbench.app import MainWindow

    win = MainWindow(LBM2D(width=96, height=48))
    win.timer.stop()
    win._adopt_scene(load(CYLINDER))
    win._apply_and_refresh()
    yield win
    dispose_window(win)


@pytest.fixture
def panel(window):
    return window.outcome_panel


@pytest.fixture
def short_demo(window):
    """The real demo path, with the preset's 5000 steps shortened.

    Running the cylinder preset's genuine autorun_steps twice cost 100 seconds of a
    96x48 CPU grid and told us nothing the shortened run does not: the code path is
    identical, only the number is smaller. The preset's own target is still asserted
    in `test_the_target_comes_from_the_preset`.
    """
    window.scene.product.autorun_steps = SHORT_TARGET
    window._demo_target = SHORT_TARGET
    window.outcome_panel.set_demo_target(SHORT_TARGET)
    return window, window.outcome_panel


class TestTheBarTracksTheRealStepCount:
    def test_the_bar_has_a_real_minimum(self, panel):
        panel.set_demo_target(500)
        assert panel.progress.minimum() == 0
        assert panel.progress.value() >= 0

    def test_it_is_not_left_in_the_indeterminate_state(self, panel):
        """-1 is Qt's "cannot tell" value, and it is what the bar used to report."""
        panel.set_demo_target(500)
        assert panel.progress.value() != -1

    def test_it_advances_with_the_step_count(self, window, panel):
        panel.set_demo_target(5000)
        window.sim.step()
        window.step_count += 1
        panel.update_outcome(window.step_count)
        window.sim.step()
        window.step_count += 1
        panel.update_outcome(window.step_count)
        assert panel.progress.value() == window.step_count

    def test_it_updates_on_every_step_not_every_fifteenth(self, panel):
        """The rest of the panel is throttled to 15 steps; this is not."""
        panel.set_demo_target(5000)
        seen = []
        for step in range(1, 6):
            panel.update_outcome(step)
            seen.append(panel.progress.value())
        assert seen == [1, 2, 3, 4, 5]

    def test_it_reaches_the_target_exactly(self, panel):
        panel.set_demo_target(300)
        for step in range(1, 301):
            panel.update_outcome(step)
        assert panel.progress.value() == 300

    def test_it_clamps_past_the_target(self, panel):
        """Stepping beyond the demo by hand should show a full bar, not 512/500."""
        panel.set_demo_target(500)
        panel.update_outcome(512)
        assert panel.progress.value() == 500

    def test_it_reports_real_steps_rather_than_a_percentage(self, panel):
        panel.set_demo_target(500)
        assert "%v" in panel.progress.format()
        assert "%m" in panel.progress.format()

    def test_no_target_means_nothing_is_shown(self, panel):
        panel.set_demo_target(0)
        assert panel.progress.isHidden()
        panel.update_outcome(4000)
        assert panel.progress.isHidden()

    def test_a_target_means_the_bar_is_shown(self, panel):
        panel.set_demo_target(5000)
        assert not panel.progress.isHidden()


class TestTheGuidedDemoEndToEnd:
    def test_running_the_demo_moves_the_bar_from_zero(self, short_demo):
        window, panel = short_demo
        window._run_guided_demo()
        assert panel.progress.value() == 0
        for _ in range(60):
            window.tick()
        assert panel.progress.value() == window.step_count
        assert panel.progress.value() > 0

    def test_the_bar_is_full_when_the_demo_completes(self, short_demo):
        window, panel = short_demo
        window._run_guided_demo()
        while window._demo_running:
            window.tick()
        assert window.step_count == SHORT_TARGET
        assert panel.progress.value() == panel.progress.maximum() == SHORT_TARGET

    def test_the_bar_restarts_after_a_reset(self, short_demo):
        window, panel = short_demo
        window._run_guided_demo()
        for _ in range(50):
            window.tick()
        assert panel.progress.value() == 50
        window._reset_runtime_state()
        window._apply_and_refresh()
        panel.set_demo_target(window.scene.product.autorun_steps)
        assert panel.progress.value() == 0

    def test_a_scene_without_autorun_steps_has_no_bar(self, app):
        from scene.scene import Scene
        from workbench.app import MainWindow

        win = MainWindow(LBM2D(width=48, height=48))
        win.timer.stop()
        try:
            scene = Scene(name="bare", width=48, height=48)
            win._adopt_scene(scene)
            win._apply_and_refresh()
            assert win.outcome_panel.demo_target == 0
            assert win.outcome_panel.progress.isHidden()
        finally:
            dispose_window(win)

    def test_the_target_comes_from_the_preset(self, window, panel):
        assert panel.demo_target == window.scene.product.autorun_steps
        assert panel.progress.maximum() == window.scene.product.autorun_steps


class TestTheToolbarLabelTellsTheTruth:
    def test_a_paused_demo_does_not_say_running(self, window):
        window._run_guided_demo()
        assert window.demo_btn.text() == "Running..."
        window.toggle_pause()
        assert window.paused
        assert window.demo_btn.text() == "Run Demo"
        window.toggle_pause()
        assert window.demo_btn.text() == "Running..."

    def test_grabbing_the_timeline_also_clears_the_label(self, short_demo):
        window, _panel = short_demo
        window._run_guided_demo()
        for _ in range(30):
            window.tick()
        window._on_timeline_pressed()
        assert window.paused
        assert window.demo_btn.text() == "Run Demo"

    def test_the_label_is_cleared_when_the_demo_completes(self, short_demo):
        window, _panel = short_demo
        window._run_guided_demo()
        while window._demo_running:
            window.tick()
        assert window.demo_btn.text() == "Run Demo"

    def test_a_reset_clears_the_label(self, window):
        window._run_guided_demo()
        assert window.demo_btn.text() == "Running..."
        window._reset_runtime_state()
        assert window.demo_btn.text() == "Run Demo"

    def test_pausing_does_not_stop_the_demo_target_itself(self, short_demo):
        """Pause is a pause: the bar must not advance, resuming must not reset it."""
        window, panel = short_demo
        window._run_guided_demo()
        for _ in range(50):
            window.tick()
        window.toggle_pause()
        for _ in range(25):
            window.tick()
        assert panel.progress.value() == window.step_count == 50
        window.toggle_pause()
        window.tick()
        assert panel.progress.value() == window.step_count == 51


class TestTheBarAndThePanelAgree:
    def test_the_panel_step_count_is_the_source(self, window, panel):
        """The bar must not hold a step count of its own that can drift."""
        window._run_guided_demo()
        for _ in range(90):
            window.tick()
        assert panel.progress.value() == window.step_count

    def test_setting_the_target_snapshots_the_current_step(self, panel):
        """Re-targeting mid-run must not show a stale zero."""
        panel.update_outcome(742)
        panel.set_demo_target(1000)
        assert panel.progress.value() == 742

    def test_a_non_demo_product_still_drives_the_value(self, app):
        """The bar is driven by the step count, not by the demo flag."""
        from scene.scene import Scene
        from workbench.app import MainWindow

        win = MainWindow(LBM2D(width=48, height=48))
        win.timer.stop()
        try:
            win._adopt_scene(Scene(name="bare", width=48, height=48))
            win._apply_and_refresh()
            win.outcome_panel.set_demo_target(600)
            for _ in range(30):
                win.tick()
            assert win.outcome_panel.progress.value() == 30
        finally:
            dispose_window(win)


def _demo_btn_writers() -> set[str]:
    """The names of every MainWindow method that writes the demo button's text."""
    import ast
    import inspect
    import textwrap

    from workbench.app import MainWindow

    tree = ast.parse(textwrap.dedent(inspect.getsource(MainWindow)))
    owners: dict[str, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "setText"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "demo_btn"
            ):
                owners[ast.unparse(node)] = self.stack[-1]
            self.generic_visit(node)

    Visitor().visit(tree)
    return set(owners.values())


class TestTheSourceOfTruthIsOnePlace:
    def test_the_label_is_written_in_exactly_one_method(self):
        """A `setText` on the demo button anywhere else re-opens the same bug.

        Located by AST, so the one legitimate write -- inside `_sync_demo_button`
        itself -- is *identified* rather than filtered out by matching its text. The
        first version of this test grepped for `demo_btn.setText`, which meant it also
        matched the fix it was written to require, and so could only ever fail.
        """
        assert _demo_btn_writers() == {"_sync_demo_button"}

    def test_the_button_state_is_derived_from_pause_and_demo(self, window):
        window._demo_running = True
        window.paused = False
        window._sync_demo_button()
        assert window.demo_btn.text() == "Running..."
        window.paused = True
        window._sync_demo_button()
        assert window.demo_btn.text() == "Run Demo"
        window.paused = False
        window._demo_running = False
        window._sync_demo_button()
        assert window.demo_btn.text() == "Run Demo"


def test_the_product_still_defaults_to_no_autorun():
    """Nothing about the fix assumes every preset has an autorun target."""
    assert SceneProductMeta().autorun_steps == 0
