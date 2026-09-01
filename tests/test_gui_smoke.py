"""Offscreen GUI smoke tests: MainWindow lifecycle and Phase 2-5 wiring.

These tests construct the full workbench headlessly (QT_QPA_PLATFORM=offscreen)
and exercise the interactive code paths that the engine test suite never
touches: sidebar accordion, transport bar, visualize toggles, shortcuts,
and the pre-run sanity banner.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from engines.lbm2d import LBM2D

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    instance = QApplication.instance() or QApplication([])
    return instance


@pytest.fixture(scope="module")
def window(app):
    from workbench.app import MainWindow

    sim = LBM2D(width=64, height=64, viscosity=0.01)
    win = MainWindow(sim)
    win.timer.stop()
    win.show()
    app.processEvents()
    yield win
    win.close()


def test_mainwindow_constructs_and_shows(window):
    assert window.isVisible()
    assert window.step_count == 0


def test_sidebar_has_all_sections(window):
    assert list(window.sidebar._sections) == [
        "scene",
        "parameters",
        "visualize",
        "analysis",
        "outcome",
    ]
    window.sidebar.expand_section("outcome")
    assert window.sidebar.section("outcome").is_expanded()


def test_sidebar_sections_collapse_and_expand(window):
    for section in window.sidebar._sections.values():
        section.collapse()
        assert not section.is_expanded()
        section.expand()
        assert section.is_expanded()


def test_sanity_banner_present_at_step_0(window):
    assert window.sanity_banner.isVisible()
    assert window.sanity_banner.text()


def test_apply_refresh_and_reset(window):
    window._apply_and_refresh()
    window.reset(silent=True)
    assert window.step_count == 0
    assert window.timeline_label.text() == "Step 0"


def test_step_and_tick_advance_simulation(window, app):
    before = window.step_count
    window.step_once()
    window.tick()
    app.processEvents()
    assert window.step_count >= before + 1


def test_transport_bar_wiring(window):
    bar = window.transport_bar
    window.paused = False
    bar.set_play_state(False)
    assert bar.play_btn.text() == "Play"
    window.paused = True
    bar.set_play_state(True)
    assert bar.play_btn.text() == "Pause"
    window.toggle_pause()
    assert window.paused is False
    bar.set_fps(42.0)
    assert bar.fps_label.text() == "FPS: 42"


def test_speed_shift_updates_factor(window):
    window._shift_speed(+1)
    assert window._speed_factor > 1.0
    window._shift_speed(-1)
    assert window._speed_factor == 1.0


def test_visualize_checkboxes_toggle_viewport(window):
    window._on_vis_toggle("streamlines", 2)
    assert window.viewport._show_streamlines
    window._on_vis_toggle("streamlines", 0)
    assert not window.viewport._show_streamlines
    window._on_vis_toggle("hud", 0)
    assert not window.viewport._show_hud


def test_field_combo_cycles_colormap(window):
    first = window.field_combo.currentData()
    window._cycle_field()
    second = window.field_combo.currentData()
    assert first != second
    assert second in (
        window.field_combo.itemData(i) for i in range(window.field_combo.count())
    )


def test_draw_mode_shortcuts_switch_tools(window, app):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    vp = window.viewport
    vp.setFocus(Qt.FocusReason.ShortcutFocusReason)
    app.processEvents()
    key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_2, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(vp, key)
    app.processEvents()
    assert vp.draw_mode == "circle"


def test_expert_mode_toggles(window):
    window.set_expert_mode(True)
    assert window._expert_mode
    assert window.tool_strip.polygon_btn().isVisible()
    window.set_expert_mode(False)
    assert not window._expert_mode
    assert not window.tool_strip.polygon_btn().isVisible()


@pytest.mark.parametrize(
    "model_type",
    ["power_law", "carreau", "bingham"],
)
def test_every_non_newtonian_model_applies(window, model_type):
    """Each rheology preset must reach the engine without a TypeError.

    The Bingham branch used to call BinghamModel(yield_stress=...), which is
    not a parameter name, so selecting Bingham raised inside the Qt slot and
    left the engine silently on its previous collision operator.
    """
    from engines.non_newtonian import NonNewtonianCollision

    panel = window.scene_panel
    panel._non_newtonian_model_combo.setCurrentIndex(
        panel._non_newtonian_model_combo.findData(model_type)
    )
    panel._apply_non_newtonian_model()
    assert isinstance(window.sim.collision_op, NonNewtonianCollision)
