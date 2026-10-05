"""Undo history: reachable, reversible, and one scene.

PLAN.md C31/C34 (P0-5). Two defects, both about the snapshot being taken in the
wrong place or the wrong object:

  * `_PropEditor._set` mutated the dataclass and *then* emitted, so
    `_on_prop_changed` snapshotted the already-edited scene. Undoing the first
    property edit a user made was a no-op -- the edit was simply unreachable.
  * `_undo`/`_redo` restored by rebinding `self.scene = dict_to_scene(...)`. The
    window kept the object it already had, so after one undo the panel was editing
    a scene nothing else could see.

A third gap sat alongside them and is easy to miss: the four simulation-parameter
spinboxes write to `Scene` as well as to the engine, and pushed no snapshot at all,
so viscosity, inflow, smoke diffusion and smoke decay were not undoable. Fixing
that last one introduced a new failure -- `sync_params_from_scene()` drives the
*same* widgets with `setValue`, which emits `valueChanged`, so after an undo (where
the values genuinely differ) an unguarded sync would record a change nobody made
and push the restored history straight back off the stack.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from engines.lbm2d import LBM2D
from scene.scene import CircleObstacle, ProbeSpec, RectObstacle, Scene
from tests.conftest import dispose_window

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    from workbench.app import MainWindow

    win = MainWindow(LBM2D(width=48, height=48, viscosity=0.01))
    win.timer.stop()
    monkeypatch.setattr(win, "_prompt_poe_prediction", lambda *a, **kw: None)
    yield win
    dispose_window(win)


def _edit_name(panel, value: str) -> None:
    """Type into the name field of the current property editor."""
    editor = panel._current_editor
    assert editor is not None, "no property editor is open"
    editor._widgets["name"].setText(value)


class TestEveryEditIsReachable:
    """The snapshot must record the scene as it was *before* the mutation."""

    def test_a_property_edit_is_reachable(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(
            CircleObstacle(name="original", x=10, y=10, radius=4)
        )
        panel._rebuild_tree()
        panel._on_item_selected(panel.tree.topLevelItem(0).child(0), None)
        assert panel._current_editor is not None

        _edit_name(panel, "renamed")
        assert window.scene.obstacles[0].name == "renamed"

        panel._undo()
        assert window.scene.obstacles[0].name == "original", (
            "undoing the first property edit was a no-op: the snapshot was taken "
            "after the mutation, so it recorded the new value"
        )

    def test_a_numeric_property_edit_is_reachable(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(
            CircleObstacle(name="ball", x=10, y=10, radius=4)
        )
        panel._rebuild_tree()
        panel._on_item_selected(panel.tree.topLevelItem(0).child(0), None)
        radius = panel._current_editor._widgets["radius"]

        radius.setValue(12)
        assert window.scene.obstacles[0].radius == 12

        panel._undo()
        assert window.scene.obstacles[0].radius == 4

    def test_a_removal_is_reachable(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(RectObstacle(name="blk", x=4, y=4, w=8, h=8))
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0).child(0))
        panel._remove_selected()

        assert window.scene.obstacles == []
        panel._undo()
        assert [o.name for o in window.scene.obstacles] == ["blk"]

    def test_a_canvas_placed_probe_is_reachable(self, window):
        panel = window.scene_panel
        assert window.scene.probes == []
        panel.add_probe_from_viewport(ProbeSpec(name="p", x=5, y=5))
        assert len(window.scene.probes) == 1

        panel._undo()
        assert window.scene.probes == [], (
            "a probe placed on the canvas is not undoable; it used to be appended "
            "by the window, bypassing the panel's snapshot"
        )


class TestParameterEditsAreUndoable:
    """These four write to Scene as well as to the engine, so they are scene edits."""

    @pytest.mark.parametrize(
        "widget,attr,new_value",
        [
            ("_param_visc", "viscosity", 0.05),
            ("_param_u_inflow", "u_inflow", 0.08),
            ("_param_diff", "smoke_diffusion", 0.2),
            ("_param_decay", "smoke_decay", 0.9),
        ],
    )
    def test_edit_then_undo(self, window, widget, attr, new_value):
        panel = window.scene_panel
        before = getattr(window.scene, attr)

        getattr(panel, widget).setValue(new_value)
        assert getattr(window.scene, attr) == pytest.approx(new_value)

        panel._undo()
        assert getattr(window.scene, attr) == pytest.approx(before)

    def test_the_engine_follows_the_undo(self, window):
        """Undo restores the Scene; the engine parameter has to match or the next
        apply silently overwrites the restored value."""
        panel = window.scene_panel
        panel._param_visc.setValue(0.05)
        panel._undo()
        assert panel.sim.viscosity == pytest.approx(window.scene.viscosity)


class TestSyncDoesNotFabricateHistory:
    def test_a_load_time_sync_pushes_nothing(self, window):
        panel = window.scene_panel
        depth = len(panel._undo_stack)
        panel._param_visc.setValue(0.04)
        panel.sync_params_from_scene()
        assert len(panel._undo_stack) == depth + 1  # only the user edit

    def test_sync_after_undo_does_not_push_the_history_back(self, window):
        """The specific re-entrancy the guard exists for: after an undo the
        spinner values differ from the restored scene, so setValue emits
        valueChanged and an unguarded handler would push an entry -- and clear the
        redo stack -- for a change the user never made."""
        panel = window.scene_panel
        panel._param_visc.setValue(0.05)
        panel._undo()
        assert panel._param_visc.value() == pytest.approx(window.scene.viscosity)

        depth, redo_depth = len(panel._undo_stack), len(panel._redo_stack)
        panel.sync_params_from_scene()

        assert len(panel._undo_stack) == depth
        assert len(panel._redo_stack) == redo_depth, (
            "syncing the spinners after an undo cleared the redo stack, so the "
            "redo that would restore the undone edit is lost"
        )
        assert panel._param_visc.value() == pytest.approx(window.scene.viscosity)

    def test_the_guard_is_always_released(self, window):
        """`sync_params_from_scene` must not leave the panel permanently deaf to
        undo, even if a spinbox rejects a value."""
        panel = window.scene_panel
        panel.sync_params_from_scene()
        assert panel._syncing_params is False


class TestSceneIdentityIsPreserved:
    def test_undo_keeps_the_same_object(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=6, y=6, radius=3))
        before = window.scene

        panel._undo()

        assert window.scene is before, (
            "undo replaced the Scene object, so every other holder was left "
            "referring to a discarded scene"
        )
        assert panel.scene is before

    def test_redo_keeps_the_same_object(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=6, y=6, radius=3))
        before = window.scene
        panel._undo()
        panel._redo()

        assert window.scene is before
        assert len(window.scene.obstacles) == 1

    def test_every_holder_survives_a_full_undo_redo_cycle(self, window):
        holders = (
            window.viewport,
            window.scene_panel,
            window.analysis_panel,
            window.outcome_panel,
        )
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=6, y=6, radius=3))

        for _ in range(2):
            panel._undo()
            panel._redo()

        for holder in holders:
            assert holder.scene is window.scene

    def test_restore_clears_the_property_editor(self, window):
        """The editor holds a direct reference into the container list that
        `_restore` replaces, so leaving it open would have it edit a detached
        object."""
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=6, y=6, radius=3))
        panel._rebuild_tree()
        panel._on_item_selected(panel.tree.topLevelItem(0).child(0), None)
        assert panel._current_editor is not None

        panel._undo()
        assert panel._current_editor is None


class TestUndoRedoRoundTrip:
    def test_a_sequence_rewinds_and_replays(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="one", x=4, y=4, radius=2))
        panel.add_obstacle_from_viewport(RectObstacle(name="two", x=20, y=4, w=6, h=6))
        assert [o.name for o in window.scene.obstacles] == ["one", "two"]

        panel._undo()
        assert [o.name for o in window.scene.obstacles] == ["one"]
        panel._undo()
        assert window.scene.obstacles == []
        panel._redo()
        panel._redo()
        assert [o.name for o in window.scene.obstacles] == ["one", "two"]

    def test_a_new_edit_clears_the_redo_stack(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="one", x=4, y=4, radius=2))
        panel._undo()
        assert panel._redo_stack

        panel.add_obstacle_from_viewport(
            CircleObstacle(name="other", x=9, y=9, radius=2)
        )
        assert panel._redo_stack == []
        assert not panel.redo_btn.isEnabled()

    def test_undo_on_an_empty_stack_is_a_no_op(self, window):
        panel = window.scene_panel
        before = window.scene
        panel._undo()
        panel._redo()
        assert window.scene is before

    def test_the_history_is_bounded(self, window):
        panel = window.scene_panel
        for i in range(panel._max_history + 5):
            panel.add_obstacle_from_viewport(
                CircleObstacle(name=f"o{i}", x=2, y=2, radius=1)
            )
        assert len(panel._undo_stack) == panel._max_history


class TestButtonsReflectTheStacks:
    def test_undo_enables_then_disables(self, window):
        panel = window.scene_panel
        assert not panel.undo_btn.isEnabled()
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=4, y=4, radius=2))
        assert panel.undo_btn.isEnabled()
        panel._undo()
        assert not panel.undo_btn.isEnabled()
        assert panel.redo_btn.isEnabled()

    def test_redo_disables_when_the_stack_empties(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=4, y=4, radius=2))
        panel._undo()
        panel._redo()
        assert not panel.redo_btn.isEnabled()


class TestHistoryDoesNotCrossADocumentBoundary:
    """The stacks hold JSON snapshots of the document being replaced.

    Before C32 the Undo button stayed live across a load, and pressing it replaced
    the newly opened scene with the contents of whatever was open before — silently,
    with nothing in the title to show the document had changed underneath.
    """

    def test_file_new_drops_the_history(self, window):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=4, y=4, radius=2))
        assert panel._undo_stack

        window._file_new()

        assert panel._undo_stack == []
        assert panel._redo_stack == []
        assert not panel.undo_btn.isEnabled()
        assert not panel.redo_btn.isEnabled()

    def test_a_load_cannot_undo_into_the_previous_document(self, window, tmp_path):
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(
            CircleObstacle(name="ghost", x=4, y=4, radius=2)
        )
        panel.add_obstacle_from_viewport(
            CircleObstacle(name="ghost2", x=9, y=9, radius=2)
        )

        from PySide6.QtWidgets import QFileDialog

        path = tmp_path / "other.json"
        path.write_text(json.dumps({"name": "other", "width": 32, "height": 32}))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                QFileDialog,
                "getOpenFileName",
                staticmethod(lambda *a, **kw: (str(path), "")),
            )
            window._file_open()

        panel._undo()

        assert window.scene.name == "other", (
            "undo reached across a document boundary and replaced the scene that "
            "was just opened"
        )
        assert window.scene.obstacles == []

    def test_a_preset_load_drops_the_history(self, window):
        from presets.loader import list_presets

        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=4, y=4, radius=2))

        preset = next(p for p in list_presets() if p["id"] == "lid_driven_cavity")
        window._load_preset_file(preset["file"])

        assert panel._undo_stack == []
        assert panel.scene is window.scene
        assert window.scene.name == preset["name"]

    def test_a_wizard_template_drops_the_history(self, window):
        from workbench.app import WizardTemplate

        panel = window.scene_panel
        panel.add_obstacle_from_viewport(CircleObstacle(name="a", x=4, y=4, radius=2))

        window._on_wizard_template(
            WizardTemplate(
                name="t",
                category="c",
                description="d",
                icon="",
                scene=Scene(name="wizard", width=32, height=32),
            )
        )

        assert panel._undo_stack == []
        assert panel.scene is window.scene

    def test_a_reset_keeps_the_history(self, window):
        """A reset re-seeds the *same* document. Clearing its history would make the
        Reset button silently discard the user's edits, which is a different and
        much worse bug than the one C32 fixed."""
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(
            CircleObstacle(name="keepme", x=4, y=4, radius=2)
        )
        depth = len(panel._undo_stack)

        window.reset(silent=True)

        assert len(panel._undo_stack) == depth
        assert panel.undo_btn.isEnabled()
        panel._undo()
        assert window.scene.obstacles == []

    def test_an_engine_switch_keeps_the_history(self, window):
        """Same document, different solver: the edit history is still meaningful."""
        panel = window.scene_panel
        panel.add_obstacle_from_viewport(
            CircleObstacle(name="keepme", x=4, y=4, radius=2)
        )
        depth = len(panel._undo_stack)

        window._set_physics_mode(1)

        assert len(panel._undo_stack) == depth

    def test_edits_after_a_load_are_undoable_again(self, window):
        panel = window.scene_panel
        window._file_new()
        panel.add_obstacle_from_viewport(
            CircleObstacle(name="fresh", x=4, y=4, radius=2)
        )

        assert len(panel._undo_stack) == 1
        panel._undo()
        assert window.scene.obstacles == []


class TestInPlaceRestoreHelper:
    """`update_scene_in_place` is the contract the undo path depends on."""

    @staticmethod
    def _roundtrip(scene):
        from scene.serializer import scene_to_dict, update_scene_in_place

        return update_scene_in_place(scene, scene_to_dict(scene))

    def test_identity_is_preserved(self):
        from scene.serializer import scene_to_dict, update_scene_in_place

        scene = Scene(name="s", width=32, height=32)
        assert update_scene_in_place(scene, scene_to_dict(scene)) is scene
        assert scene.name == "s"

    def test_every_field_is_restored(self):
        from scene.serializer import scene_to_dict, update_scene_in_place

        scene = Scene(name="s", width=32, height=32, viscosity=0.03)
        original = scene_to_dict(scene)
        scene.name = "changed"
        scene.viscosity = 0.09
        scene.obstacles.append(CircleObstacle(name="x", x=1, y=1, radius=1))

        update_scene_in_place(scene, original)

        assert scene.name == "s"
        assert scene.viscosity == pytest.approx(0.03)
        assert scene.obstacles == []

    def test_nested_dataclasses_are_replaced_not_shared(self):
        """Product metadata is a dataclass of its own; restoring it must not leave
        the scene pointing at the temporary object's child."""
        from scene.serializer import scene_to_dict, update_scene_in_place

        scene = Scene(name="s", width=32, height=32)
        scene.product.recommended_colormap = "vorticity"
        update_scene_in_place(scene, scene_to_dict(scene))
        assert scene.product.recommended_colormap == "vorticity"

    def test_a_snapshot_missing_a_field_resets_it(self):
        """An older snapshot must not leave a newer field holding a stale value."""
        from scene.serializer import update_scene_in_place

        scene = Scene(name="s", width=32, height=32, smoke_decay=0.5)
        update_scene_in_place(scene, {"name": "s", "width": 32, "height": 32})
        assert scene.smoke_decay == pytest.approx(Scene().smoke_decay)
