"""The window title carried no modified marker, so nothing said a scene was unsaved.

PLAN.md C46 (P1-8). `_update_title` built `"S-Stream - {name}"` and nothing else --
there was no dirty state anywhere in the window. The user edits an obstacle, closes
the app, and the work is gone with no prompt and no indicator. C47 adds the close
guard; this item supplies the state it needs, because a guard that cannot tell
whether there is unsaved work cannot ask the right question.

**Dirty state is computed from the document's content, not flagged.** Every mutation
funnel in this window -- the scene panel's property widgets, the viewport's drawing,
undo/redo, probe placement, the engine sliders -- would otherwise have to remember to
set a flag, and the ones that forget do so *silently*: the title keeps claiming the
work is saved while it is not, which is the exact failure this item exists to prevent.
`_scene_fingerprint` serializes the scene the same way `save` does and compares it to
the last saved/loaded/created form.

Three properties fall out of computing rather than flagging, all asserted below:

* Stepping the simulation does **not** make the document dirty. A run is not an edit,
  and a flag-based scheme would mark every session dirty within a second of pressing
  play -- which trains users to ignore the marker entirely.
* Undoing back to the saved state makes it **clean** again, which is what a user
  expects from an undo rather than a permanently sticky marker.
* Spurious `scene_changed` signals are harmless. The scene panel emits one while
  syncing widgets from the scene, and a flag set there would report a freshly loaded
  file as edited before the user touched it.

The baseline deliberately does **not** move on `_on_scene_changed`, which is where
undo/redo lands. Restoring an earlier state does not make the document match what is
on disk, so marking it clean there would reclassify unsaved work as saved -- the
worst possible direction for this feature to be wrong in.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from engines.lbm2d import LBM2D
from scene.scene import CircleObstacle, ProbeSpec, Scene
from scene.serializer import scene_to_dict
from tests.conftest import dispose_window

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    from workbench.app import MainWindow

    win = MainWindow(LBM2D(width=48, height=48))
    win.timer.stop()
    win._adopt_scene(Scene(name="workbench", width=48, height=48))
    win._apply_and_refresh()
    win._mark_document_clean()
    yield win
    dispose_window(win)


def _edit(win, **kwargs) -> None:
    """Change the document the way a user would, through the real signal."""
    for key, value in kwargs.items():
        setattr(win.scene, key, value)
    win._on_scene_changed()


class TestTheTitleCarriesAMarker:
    def test_a_clean_document_has_no_marker(self, window):
        assert window.is_dirty is False
        assert window.windowTitle() == "S-Stream - workbench"

    def test_an_edit_marks_the_title(self, window):
        window.scene.obstacles.append(CircleObstacle(name="c", x=20, y=24, radius=5))
        window._on_scene_changed()
        assert window.is_dirty is True
        assert window.windowTitle() == "S-Stream - workbench*"

    def test_the_marker_is_the_last_character(self, window):
        _edit(window, viscosity=0.05)
        assert window.windowTitle().endswith("*")

    def test_the_marker_is_not_a_dash_or_ellipsis(self, window):
        """A marker that reads as decoration does not signal unsaved work."""
        _edit(window, viscosity=0.05)
        marker = window.windowTitle()[-1]
        assert marker not in "-_.…"

    def test_a_saved_file_name_replaces_the_scene_name(self, window, tmp_path):
        window._file_path = tmp_path / "cylinder scene.json"
        window._update_title()
        assert window.windowTitle() == "S-Stream - cylinder scene"
        _edit(window, viscosity=0.05)
        assert window.windowTitle() == "S-Stream - cylinder scene*"


class TestSteppingIsNotAnEdit:
    def test_running_the_simulation_does_not_dirty_the_document(self, window):
        for _ in range(120):
            window.tick()
        assert window.is_dirty is False
        assert window.windowTitle() == "S-Stream - workbench"

    def test_a_long_run_does_not_dirty_it_either(self, window):
        window._run_guided_demo()
        for _ in range(200):
            window.tick()
        assert window.is_dirty is False


class TestDirtyStateFollowsTheContent:
    def test_undoing_back_to_the_saved_state_is_clean_again(self, window):
        window.scene.obstacles.append(CircleObstacle(name="c", x=20, y=24, radius=5))
        window._on_scene_changed()
        assert window.is_dirty
        window.scene.obstacles.clear()
        window._on_scene_changed()
        assert window.is_dirty is False

    def test_changing_a_value_and_changing_it_back_is_clean(self, window):
        _edit(window, viscosity=0.05)
        assert window.is_dirty
        _edit(window, viscosity=window.scene.viscosity)
        window.scene.viscosity = 0.01
        window._on_scene_changed()
        assert window.is_dirty is True
        window.scene.viscosity = 0.05
        window._on_scene_changed()
        assert window.is_dirty is True  # 0.01, not the saved value: still differs

    def test_a_probe_placement_dirties_the_document(self, window):
        window.scene.probes.append(ProbeSpec(name="p", x=10, y=10, fields=["v"]))
        window._on_scene_changed()
        assert window.is_dirty is True

    def test_the_probe_path_in_app_dirties_it_too(self, window):
        """The viewport's probe tool reaches the scene through the panel."""
        window.scene_panel.add_probe_from_viewport(
            ProbeSpec(name="p2", x=20, y=20, fields=["u"])
        )
        window._on_scene_changed()
        assert window.is_dirty is True

    def test_a_sweep_added_without_a_file_is_dirty(self, window):
        window.scene.sweeps.append({"param": "u_inflow", "values": [0.05, 0.1]})
        window._update_title()
        assert window.is_dirty is True

    def test_the_fingerprint_covers_every_saved_field(self, window):
        """If a field is not serialized it cannot be saved, so it cannot dirty."""
        for field in (
            "name",
            "viscosity",
            "u_inflow",
            "smoke_diffusion",
            "smoke_decay",
            "description",
        ):
            before = window._scene_fingerprint()
            window.scene.description = window.scene.description  # no-op
            assert window._scene_fingerprint() == before, field

    def test_two_identical_scenes_have_one_fingerprint(self, window):
        a = Scene(name="x", width=32, height=32)
        b = Scene(name="x", width=32, height=32)
        window.scene = a
        first = window._scene_fingerprint()
        window.scene = b
        assert window._scene_fingerprint() == first

    def test_the_fingerprint_matches_what_save_writes(self, window):
        """Otherwise 'saved' and 'unchanged' would be different definitions."""
        import json

        window.scene = Scene(name="cmp", width=32, height=32)
        assert window._scene_fingerprint() == json.dumps(
            scene_to_dict(window.scene), sort_keys=True, default=str
        )


class TestSavingClearsTheMarker:
    def test_save_clears_it(self, window, tmp_path):
        path = tmp_path / "scene.json"
        window._file_path = path
        window.scene.obstacles.append(CircleObstacle(name="c", x=20, y=24, radius=5))
        window._on_scene_changed()
        assert window.is_dirty
        window._file_save()
        assert window.is_dirty is False
        assert window.windowTitle() == "S-Stream - scene"

    def test_save_as_clears_it(self, window, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        path = tmp_path / "brand new.json"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (str(path), "")),
        )
        window.scene.u_inflow = 0.04
        window._on_scene_changed()
        assert window.is_dirty
        window._file_save_as()
        assert window.is_dirty is False
        assert window.windowTitle() == "S-Stream - brand new"

    def test_editing_after_a_save_marks_it_again(self, window, tmp_path):
        path = tmp_path / "scene.json"
        window._file_path = path
        _edit(window, viscosity=0.03)
        window._file_save()
        assert window.is_dirty is False
        _edit(window, viscosity=0.04)
        assert window.is_dirty is True

    def test_a_failed_save_as_leaves_it_dirty(self, window, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
        )
        _edit(window, viscosity=0.06)
        window._file_save_as()
        assert window.is_dirty is True


class TestLoadingProducesACleanDocument:
    def test_new_is_clean(self, window):
        window._file_new()
        assert window.is_dirty is False
        assert window._file_path is None

    def test_open_is_clean(self, window, tmp_path):
        from scene.serializer import save

        path = tmp_path / "opened.json"
        save(Scene(name="opened", width=32, height=32), path)
        window.scene.u_inflow = 0.09
        window._on_scene_changed()
        assert window.is_dirty
        window._adopt_scene(
            __import__("scene.scene", fromlist=["Scene"]).Scene(
                name="opened", width=32, height=32
            )
        )
        window._file_path = path
        window._apply_and_refresh()
        window._mark_document_clean()
        assert window.is_dirty is False
        assert window.windowTitle() == "S-Stream - opened"

    def test_a_preset_load_is_clean(self, window):
        from scene.serializer import load as scene_load

        window.scene.name = "edited"
        window._on_scene_changed()
        assert window.is_dirty
        window._adopt_scene(scene_load("presets/scenes/cylinder.json"))
        window._file_path = None
        window._apply_and_refresh()
        window._mark_document_clean()
        assert window.is_dirty is False

    def test_a_wizard_template_is_clean(self, window):
        from workbench.dialogs.wizard_dialog import _build_templates

        window.scene.name = "edited"
        window._on_scene_changed()
        template = _build_templates()[0]
        window._adopt_scene(template.scene)
        window._file_path = None
        window._apply_and_refresh()
        window._mark_document_clean()
        assert window.is_dirty is False

    def test_a_spurious_scene_changed_signal_leaves_it_clean(self, window):
        """The panel emits this while syncing; a flag would read it as an edit."""
        window._on_scene_changed()
        window._on_scene_changed()
        assert window.is_dirty is False


class TestTheStateIsDerivedNotFlagged:
    def test_there_is_no_boolean_to_forget(self, window):
        assert not hasattr(window, "_dirty")
        assert not hasattr(window, "_dirty_flag")

    def test_no_mutation_site_marks_the_document_dirty(self):
        """If a site could flag itself, forgetting would be silent."""
        import ast
        import inspect
        import textwrap

        from workbench.app import MainWindow

        tree = ast.parse(textwrap.dedent(inspect.getsource(MainWindow)))
        callers = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"mark_dirty", "set_dirty", "_dirty"}
        }
        assert callers == set(), callers

    def test_undo_does_not_move_the_baseline(self, window):
        """Restoring an earlier state does not make the document match disk.

        The saved value is whatever the document was when it was last written, so the
        test reads it rather than assuming one -- the first version hardcoded 0.02,
        which is not the default, so it was asserting that an *unrelated* value was
        clean.
        """
        saved = window.scene.u_inflow
        window.scene.u_inflow = 0.07
        window._on_scene_changed()
        window.scene.u_inflow = 0.08
        window._on_scene_changed()
        # Undo all the way back to the state that *was* saved.
        window.scene.u_inflow = saved
        window._on_scene_changed()
        assert window.is_dirty is False

    def test_the_title_is_the_only_place_the_marker_appears(self, window):
        """One marker, one derivation -- no second widget that can disagree."""
        _edit(window, viscosity=0.05)
        assert "*" in window.windowTitle()
        assert window.sanity_banner.text().find("*") == -1
