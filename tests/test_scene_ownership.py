"""One owner for Scene identity.

PLAN.md C30 (P0-5). Five objects hold a reference to the `Scene`: the window and
the viewport, scene panel, analysis panel and outcome panel. The panel restored
undo history by rebinding `self.scene = dict_to_scene(...)`, and the window never
noticed, so after a single undo the panel was editing an object the rest of the
app could not see -- viewport, probes, save and export all reading the pre-undo
document. Nothing raised: the next edit was applied to a scene that was silently
on its way out.

These tests pin the ownership invariant and, just as importantly, that the fix is
not "the window copies the panel's reference and lives with it": a Scene must
enter the app in one place, and every load path must go through it.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from engines.lbm2d import LBM2D
from scene.scene import CircleObstacle, Scene

pytest.importorskip("PySide6")

APP_PY = Path(__file__).resolve().parents[1] / "workbench" / "app.py"

LOAD_PATHS = ("_file_new", "_file_open", "_load_preset_file", "_on_wizard_template")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    from workbench.app import MainWindow

    win = MainWindow(LBM2D(width=48, height=48, viscosity=0.01))
    win.timer.stop()
    # The wizard path shows a modal prediction prompt; nothing here is testing it.
    monkeypatch.setattr(win, "_prompt_poe_prediction", lambda *a, **kw: None)
    yield win
    win.close()


def _holders(win):
    return {
        "viewport": win.viewport,
        "scene_panel": win.scene_panel,
        "analysis_panel": win.analysis_panel,
        "outcome_panel": win.outcome_panel,
    }


class TestEveryHolderSharesOneScene:
    def test_after_construction(self, window):
        for name, holder in _holders(window).items():
            assert holder.scene is window.scene, f"{name} holds a different Scene"

    def test_after_file_new(self, window):
        window._file_new()
        for name, holder in _holders(window).items():
            assert holder.scene is window.scene, f"{name} diverged after File > New"

    def test_after_file_open(self, window, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        path = tmp_path / "scene.json"
        path.write_text(json.dumps({"name": "from-disk", "width": 32, "height": 32}))
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **kw: (str(path), "")),
        )
        window._file_open()
        assert window.scene.name == "from-disk"
        for name, holder in _holders(window).items():
            assert holder.scene is window.scene, f"{name} diverged after File > Open"

    def test_after_a_wizard_template(self, window):
        from workbench.app import WizardTemplate

        window._on_wizard_template(
            WizardTemplate(
                name="t",
                category="c",
                description="d",
                icon="",
                scene=Scene(name="wizard", width=32, height=32),
            )
        )
        for name, holder in _holders(window).items():
            assert holder.scene is window.scene, f"{name} diverged after the wizard"

    def test_after_a_panel_initiated_undo(self, window):
        """The defect itself.

        Undo rebinds the panel's own reference. Before the fix the window kept the
        object it already had, so the panel went on editing a scene the viewport,
        probes, save and export could not see -- and nothing raised.
        """
        window.scene_panel.add_obstacle_from_viewport(
            CircleObstacle(name="body", x=10, y=10, radius=4)
        )
        window.scene_panel._undo()

        for name, holder in _holders(window).items():
            assert holder.scene is window.scene, f"{name} diverged after undo"

    def test_a_divergent_panel_is_corrected_not_merely_read(self, window):
        """Force the divergence the old code produced, then check the window's
        reaction rather than trusting that no path can cause it."""
        orphan = Scene(name="orphan", width=32, height=32)
        window.scene_panel.scene = orphan

        window._on_scene_changed()

        assert window.scene is orphan
        for name, holder in _holders(window).items():
            assert holder.scene is orphan, f"{name} did not follow the adopted scene"

    def test_an_ordinary_edit_keeps_every_holder_agreed(self, window):
        window.scene_panel.add_obstacle_from_viewport(
            CircleObstacle(name="body", x=8, y=8, radius=3)
        )
        for name, holder in _holders(window).items():
            assert holder.scene is window.scene, f"{name} diverged after an add"


class TestAdoptionIsASingleEntryPoint:
    def test_adopt_scene_exists(self):
        from workbench.app import MainWindow

        assert callable(MainWindow._adopt_scene)

    @pytest.mark.parametrize("name", LOAD_PATHS)
    def test_no_load_path_assigns_scene_directly(self, name):
        """A load path that assigns `self.scene` and then applies would leave the
        holders reading the document it just discarded."""
        from workbench.app import MainWindow

        src = textwrap.dedent(inspect.getsource(getattr(MainWindow, name)))
        assert "self.scene =" not in src, (
            f"{name} assigns self.scene directly; route it through _adopt_scene "
            "so the holders are re-pointed"
        )
        assert "_adopt_scene(" in src, f"{name} does not go through _adopt_scene"

    def test_scene_is_written_only_by_the_owner(self):
        """AST check over app.py: `self.scene` is assigned in three places -- the
        initial value, `_adopt_scene`, and the divergence guard. A fourth writer
        would be a second source of truth."""
        tree = ast.parse(APP_PY.read_text())
        writers = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for sub in ast.walk(node):
                # Both `self.scene = x` and `self.scene: Scene = x` (the initial
                # value in __init__) are assignments of the same field.
                targets = (
                    sub.targets
                    if isinstance(sub, ast.Assign)
                    else [sub.target] if isinstance(sub, ast.AnnAssign) else []
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "scene"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        writers.add(node.name)
        assert writers == {"__init__", "_adopt_scene", "_on_scene_changed"}, (
            f"self.scene is written in {sorted(writers)}; new writers must go "
            "through _adopt_scene"
        )

    def test_share_scene_covers_every_holder(self):
        from workbench.app import MainWindow

        src = inspect.getsource(MainWindow._share_scene)
        for name in _holders_names():
            assert (
                f"self.{name}.set_scene(self.scene)" in src
            ), f"_share_scene does not re-point {name}"


def _holders_names():
    return ("viewport", "scene_panel", "analysis_panel", "outcome_panel")


class TestPropertyEditorCannotOutliveItsScene:
    @staticmethod
    def _select_first_obstacle(win):
        """Open the property editor on the first obstacle in the tree."""
        panel = win.scene_panel
        panel._rebuild_tree()
        top = panel.tree.topLevelItem(0)
        child = top.child(0)
        panel.tree.setCurrentItem(child)
        panel._on_item_selected(child, None)
        assert panel._current_editor is not None
        return panel._current_editor

    def test_set_scene_clears_the_property_editor(self, window):
        """The editor holds a direct reference to the dataclass it edits. Left
        alive across an adoption, the next keystroke writes into a discarded
        object."""
        window.scene_panel.add_obstacle_from_viewport(
            CircleObstacle(name="body", x=8, y=8, radius=3)
        )
        editor = self._select_first_obstacle(window)
        target = editor._obj

        window.scene_panel.set_scene(Scene(name="fresh", width=32, height=32))

        assert window.scene_panel._current_editor is None, (
            "set_scene kept the property editor alive, so it still points at "
            f"{target!r} -- an obstacle of a discarded Scene"
        )

    def test_the_editor_is_never_left_editing_a_foreign_object(self, window):
        """The stronger statement: after any adoption, no live editor target
        belongs to the previous scene."""
        window.scene_panel.add_obstacle_from_viewport(
            CircleObstacle(name="body", x=8, y=8, radius=3)
        )
        self._select_first_obstacle(window)
        old_scene = window.scene

        window._adopt_scene(Scene(name="fresh", width=32, height=32))

        editor = window.scene_panel._current_editor
        if editor is not None:
            current = window.scene
            candidates = [
                current,
                current.product,
                *current.obstacles,
                *current.emitters,
                *current.probes,
            ]
            assert any(editor._obj is obj for obj in candidates), (
                "the property editor survived adoption and still targets an "
                f"object owned by the previous scene {old_scene.name!r}"
            )

    def test_the_panel_never_creates_a_scene(self, window):
        """`default_scene()` in the panel would be a second owner."""
        src = inspect.getsource(type(window.scene_panel))
        assert (
            "default_scene()" not in src
        ), "ScenePanel constructs a Scene; MainWindow is the sole owner"
