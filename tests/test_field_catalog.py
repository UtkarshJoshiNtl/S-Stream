"""The field catalog must come from the engine, not a hand-written list.

`workbench/app.py` used to carry a module-level `_COLORMAPS` list of all nine
field names in the project and offer every one of them on every engine. On a
plain `LBM2D` that meant the Field combo advertised `phase`, `component1`,
`component2` and `color` -- none of which that engine can produce. Selecting one
silently rendered something else (see C19/C20).

The contract now: the UI shows exactly `SimEngine.get_field_names()`.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from resources.colormaps import FIELD_REGISTRY, MODE_TO_CMAP
from workbench.app import _DEFAULT_FIELD, MainWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    """QPainter onto a QImage needs a QApplication, even offscreen."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_APP_PY = Path(__file__).resolve().parents[1] / "workbench" / "app.py"

_ENGINES = ["LBM2D", "LBM2DLiquid", "LBM2DMultiComponent"]
_MULTIPHASE_ONLY = ["phase", "component1", "component2", "color"]


def _sim(name: str):
    if name == "LBM2D":
        return LBM2D(48, 48)
    if name == "LBM2DLiquid":
        return LBM2DLiquid(48, 48)
    if name == "LBM2DMultiComponent":
        return LBM2DMultiComponent(48, 48)
    raise AssertionError(name)


@pytest.mark.parametrize("engine", _ENGINES)
class TestEngineFieldCatalog:
    def test_declared_fields_are_all_producible(self, engine):
        """get_field_names() must not promise anything get_field() refuses."""
        sim = _sim(engine)
        for name in sim.get_field_names():
            arr = sim.get_field(name)
            assert isinstance(arr, np.ndarray)
            assert arr.ndim >= 2, f"{engine}.{name} is not a field-shaped array"

    def test_declared_fields_have_a_label_and_a_lut(self, engine):
        for name in _sim(engine).get_field_names():
            assert name in FIELD_REGISTRY, f"{name} has no display label"
            assert name in MODE_TO_CMAP, f"{name} has no colourmap"

    def test_default_field_is_always_offered(self, engine):
        assert _DEFAULT_FIELD in _sim(engine).get_field_names()


class TestCatalogTracksCapabilities:
    def test_plain_lbm2d_offers_no_multiphase_fields(self):
        """The regression that motivated C18."""
        names = LBM2D(48, 48).get_field_names()
        for field in _MULTIPHASE_ONLY:
            assert field not in names, f"LBM2D must not advertise {field!r}"

    def test_multiphase_only_fields_belong_to_their_engines(self):
        assert "phase" in LBM2DLiquid(48, 48).get_field_names()
        mc = LBM2DMultiComponent(48, 48).get_field_names()
        for field in ("component1", "component2", "color"):
            assert field in mc

    def test_thermal_toggles_temperature(self):
        """Temperature is conditional, so the catalog is genuinely dynamic."""
        sim = LBM2D(48, 48)
        sim.thermal_enabled = True
        assert "temperature" in sim.get_field_names()
        sim.thermal_enabled = False
        assert "temperature" not in sim.get_field_names()

    def test_no_duplicates(self):
        for engine in _ENGINES:
            names = _sim(engine).get_field_names()
            assert len(names) == len(set(names))


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_colormap(self, name: str) -> None:
        self.calls.append(name)


def _bare_window(sim):
    """A MainWindow with just enough surface for _available_fields/_set_colormap.

    Constructing a real MainWindow would spin up a QOpenGLWidget, which cannot
    paint under the offscreen platform used for tests.
    """
    window = MainWindow.__new__(MainWindow)
    window.sim = sim
    window.viewport = _Recorder()
    window.analysis_panel = _Recorder()
    return window


class TestNoHardcodedCatalog:
    def test_app_py_defines_no_field_list(self):
        """Guard the fix itself: a module-level list of field names is exactly
        how this bug came back."""
        tree = ast.parse(_APP_PY.read_text())
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            assert "_COLORMAPS" not in targets

    def test_catalog_comes_from_the_engine(self):
        """_available_fields must mirror the engine, including its ordering."""
        sim = LBM2D(48, 48)
        assert _bare_window(sim)._available_fields() == sim.get_field_names()

    def test_unavailable_field_is_refused_and_reported(self, caplog):
        """Refused, and loud about it -- the old code dropped the request
        silently, which is how the dead entries went unnoticed."""
        window = _bare_window(LBM2D(48, 48))
        with caplog.at_level("WARNING"):
            window._set_colormap("component1")
        assert window.viewport.calls == [_DEFAULT_FIELD]
        assert any("component1" in r.getMessage() for r in caplog.records)

    def test_available_field_is_passed_through(self):
        window = _bare_window(LBM2D(48, 48))
        window._set_colormap("vorticity")
        assert window.viewport.calls == ["vorticity"]
        assert window.analysis_panel.calls == ["vorticity"]


class TestNoSilentFieldFallback:
    """C19: a request for an unavailable field must be reported, not quietly
    satisfied with a different field."""

    def test_viewport_logs_and_refuses_an_unknown_field(self, caplog):
        from workbench.viewport import Viewport

        viewport = Viewport.__new__(Viewport)
        viewport.sim = LBM2D(48, 48)
        viewport._cmap_tex = None
        with caplog.at_level("ERROR"):
            viewport.set_colormap("component1")
        assert viewport._colormap == _DEFAULT_FIELD
        assert any("component1" in r.getMessage() for r in caplog.records)

    def test_viewport_accepts_a_valid_field(self):
        from workbench.viewport import Viewport

        viewport = Viewport.__new__(Viewport)
        viewport.sim = LBM2D(48, 48)
        viewport._cmap_tex = None
        viewport.set_colormap("vorticity")
        assert viewport._colormap == "vorticity"

    def test_export_refuses_to_substitute_another_field(self, tmp_path, qapp):
        """An exported figure labelled with one field but showing another is
        the worst possible outcome, so the export must fail instead."""
        from export.image import export_image
        from scene.scene import default_scene

        out = tmp_path / "wrong.png"
        with pytest.raises(ValueError, match="component1"):
            export_image(
                sim=LBM2D(32, 32),
                scene=default_scene(),
                path=out,
                include_annotations=False,
                colormap="component1",
            )
        assert not out.exists(), "export wrote a file despite the bad field"

    def test_export_still_works_for_a_valid_field(self, tmp_path, qapp):
        from export.image import export_image
        from scene.scene import default_scene

        out = tmp_path / "ok.png"
        export_image(
            sim=LBM2D(32, 32),
            scene=default_scene(),
            path=out,
            include_annotations=False,
            colormap="vorticity",
        )
        assert out.exists()
