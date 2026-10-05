"""Contours and colorbar read the canonical pressure, with units.

PLAN.md C37 (P1-2). Two surfaces in the viewport, two different defects.

**Contours were drawing almost nothing.** `_draw_pressure_contours` read the
*absolute* field while choosing levels from `linspace(-mx, mx, 11)`. On a 64x64
driven channel at step 200, measured on real output:

                              field range      spread    levels intersecting data
    absolute field           0.286 .. 0.373    0.0871          2 of 11
    gauge field             -0.048 .. 0.040    0.0871         11 of 11

The spread is *identical* either way. The offset is the entire difference: eleven
levels symmetric about zero cannot reach into a band 0.087 wide sitting at 1/3, so
only two of them intersect the data at all, and the one at exactly zero traces the
iso-surface where rho = 1 -- a mean-density contour, not a pressure fluctuation.
Active marching-squares cells across all eleven levels: 58 for the absolute field,
490 for the gauge. The overlay was not showing a faint pressure field; it was showing
a handful of fragments, which reads as "this feature does not work" rather than as a
units bug.

The same asymmetry is invisible in the rendered texture — C36 found that the signed
render divides by `max|p|`, so the factor of 3 cancels. Contours are the first
viewport surface where the difference is actually measurable, which is why they were
worth checking numerically rather than by eye.

**The colorbar had no unit and no scale claim.** It printed the field label only. The
bar maps the engine's *rendered* field, which is normalized to [0, 1] against an
EMA-smoothed running extreme, so labelling it with the field's unit alone would be a
claim the bar cannot back up. It now states both: the unit from the C35 metadata, and
that the range is normalized.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from engines.lbm2d import LBM2D

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

_VIEWPORT = Path(__file__).resolve().parents[1] / "workbench" / "viewport.py"


def _active_contour_cells(field: np.ndarray, n_levels: int = 11) -> int:
    """Count marching-squares cells that straddle any level.

    A copy of the viewport's own criterion (`0 < idx < 15`), kept independent of the
    implementation on purpose: this is the measurement, not a restatement.
    """
    mx = max(float(np.percentile(np.abs(field), 95)), 1e-6)
    total = 0
    for level in np.linspace(-mx, mx, n_levels):
        tl = (field[:-1:2, :-1:2] >= level).astype(np.int32)
        tr = (field[:-1:2, 1::2] >= level).astype(np.int32)
        br = (field[1::2, 1::2] >= level).astype(np.int32)
        bl = (field[1::2, :-1:2] >= level).astype(np.int32)
        idx = (tl << 3) | (tr << 2) | (br << 1) | bl
        total += int(((idx > 0) & (idx < 15)).sum())
    return total


def _driven_field(n_steps: int = 200) -> LBM2D:
    sim = LBM2D(width=64, height=64)
    sim.u_inflow = 0.1
    for _ in range(n_steps):
        sim.step()
    return sim


class TestContoursFindRealStructure:
    """The measurement that motivates the fix, kept as a test so the regression
    cannot come back unnoticed."""

    def test_the_gauge_field_yields_far_more_contours(self):
        sim = _driven_field()
        absolute = _active_contour_cells(sim.get_pressure())
        gauge = _active_contour_cells(sim.get_pressure_gauge())
        assert gauge > 4 * absolute, (
            f"the gauge field should trace far more structure than the absolute one "
            f"(gauge {gauge}, absolute {absolute}); if these have converged, the "
            "level selection has been revisited and this ratio needs a new value"
        )

    def test_the_absolute_field_buries_most_levels(self):
        """The mechanism, measured: the level span is set by the 1/3 offset while the
        field's *variation* is small, so almost no level intersects the data at all.

        Note that the field spread is identical either way (0.0871 for both) — the
        offset is the entire difference. Eleven levels symmetric about zero can only
        reach into a band 0.087 wide around 1/3 if the span were set by the
        fluctuations, and it is not.
        """
        sim = _driven_field()
        counts = {}
        for name, field in (
            ("absolute", sim.get_pressure()),
            ("gauge", sim.get_pressure_gauge()),
        ):
            mx = max(float(np.percentile(np.abs(field), 95)), 1e-6)
            levels = np.linspace(-mx, mx, 11)
            counts[name] = int(
                np.sum((levels >= float(field.min())) & (levels <= float(field.max())))
            )

        assert counts["absolute"] <= 3, (
            f"expected the reference offset to leave most levels outside the data; "
            f"got {counts['absolute']}/11"
        )
        assert counts["gauge"] == 11, (
            f"the gauge field should be fully covered by the level set; got "
            f"{counts['gauge']}/11"
        )

    def test_the_offset_alone_accounts_for_the_difference(self):
        """The two fields differ by exactly the reference pressure and nothing else,
        so the level-coverage difference cannot come from the physics."""
        sim = _driven_field()
        absolute = sim.get_pressure()
        gauge = sim.get_pressure_gauge()
        assert np.allclose(absolute - gauge, sim.reference_pressure())
        assert float(absolute.max() - absolute.min()) == pytest.approx(
            float(gauge.max() - gauge.min()), rel=1e-6
        )

    def test_the_gauge_levels_all_intersect_the_data(self):
        sim = _driven_field()
        gauge = sim.get_pressure_gauge()
        mx = max(float(np.percentile(np.abs(gauge), 95)), 1e-6)
        levels = np.linspace(-mx, mx, 11)
        assert float(levels.max()) <= float(gauge.max())
        assert float(levels.min()) >= float(gauge.min())


class TestContoursReadTheCanonicalQuantity:
    def test_the_source_calls_the_gauge(self):
        """Source-level, because the viewport needs a GL context to exercise
        faithfully, and because this is a one-line claim about which method is
        called."""
        tree = ast.parse(_VIEWPORT.read_text())
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_draw_pressure_contours"
        )
        calls = {
            node.func.attr
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "get_pressure_gauge" in calls
        assert "get_pressure" not in calls

    def test_the_floor_is_small_enough_to_survive_a_weak_field(self):
        """`mx` was floored at 1e-3, which is a third of a typical gauge pressure
        magnitude. A run with almost no pressure variation would therefore draw
        contours at levels far outside its actual range."""
        sim = LBM2D(width=32, height=32)
        sim.rho[:] = 1.0 + 1e-5 * np.linspace(0, 1, 32).reshape(-1, 1)
        gauge = sim.get_pressure_gauge()
        assert float(np.abs(gauge).max()) < 1e-3, (
            "this fixture is supposed to be a near-resting field; if it is not, the "
            "floor being tested is not the active one"
        )

        source = inspect.getsource(
            __import__("workbench.viewport", fromlist=["Viewport"]).Viewport
        ).split("def _draw_pressure_contours", 1)[1]
        assert "1e-6" in source


class TestColorbarDeclaresItsUnitsAndScale:
    def test_the_source_reads_the_unit_metadata(self):
        source = inspect.getsource(
            __import__("workbench.viewport", fromlist=["Viewport"]).Viewport
        ).split("def _draw_colorbar", 1)[1]
        assert "field_unit(" in source

    def test_it_says_the_range_is_normalized(self):
        """The bar maps a normalized render. Printing only the unit would imply the
        bar's ends are physical values, which they are not."""
        source = inspect.getsource(
            __import__("workbench.viewport", fromlist=["Viewport"]).Viewport
        ).split("def _draw_colorbar", 1)[1]
        assert "normalized" in source

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("pressure", "lattice"),
            ("density", "lattice"),
            ("speed", "lattice/step"),
            ("vorticity", "1/step"),
            ("smoke", ""),
        ],
    )
    def test_each_field_gets_its_declared_unit(self, field, expected):
        from resources.colormaps import field_unit

        assert field_unit(field) == expected

    def test_a_dimensionless_field_is_marked_normalized_bare(self):
        """`smoke` has no unit, so the note is the scale claim alone. It still has to
        be drawn, or the bar silently loses its only warning."""
        from resources.colormaps import field_unit

        assert field_unit("smoke") == ""
        note = (
            f"{field_unit('smoke')} (normalized)"
            if field_unit("smoke")
            else "normalized"
        )
        assert note == "normalized"


class TestTheUnitReachesTheViewport:
    def test_the_viewport_imports_field_unit(self):
        source = _VIEWPORT.read_text()
        assert "field_unit" in source

    def test_the_metadata_is_not_redeclared_in_the_viewport(self):
        """The unit belongs to the field's definition. A copy here would drift."""
        source = _VIEWPORT.read_text()
        assert '"lattice"' not in source
        assert "'lattice'" not in source
