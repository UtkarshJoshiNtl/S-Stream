"""The Field Statistics panel must describe the field you selected.

**Selecting a field could show you a different field's numbers.** The panel computed
its statistics from a three-branch if/elif whose final `else` returned the speed
magnitude, so every field outside the two it handled got *speed* statistics:

    field selected          statistics actually shown
    smoke                   smoke                          (correct by accident)
    density                 rho                            (correct)
    phase                   rho - 1                        (shared with pressure)
    pressure                rho - 1                        (shared with phase)
    speed                   |u|                            (correct)
    vorticity               |u|                            wrong field
    temperature             |u|                            wrong field
    component1 / component2 |u|                            wrong field
    color                   |u|                            wrong field

Two separate defects sat in that table. The `else` substituted a different
quantity, and the `phase`/`pressure` pair shared one expression, so those two
fields reported *identical* numbers for two different quantities. `rho - 1` is also
three times the gauge pressure (PLAN.md C35/C36), so the pressure row was wrong a
second time.

Nothing on screen could catch it: the group title read only "Field Statistics", so a
panel describing speed while the viewport showed vorticity was internally consistent
as far as the window was concerned. Both are now pinned — the panel takes the
canonical un-normalized field from `SimEngine.get_field_raw` and names the field and
its unit in its own title, and a field the engine cannot physically produce says so
instead of borrowing another one.

The measurement tests here pass against both the old and new panel on purpose: they
measure the *engine's* fields, so they describe the defect. The tests that fail
before the fix are the ones asserting which field the panel consulted, because that
is the part that changed.
"""

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from engines.lbm3d import LBM3D
from resources.colormaps import FIELD_REGISTRY


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def driven():
    """A 64x64 channel with a real wake, 200 steps in."""
    sim = LBM2D(width=64, height=64)
    sim.u_inflow = 0.08
    for _ in range(200):
        sim.step()
    return sim


class TestTheBaseDefinesEachPhysicalField:
    def test_smoke_is_the_raw_smoke(self, driven):
        assert np.array_equal(driven.get_field_raw("smoke"), driven.get_smoke())

    def test_density_is_the_raw_density(self, driven):
        assert np.array_equal(driven.get_field_raw("density"), driven.get_density())

    def test_pressure_is_the_gauge_pressure_not_rho_minus_one(self, driven):
        raw = driven.get_field_raw("pressure")
        assert np.allclose(raw, driven.get_pressure_gauge())
        # The old panel used rho - 1, which is three times this.
        assert not np.allclose(raw, driven.get_density() - 1.0)

    def test_speed_is_the_horizontal_speed(self, driven):
        vel = driven.get_velocity()
        expected = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
        assert np.allclose(driven.get_field_raw("speed"), expected)

    def test_vorticity_is_the_curl_not_the_speed(self, driven):
        vort = driven.get_field_raw("vorticity")
        assert not np.allclose(vort, driven.get_field_raw("speed"))
        assert vort.shape == driven.get_velocity().shape[:-1]

    def test_a_render_cannot_tell_two_scales_apart(self):
        """Why statistics need the raw field: the render has no scale in it.

        Normalization is affine against the field's own range, so doubling every
        density leaves the `[0, 1]` render *bit-identical* while the physical field
        doubles. A min/max/mean read off the render therefore cannot report a
        magnitude, which is the whole reason `get_field_raw` exists.
        """
        sim = LBM2D(width=16, height=16)
        sim.rho[:] = 1.0
        sim.rho[8, 8] = 1.2
        render_at_rho = sim.get_field("density").copy()
        raw_at_rho = sim.get_field_raw("density").copy()

        sim.rho *= 2.0
        render_at_2rho = sim.get_field("density")
        raw_at_2rho = sim.get_field_raw("density")

        assert np.array_equal(render_at_rho, render_at_2rho)
        assert np.allclose(raw_at_2rho, 2.0 * raw_at_rho)
        assert render_at_rho.max() <= 1.0 and render_at_rho.min() >= 0.0


class TestCurlHasOneDefinition:
    """The base vorticity must equal what each engine used to compute inline.

    Six hand-written copies of `dv/dx - du/dy` is six chances to disagree about
    which axis is x, and a disagreement there shows up only as a subtly different
    image on another backend. The base definition is now checked against both
    engines' former inline versions, axis for axis.
    """

    def test_matches_the_two_dimensional_inline_version(self):
        sim = LBM2D(width=24, height=20)
        sim.u[:] = 0.1
        sim.v[:] = 0.02
        u, v = sim.u, sim.v
        dvdx = np.zeros_like(u, dtype=np.float32)
        dudy = np.zeros_like(u, dtype=np.float32)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
        assert np.allclose(sim.vorticity(), dvdx - dudy, atol=1e-6)

    def test_matches_the_three_dimensional_inline_version(self):
        sim = LBM3D(width=8, height=8, depth=6)
        vel = sim.get_velocity_view()
        u, v = vel[..., 0], vel[..., 1]
        dvdx = np.zeros_like(u, dtype=np.float32)
        dudy = np.zeros_like(u, dtype=np.float32)
        dvdx[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) * 0.5
        dudy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) * 0.5
        assert np.allclose(sim.vorticity(), dvdx - dudy, atol=1e-6)

    def test_three_dimensional_speed_sums_the_last_two_components(self):
        sim = LBM3D(width=8, height=8, depth=6)
        vel = sim.get_velocity()
        # Components are stored (z, y, x), so horizontal speed is the last two.
        assert np.allclose(
            sim.speed_magnitude(), np.sqrt(vel[..., 1] ** 2 + vel[..., 2] ** 2)
        )
        # Not all three, and not the first two.
        assert not np.allclose(sim.speed_magnitude(), np.sqrt((vel**2).sum(axis=-1)))


class TestUnknownFieldsRefuse:
    def test_keyerror_names_the_engine_and_the_field(self, driven):
        with pytest.raises(KeyError) as excinfo:
            driven.get_field_raw("phase")
        message = str(excinfo.value)
        assert "LBM2D" in message and "phase" in message

    def test_temperature_without_the_thermal_model_refuses(self, driven):
        assert "temperature" not in driven.get_field_names()
        with pytest.raises(KeyError):
            driven.get_field_raw("temperature")


class TestEngineSpecificFieldsArePhysical:
    def test_liquid_phase_is_the_unnormalized_sigmoid(self):
        sim = LBM2DLiquid(width=32, height=32)
        raw = sim.get_field_raw("phase")
        assert np.allclose(raw, 1.0 / (1.0 + np.exp(-15.0 * (sim.rho - 0.5))))

    def test_multicomponent_exposes_both_components(self):
        sim = LBM2DMultiComponent(width=24, height=24)
        assert np.allclose(sim.get_field_raw("component1"), sim.rho1)
        assert np.allclose(sim.get_field_raw("component2"), sim.rho2)

    def test_multicomponent_color_is_the_normalized_difference(self):
        sim = LBM2DMultiComponent(width=24, height=24)
        raw = sim.get_field_raw("color")
        total = sim.rho1 + sim.rho2
        expected = np.divide(
            sim.rho1 - sim.rho2,
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )
        assert np.allclose(raw, expected)

    def test_every_offered_field_has_a_physical_accessor(self):
        """A field in the dropdown must never be un-measurable."""
        for engine in (
            LBM2D(width=16, height=16),
            LBM2DLiquid(width=16, height=16),
            LBM2DMultiComponent(width=16, height=16),
        ):
            for name in engine.get_field_names():
                if name == "temperature":  # needs the thermal mixin
                    continue
                assert isinstance(np.asarray(engine.get_field_raw(name)), np.ndarray)


class TestThePanelReadsTheSelectedField:
    @staticmethod
    def _stats(panel):
        return (
            panel.min_label.text(),
            panel.max_label.text(),
            panel.mean_label.text(),
        )

    def _panel(self, app, sim, field):
        pytest.importorskip("PySide6.QtWidgets")
        from workbench.panels.analysis_panel import AnalysisPanel

        panel = AnalysisPanel(sim)
        panel.set_colormap(field)
        panel._update_field_stats()
        return panel

    def test_vorticity_no_longer_prints_the_speed(self, app, driven):
        panel = self._panel(app, driven, "vorticity")
        vort = driven.get_field_raw("vorticity")
        assert panel.min_label.text() == f"{vort.min():.4g}"
        assert panel.mean_label.text() == f"{vort.mean():.4g}"
        speed = driven.get_field_raw("speed")
        assert panel.mean_label.text() != f"{speed.mean():.4g}"

    def test_pressure_and_phase_are_no_longer_the_same_numbers(self, app):
        liquid = LBM2DLiquid(width=24, height=24)
        pressure = self._panel(app, liquid, "pressure")
        phase = self._panel(app, liquid, "phase")
        assert self._stats(pressure) != self._stats(phase)

    def test_title_names_the_field_and_its_unit(self, app, driven):
        panel = self._panel(app, driven, "pressure")
        title = panel.stats_group.title()
        assert "Pressure" in title
        assert FIELD_REGISTRY["pressure"].unit in title

    def test_an_unavailable_field_says_so_instead_of_substituting(self, app, driven):
        panel = self._panel(app, driven, "phase")
        assert panel.min_label.text() == "—"
        assert panel.mean_label.text() == "—"
        assert "unavailable" in panel.stats_group.title()
        assert "LBM2D" in panel.stats_group.toolTip()

    def test_no_reported_statistic_is_a_copy_of_another_field(self, app, driven):
        """The old bug in one assertion: all but two fields were identical."""
        seen = {}
        for name in ("smoke", "speed", "vorticity", "pressure", "density"):
            panel = self._panel(app, driven, name)
            assert (
                self._stats(panel) not in seen
            ), f"{name} duplicates {seen.get(self._stats(panel))}"
            seen[self._stats(panel)] = name

    def test_the_source_does_not_fall_back_to_speed(self):
        import inspect

        from workbench.panels.analysis_panel import AnalysisPanel

        source = inspect.getsource(AnalysisPanel._update_field_stats)
        assert "get_field_raw" in source
        assert "get_velocity" not in source
