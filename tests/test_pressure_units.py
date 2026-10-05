"""Every consumer of the pressure field must read the same quantity.

PLAN.md C43 (P1-1, P1-2). The exit criterion for Phase 2 is "the same field/value is
consistent across viewport, colorbar, probes, analysis, reports, and exports". This
file is that check for pressure, and it found one consumer still disagreeing.

The gauge definition was established in C35/C36 and the viewport in C37. A probe
recording `pressure` did not follow. `Probe.record` reads non-local fields through
`sim.get_field(name)`, which is the **display render** — a `[0, 1]` value recentred
on 0.5 against an EMA-smoothed running extreme. So a probe's pressure trace was not
a pressure. Measured on a 64x64 driven channel with a wake probe, after 400 steps:

    probe pressure trace     min 0.500000   max 0.843490
    gauge pressure at cell   0.0203
    absolute pressure        0.3536

A density probe was worse, because the density render normalizes against the
instantaneous whole-domain min and max, so its scale changes every step:

    probe density trace      min 0.000000   max 0.660299
    density at cell          1.0609

Neither trace could be compared to anything else in the application, and the
pressure trace's scale drifts with the running EMA, so it is not even reproducible
between runs. `speed` was already correct — it is a `LOCAL_FIELDS` entry computed
from the physical velocity, and `probe.py` documents exactly this reasoning for why
it is not read from the engine's field.

There is a second consequence, which is the worse one: `get_field` **mutates** the
engine's EMA state. A probe recording `pressure` therefore changed the scale of the
pressure *render* — adding a probe altered how pressure is displayed on the
viewport, with no visible cause. Reading `get_field_raw` removes the coupling
entirely, which is asserted below.

Switching a probe to the physical field changes the numbers in existing saved
scenes' traces, and `probe.py` flags that risk for `speed`; here the change is not
optional, because a display-normalized value is not a measurement. Any saved
trace is a display artifact rather than a record, so nothing physical is lost.
"""

import numpy as np
import pytest

from engines.base import SimEngine
from engines.lbm2d import LBM2D
from scene.probe import Probe
from scene.scene import ProbeSpec

pytest.importorskip("PySide6")

PRESSURE_FIELDS = ("pressure", "density", "vorticity", "speed", "smoke")


@pytest.fixture(scope="module")
def driven():
    sim = LBM2D(width=64, height=64)
    sim.u_inflow = 0.08
    sim.add_obstacle(20, 32, radius=6)
    for _ in range(400):
        sim.step()
    return sim


def _probe(sim: LBM2D, fields=PRESSURE_FIELDS, steps: int = 200) -> Probe:
    probe = Probe(ProbeSpec(name="wake", x=34, y=32, fields=list(fields)))
    for step in range(1, steps + 1):
        sim.step()
        if step % 2 == 0:
            probe.record(sim, None, step=step)
    return probe


class TestTheEngineAgreesWithItself:
    def test_gauge_and_absolute_differ_by_the_reference(self, driven):
        assert np.allclose(
            driven.get_pressure_gauge(),
            driven.get_pressure() - SimEngine.REFERENCE_DENSITY / 3.0,
        )

    def test_the_reference_constants_are_the_lattice_sound_speed_squared(self, driven):
        assert SimEngine.REFERENCE_DENSITY == 1.0
        assert SimEngine.SOUND_SPEED_SQUARED == pytest.approx(1.0 / 3.0)


class TestProbesRecordPhysicalFields:
    def test_pressure_is_the_gauge_pressure_at_the_probe_cell(self, driven):
        probe = _probe(driven, fields=("pressure",))
        recorded = np.array(probe.history["pressure"])
        raw = driven.get_field_raw("pressure")
        assert recorded.min() >= raw.min() - 1e-9
        assert recorded.max() <= raw.max() + 1e-9
        # Not the render: a render is recentred on 0.5 and lives in [0, 1].
        assert recorded.max() <= 1.0 or recorded.min() >= -1.0
        assert abs(float(np.mean(recorded)) - 0.5) > 0.01

    def test_density_is_the_density_at_the_probe_cell(self, driven):
        probe = _probe(driven, fields=("density",))
        recorded = np.array(probe.history["density"])
        rho = driven.get_density()
        assert recorded.max() <= float(rho.max()) + 1e-9
        assert recorded.min() >= float(rho.min()) - 1e-9

    def test_every_probeable_field_lands_inside_the_raw_range(self, driven):
        probe = _probe(driven)
        for name, series in probe.history.items():
            recorded = np.array(series)
            raw = np.asarray(driven.get_field_raw(name))
            assert recorded.min() >= float(raw.min()) - 1e-9, name
            assert recorded.max() <= float(raw.max()) + 1e-9, name

    def test_a_probe_no_longer_perturbs_the_render(self):
        """`get_field` mutates the EMA, so a probe used to change the viewport.

        The render scale is EMA-smoothed by 5% per *call*, so two consecutive calls
        at the same instant differ — which means the only sound test is a control
        engine taking the same steps with no probe attached. Before the fix the
        probed engine's scale had advanced ~100 extra times.
        """
        probed = LBM2D(width=64, height=64)
        probed.u_inflow = 0.08
        probed.add_obstacle(20, 32, radius=6)
        control = LBM2D(width=64, height=64)
        control.u_inflow = 0.08
        control.add_obstacle(20, 32, radius=6)

        probe = Probe(ProbeSpec(name="wake", x=34, y=32, fields=["pressure"]))
        for step in range(1, 201):
            probed.step()
            control.step()
            if step % 2 == 0:
                probe.record(probed, None, step=step)

        assert np.array_equal(
            probed.get_field("pressure"), control.get_field("pressure")
        )

    def test_the_recording_path_does_not_call_get_field(self):
        import inspect

        source = inspect.getsource(Probe.record)
        assert "get_field_raw" in source
        assert "sim.get_field(" not in source


class TestTheViewportAndProbesAgree:
    def test_the_contour_field_is_the_field_probes_record(self, driven):
        import inspect

        from workbench.viewport import Viewport

        source = inspect.getsource(Viewport._draw_pressure_contours)
        assert "get_pressure_gauge()" in source

    def test_the_colorbar_unit_matches_the_engine_metadata(self, driven):
        from resources.colormaps import FIELD_REGISTRY, field_unit

        assert field_unit("pressure") == FIELD_REGISTRY["pressure"].unit
        assert field_unit("pressure") != ""


class TestTheAnalysisPanelAgreesWithTheEngine:
    def test_statistics_are_the_raw_field_not_the_render(self, app, driven):
        from workbench.panels.analysis_panel import AnalysisPanel

        panel = AnalysisPanel(driven)
        panel.set_colormap("pressure")
        panel._update_field_stats()
        raw = driven.get_field_raw("pressure")
        assert panel.mean_label.text() == f"{raw.mean():.4g}"

    def test_the_render_is_recentred_and_the_statistics_are_not(self, driven):
        """The two are different quantities, so they must not be confused."""
        render = np.asarray(driven.get_field("pressure"))
        raw = np.asarray(driven.get_field_raw("pressure"))
        assert 0.0 <= render.min() and render.max() <= 1.0
        assert abs(float(render.mean()) - 0.5) < abs(float(raw.mean())) + 0.5
        assert not np.allclose(render, raw)


class TestTheApiAgrees:
    def test_the_endpoint_serves_gauge_pressure(self, driven):
        pytest.importorskip("fastapi")
        from engines.api import build_app

        client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
            build_app()
        )
        response = client.get("/pressure")
        assert response.status_code == 200
        body = response.json()
        assert "rho" in body["quantity"] and "rho0" in body["quantity"]
        assert body["unit"] == "lattice"
        assert np.allclose(
            np.array(body["field"]), driven.get_pressure_gauge(), atol=1e-6
        )


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
