"""A diverged run has to say so, while it is diverging.

PLAN.md C44 (P1-3). `check_sanity` inspected only *setup* -- omega, viscosity, inlet
speed, obstacle size, probe count -- so every warning it could raise was already true
at step 0 and stayed true until the scene changed. Two consequences, both measured
on a 48x48 channel with an r=5 cylinder:

    u_inflow=0.5   peak speed crosses 0.5 at step 55
    u_inflow=0.4   step 230
    u_inflow=0.3   never within 600 steps

A run that had already filled the lattice with NaN produced *no warning at all*,
because a sane setup that has since diverged looks exactly like a sane setup that has
not. And because the banner was rebuilt only on apply/reset/load, the scene's warnings
described step 0 for the entire run -- so even a warning that existed would not have
reached the screen.

The second half is cadence, which is the recurring lesson of this phase: two displays
of the same warnings must be refreshed at the same step, or they will disagree.

**The first density floor was wrong and flagged a healthy engine.** It was set to 0.1,
"clearly below rho0 = 1" -- and the Shan-Chen liquid engine is *designed* with a
low-density vapour background. Measured on a fresh 48x48: rho in **[0.01, 2.0]**, the
2.0 being the droplet. A bound that fires on an engine's initial condition teaches
users to ignore the banner, which is worse than having none. The floor is now 1e-3,
an order of magnitude below that vapour and still catching zero and negative mass.

**The first implementation skipped an engine entirely.** It read `sim.rho`, which the
CPU engines have and `LBM2DMultiComponent` does *not* -- that engine keeps `rho1`/
`rho2` and exposes no `rho`, so the function returned `available=False` and quietly
judged nothing. It now goes through `get_density()`/`get_velocity()`, which mean the
same thing on every engine, and the tests assert `available` is True for each one.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from analysis.sanity import (
    DENSITY_CEILING,
    DENSITY_FLOOR,
    MAX_LATTICE_SPEED,
    check_sanity,
    inspect_field_state,
)
from engines.lbm2d import LBM2D
from scene.scene import CircleObstacle, ProbeSpec, Scene

pytest.importorskip("PySide6")


def _quiet(steps: int = 0):
    """Step without letting numpy's warnings reach the test output."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for _ in range(steps):
            yield


def _scene() -> Scene:
    scene = Scene(name="cylinder", width=48, height=48)
    scene.obstacles = [CircleObstacle(name="c", x=16, y=24, radius=5)]
    scene.probes = [ProbeSpec(name="wake", x=30, y=24, fields=["v"])]
    return scene


def _driven(u_inflow: float) -> LBM2D:
    sim = LBM2D(width=48, height=48)
    sim.u_inflow = u_inflow
    sim.add_obstacle(16, 24, radius=5)
    return sim


class TestTheThresholdsAreWhereTheMeasurementIs:
    def test_the_speed_limit_is_above_the_fastest_healthy_run(self):
        sim = _driven(0.05)
        for _ in _quiet(400):
            sim.step()
        assert inspect_field_state(sim).max_speed < MAX_LATTICE_SPEED / 4

    def test_the_speed_limit_is_twice_the_dangerous_inlet_line(self):
        """u_inflow > 0.25 is already 'outside the low-Mach regime for this solver'."""
        assert MAX_LATTICE_SPEED >= 2 * 0.25

    def test_the_density_floor_clears_the_liquid_vapour_background(self):
        pytest.importorskip("engines.lbm2d_liquid")
        from engines.lbm2d_liquid import LBM2DLiquid

        sim = LBM2DLiquid(width=48, height=48)
        rho = np.asarray(sim.get_density())
        assert DENSITY_FLOOR < rho.min(), f"floor {DENSITY_FLOOR} vs vapour {rho.min()}"
        assert inspect_field_state(sim).diverged is False

    def test_the_density_ceiling_clears_a_shan_chen_droplet(self):
        pytest.importorskip("engines.lbm2d_liquid")
        from engines.lbm2d_liquid import LBM2DLiquid

        sim = LBM2DLiquid(width=48, height=48)
        assert DENSITY_CEILING > float(np.max(sim.get_density()))

    def test_the_floor_still_catches_empty_and_negative_mass(self):
        assert DENSITY_FLOOR > 0.0
        assert DENSITY_FLOOR < 1e-2  # strictly above "identically zero"


class TestNoEngineIsSkipped:
    @pytest.mark.parametrize(
        "factory",
        ["2d", "liquid", "multicomponent"],
    )
    def test_field_state_is_available_for_every_engine(self, factory):
        if factory == "2d":
            from engines.lbm2d import LBM2D

            sim = LBM2D(width=32, height=32)
        elif factory == "liquid":
            from engines.lbm2d_liquid import LBM2DLiquid

            sim = LBM2DLiquid(width=32, height=32)
        else:
            from engines.lbm2d_multicomponent import LBM2DMultiComponent

            sim = LBM2DMultiComponent(width=32, height=32)
        health = inspect_field_state(sim)
        assert health.available, f"{factory} was not inspected at all"
        assert health.diverged is False

    def test_a_healthy_engine_reports_no_danger(self):
        sim = _driven(0.05)
        for _ in _quiet(400):
            sim.step()
        levels = [w.level for w in check_sanity(sim, _scene(), step_count=400)]
        assert "danger" not in levels


class TestDivergenceIsCaughtWhileItHappens:
    def test_a_runaway_inlet_is_reported(self):
        sim = _driven(0.5)
        for _ in _quiet(55):
            sim.step()
        warnings = check_sanity(sim, _scene(), step_count=55)
        assert any(w.title == "Simulation diverged" for w in warnings)
        assert any(w.level == "danger" for w in warnings)

    def test_the_message_names_the_number_that_tripped_it(self):
        sim = _driven(0.5)
        for _ in _quiet(55):
            sim.step()
        message = next(
            w.message
            for w in check_sanity(sim, _scene(), step_count=55)
            if w.title == "Simulation diverged"
        )
        assert "lattice speed" in message
        assert f"{MAX_LATTICE_SPEED}" in message

    def test_a_non_finite_field_is_reported_with_a_cell_count(self):
        sim = _driven(0.05)
        for _ in _quiet(50):
            sim.step()
        sim.rho[:] = np.nan
        health = inspect_field_state(sim)
        assert health.diverged
        assert health.bad_cells == 48 * 48
        assert "non-finite" in health.describe()

    def test_an_infinite_density_is_reported(self):
        sim = _driven(0.05)
        sim.rho[10, 10] = np.inf
        assert inspect_field_state(sim).diverged

    def test_a_negative_density_is_reported(self):
        sim = _driven(0.05)
        sim.rho[10, 10] = -0.5
        assert inspect_field_state(sim).diverged
        assert "positive" in inspect_field_state(sim).describe()

    def test_a_runaway_density_is_reported(self):
        sim = _driven(0.05)
        sim.rho[0, 0] = DENSITY_CEILING * 10
        assert inspect_field_state(sim).diverged

    def test_divergence_outranks_the_setup_advice(self):
        """First in the list, so the banner names the real problem first."""
        sim = _driven(0.6)
        for _ in _quiet(30):
            sim.step()
        warnings = check_sanity(sim, _scene(), step_count=30)
        assert warnings[0].title == "Simulation diverged"

    def test_a_reset_run_is_judged_afresh(self):
        """The check reads the field, so it recovers when the field does."""
        sim = _driven(0.05)
        for _ in _quiet(50):
            sim.step()
        sim.rho[:] = np.nan
        assert inspect_field_state(sim).diverged
        sim.rho[:] = 1.0
        sim.u[:] = 0.0
        assert inspect_field_state(sim).diverged is False


class TestTheBannerRefreshesDuringARun:
    @pytest.fixture(scope="module")
    def app(self):
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    @pytest.fixture
    def window(self, app):
        from workbench.app import SANITY_REFRESH_STEPS, MainWindow

        win = MainWindow(LBM2D(width=48, height=48))
        win.timer.stop()
        win._adopt_scene(_scene())
        win._apply_and_refresh()
        yield win, SANITY_REFRESH_STEPS
        win.close()

    def test_the_cadence_matches_the_outcome_panel(self, window):
        """Two displays of one warning set must not describe different steps."""
        win, cadence = window
        assert cadence == 15
        panel_source = inspect_source(win.outcome_panel.update_outcome)
        assert f"self.step_count % {cadence} != 0" in panel_source

    def test_the_banner_is_not_rebuilt_on_every_step(self, window):
        win, cadence = window
        calls = []
        original = win._update_sanity_banner
        win._update_sanity_banner = lambda: calls.append(win.step_count)
        try:
            for _ in range(cadence * 3):
                win.tick()
        finally:
            win._update_sanity_banner = original
        assert len(calls) <= 4

    def test_the_banner_refreshes_within_the_run(self, window):
        win, cadence = window
        calls = []
        original = win._update_sanity_banner
        win._update_sanity_banner = lambda: calls.append(win.step_count)
        try:
            for _ in range(cadence):
                win.tick()
        finally:
            win._update_sanity_banner = original
        assert calls, "the banner never refreshed during the run"

    def test_the_banner_picks_up_divergence_mid_run(self, window):
        """A real runaway, not a poisoned array.

        The first attempt here wrote NaN into `win.sim.rho` and ticked, which passes
        for the wrong reason and for no reason at all: `rho` is *derived* state, and
        `step()` recomputes it from the distribution, so the NaN is overwritten on the
        first tick and the engine stays perfectly healthy. Poisoning `f` instead
        would work but would test the injector, not the detector. So the window runs
        an inlet that genuinely diverges -- measured onset step 55 at u=0.5 -- and the
        banner is expected to name it on the way past.
        """
        win, cadence = window
        assert "diverged" not in win.sanity_banner.text().lower()
        win.sim.u_inflow = 0.5
        while not inspect_field_state(win.sim).diverged:
            win.tick()
        # One more cadence's worth: the banner must catch up on its own schedule.
        for _ in range(cadence):
            win.tick()
        text = win.sanity_banner.text().lower()
        assert "diverged" in text, text

    def test_a_manual_step_also_refreshes(self, window):
        win, cadence = window
        calls = []
        original = win._update_sanity_banner
        win._update_sanity_banner = lambda: calls.append(win.step_count)
        try:
            for _ in range(cadence):
                win.step_once()
        finally:
            win._update_sanity_banner = original
        assert calls, "stepping by hand never refreshed the banner"

    def test_the_banner_recovers_after_a_reset(self, window):
        win, cadence = window
        win.sim.u_inflow = 0.5
        while not inspect_field_state(win.sim).diverged:
            win.tick()
        for _ in range(cadence):
            win.tick()
        assert "diverged" in win.sanity_banner.text().lower()
        win._reset_runtime_state()
        win._apply_and_refresh()
        assert "diverged" not in win.sanity_banner.text().lower()


def inspect_source(func) -> str:
    import inspect

    return inspect.getsource(func)


class TestTheReportSeesItToo:
    def test_an_exported_report_carries_the_warning(self, tmp_path):
        from export.report import export_markdown_report

        sim = _driven(0.5)
        for _ in _quiet(55):
            sim.step()
        path = tmp_path / "r.md"
        export_markdown_report(
            path,
            _scene(),
            sim,
            55,
            warnings=check_sanity(sim, _scene(), step_count=55),
        )
        assert "Simulation diverged" in path.read_text()
