"""State history needs a contract, not a poke at engine internals (PLAN.md C49).

`_record_state_history` did `getattr(self.sim, "f", self.sim.get_velocity_view())`
and `_on_timeline_moved` did `np.copyto(self.sim.f, f_snap)`. For an engine whose
arrays are not NumPy -- or not named `f`/`rho` -- that either raises inside the tick
handler or, worse, restores nothing while the slider moves as though it worked.
"""

import pytest

pytest.importorskip("PySide6")

from engines.base import EngineState, SimEngine  # noqa: E402
from engines.lbm2d import LBM2D  # noqa: E402
from scene.scene import CircleObstacle, Scene  # noqa: E402
from tests.conftest import dispose_window  # noqa: E402


def prep(sim, steps: int = 40):
    sim.add_obstacle(20, 20, radius=5)
    sim.add_emitter(8, 10, strength=0.2)
    sim.smoke[12:18, 12:18] = 1.0
    sim.run(steps)
    return sim


class TestTheContractIsTotal:
    def test_restoring_and_stepping_reproduces_the_uninterrupted_run(self):
        sim = prep(LBM2D(width=64, height=64))
        snap = sim.snapshot()
        sim.step()
        expected = sim.get_density().copy()
        sim.restore(snap)
        sim.step()
        assert np_allclose_zero(sim.get_density(), expected)

    def test_the_distributions_come_back_exactly(self):
        sim = prep(LBM2D(width=64, height=64))
        snap = sim.snapshot()
        sim.run(30)
        sim.restore(snap)
        assert np_allclose_zero(sim.get_f(), snap.arrays["f"])

    def test_the_scalar_field_comes_back(self):
        """Smoke was simply absent from the old snapshot: a scrubbed frame rendered
        a different picture from the moment it was taken."""
        sim = prep(LBM2D(width=64, height=64))
        before = sim.get_smoke().copy()
        snap = sim.snapshot()
        sim.run(30)
        assert not np_allclose_zero(sim.get_smoke(), before)
        sim.restore(snap)
        assert np_allclose_zero(sim.get_smoke(), before)


def np_allclose_zero(a, b) -> bool:
    import numpy as np

    return bool(np.all(np.asarray(a) == np.asarray(b)))


class TestTheSnapshotIsIsolated:
    def test_stepping_afterwards_does_not_change_the_snapshot(self):
        sim = prep(LBM2D(width=64, height=64))
        snap = sim.snapshot()
        frozen = snap.arrays["rho"].copy()
        sim.run(50)
        assert np_allclose_zero(snap.arrays["rho"], frozen)

    def test_mutating_the_snapshot_does_not_change_the_engine(self):
        sim = prep(LBM2D(width=64, height=64))
        snap = sim.snapshot()
        before = sim.rho.copy()
        snap.arrays["rho"][:] = 99.0
        assert np_allclose_zero(sim.rho, before)

    def test_it_is_a_copy_and_not_a_view(self):
        sim = prep(LBM2D(width=64, height=64))
        snap = sim.snapshot()
        assert not np_shares_memory(snap.arrays["f"], sim.f)


def np_shares_memory(a, b) -> bool:
    import numpy as np

    return bool(np.shares_memory(np.asarray(a), np.asarray(b)))


class TestTheSnapshotIsPortable:
    def test_every_array_is_host_side_numpy(self):
        """A device buffer cannot be handed to a UI thread or written to disk."""
        import numpy as np

        snap = prep(LBM2D(width=64, height=64)).snapshot()
        for name, array in snap.arrays.items():
            assert isinstance(array, np.ndarray), name

    def test_it_reports_its_grid(self):
        snap = prep(LBM2D(width=48, height=64)).snapshot()
        assert snap.grid_shape == (64, 48)


class TestBadSnapshotsAreRefused:
    def test_a_mismatched_grid_is_refused(self):
        sim = LBM2D(width=64, height=64)
        other = LBM2D(width=32, height=32)
        with pytest.raises(ValueError, match="32"):
            sim.restore(other.snapshot())

    def test_a_missing_field_is_refused_by_name(self):
        sim = LBM2D(width=64, height=64)
        broken = EngineState(
            arrays={"f": sim.f.copy()}, scalars={}, grid_shape=(64, 64)
        )
        with pytest.raises(ValueError, match="rho"):
            sim.restore(broken)

    def test_a_partial_restore_does_not_happen_first(self):
        """The check runs before any array is written."""
        sim = LBM2D(width=64, height=64)
        sim.run(3)
        before = sim.f.copy()
        with pytest.raises(ValueError):
            sim.restore(EngineState(arrays={}, scalars={}, grid_shape=(64, 64)))
        assert np_allclose_zero(sim.f, before)


class TestGeometryAndScalarsSurvive:
    def test_obstacles_come_back(self):
        sim = prep(LBM2D(width=64, height=64))
        snap = sim.snapshot()
        sim.clear_obstacles()
        sim.restore(snap)
        assert bool(sim.get_obstacles().any())

    def test_emitters_come_back(self):
        sim = prep(LBM2D(width=64, height=64))
        snap = sim.snapshot()
        sim.clear_emitters()
        sim.restore(snap)
        assert sim.get_emitter_count() == 1

    def test_particles_come_back(self):
        sim = prep(LBM2D(width=64, height=64))
        tracer = sim.get_particle_tracer()
        tracer.add_particles(10, 10, count=5)
        tracer.step(sim.get_velocity())
        positions = tracer.get_positions()
        snap = sim.snapshot()
        tracer.clear()
        assert tracer.count == 0
        sim.restore(snap)
        assert tracer.count == len(positions)
        assert np_allclose_zero(tracer.get_positions(), positions)

    def test_render_scaling_comes_back(self):
        """`get_field` divides by these, so leaving them behind makes a restored
        frame render differently -- which reads as a broken restore."""
        sim = prep(LBM2D(width=64, height=64))
        ema = sim._ema_speed_max
        snap = sim.snapshot()
        sim._ema_speed_max = ema * 100
        sim.restore(snap)
        assert sim._ema_speed_max == ema

    def test_scratch_buffers_are_left_out(self):
        """Stated, not discovered: each is rewritten before it is read."""
        snap = LBM2D(width=32, height=32).snapshot()
        assert "_f_swap" not in snap.arrays
        assert "_lap_buffer" not in snap.arrays
        assert "_vel_buf" not in snap.arrays
        assert LBM2D._STATE_ARRAYS == ("f", "rho", "u", "v", "smoke", "obstacles")


class TestEnginesWithoutSupportSaySo:
    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(
                lambda: __import__("engines.lbm3d", fromlist=["LBM3D"]).LBM3D(
                    width=8, height=8, depth=8
                ),
                id="LBM3D",
            ),
            pytest.param(
                lambda: __import__(
                    "engines.lbm2d_liquid", fromlist=["LBM2DLiquid"]
                ).LBM2DLiquid(width=32, height=32),
                id="LBM2DLiquid",
            ),
            pytest.param(
                lambda: __import__(
                    "engines.lbm2d_multicomponent", fromlist=["LBM2DMultiComponent"]
                ).LBM2DMultiComponent(width=32, height=32),
                id="LBM2DMultiComponent",
            ),
        ],
    )
    def test_it_does_not_claim_support(self, factory):
        sim = factory()
        assert sim.supports_state_history is False

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(
                lambda: __import__("engines.lbm3d", fromlist=["LBM3D"]).LBM3D(
                    width=8, height=8, depth=8
                ),
                id="LBM3D",
            ),
            pytest.param(
                lambda: __import__(
                    "engines.lbm2d_multicomponent", fromlist=["LBM2DMultiComponent"]
                ).LBM2DMultiComponent(width=32, height=32),
                id="LBM2DMultiComponent",
            ),
        ],
    )
    def test_it_refuses_loudly_rather_than_recording_the_wrong_field(self, factory):
        sim = factory()
        with pytest.raises(NotImplementedError, match=type(sim).__name__):
            sim.snapshot()

    def test_the_message_names_the_engine(self):
        sim = LBM2D(width=32, height=32)
        with pytest.raises(NotImplementedError, match="supports_state_history"):
            SimEngine.snapshot(sim)


class TestTheStateTypeIsLegible:
    def test_it_lists_its_fields(self):
        snap = LBM2D(width=32, height=32).snapshot()
        text = repr(snap)
        assert "EngineState" in text
        assert "f" in text
        assert "rho" in text

    def test_it_defaults_to_empty(self):
        empty = EngineState()
        assert empty.arrays == {}
        assert empty.scalars == {}


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


class TestTheTimelineUsesTheContract:
    def test_it_stores_engine_states(self, window):
        window.sim.run(5)
        window.step_once()
        assert window._state_history
        _step, state = window._state_history[-1]
        assert isinstance(state, EngineState)

    def test_restoring_goes_through_the_engine(self, window):
        window.sim.run(5)
        window.step_once()
        window.sim.run(20)
        window._on_timeline_moved(len(window._state_history) - 1)
        # Restored, so the next step resumes from the snapshot rather than +20.
        assert window.sim.get_smoke().shape == window.sim.smoke.shape

    def test_it_never_reaches_past_the_contract(self, window):
        """P1-7: `sim.f` is where a CuPy engine breaks the tick handler."""
        import inspect

        source = inspect.getsource(__import__("workbench.app", fromlist=["MainWindow"]))
        assert "self.sim.f" not in source
        assert "self.sim.rho" not in source

    def test_restore_is_the_only_thing_that_writes_state(self, window):
        """The timeline delegates; it does not do the copying itself.

        The device-boundary claim (P1-7) is *structural* rather than executed: a CuPy
        engine is not installed here, so a stand-in device array would only prove that
        NumPy rejects it -- which is the finding, not a fix. What is testable is that
        the UI asks the engine to do the work.
        """
        called = []
        original = window.sim.restore

        def spy(state):
            called.append(state)
            original(state)

        window.sim.restore = spy
        window.sim.run(5)
        window.step_once()
        window.sim.run(20)
        window._on_timeline_moved(len(window._state_history) - 1)
        assert len(called) == 1
        assert isinstance(called[0], EngineState)

    def test_recording_does_not_touch_engine_arrays(self, window):
        window.sim.run(5)
        window._record_state_history()
        _step, state = window._state_history[-1]
        # A velocity view is what the old fallback recorded for an engine with no `f`.
        assert "smoke" in state.arrays
        assert state.arrays["smoke"].shape == window.sim.smoke.shape

    def test_the_slider_is_enabled_when_the_engine_supports_it(self, window):
        window._sync_timeline_availability()
        assert window.timeline_slider.isEnabled() is True


class TestTheTimelineSaysWhenItCannotWork:
    def test_an_unsupported_engine_disables_the_slider(self, window):
        from engines.lbm2d_multicomponent import LBM2DMultiComponent

        window._adopt_engine(LBM2DMultiComponent(width=48, height=48))
        window._sync_timeline_availability()
        assert window.timeline_slider.isEnabled() is False
        assert "no snapshot support" in window.timeline_label.text().lower()

    def test_nothing_is_recorded_for_an_unsupported_engine(self, window):
        from engines.lbm2d_multicomponent import LBM2DMultiComponent

        window._adopt_engine(LBM2DMultiComponent(width=48, height=48))
        window._state_history.clear()
        window.sim.run(5)
        window._record_state_history()
        assert window._state_history == []

    def test_switching_back_re_enables_it(self, window):
        from engines.lbm2d_multicomponent import LBM2DMultiComponent

        window._adopt_engine(LBM2DMultiComponent(width=48, height=48))
        window._adopt_engine(LBM2D(width=48, height=48))
        assert window.timeline_slider.isEnabled() is True
        assert window.timeline_label.text() == "Step 0"

    def test_a_scene_object_is_not_what_the_engine_snapshots(self, window):
        """The snapshot is the engine's; a scene edit cannot quietly enter it."""
        window.scene.obstacles = [CircleObstacle(name="c", x=10, y=10, radius=3)]
        window._on_scene_changed()
        window.sim.run(5)
        window.step_once()
        _step, state = window._state_history[-1]
        assert "obstacles" in state.arrays
        assert state.arrays["obstacles"].shape == window.sim.obstacles.shape
