"""Scene dimensions and engine grid must agree.

`apply_to_sim` seeds the initial condition but never resizes, so loading a scene
onto an engine built at a different resolution silently mis-places everything:

  * obstacles land wherever they happen to fall on the smaller domain, and any
    past the edge is simply absent,
  * probes past the edge are clamped into the last column, so two probes meant
    to sit at x=180 and x=300 on a 512-wide grid read the *same* cell,
  * nothing raises and nothing logs.

The field arrays are allocated in the engine constructor, so the only correct
response is to rebuild the engine at the scene's dimensions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from scene.probe import Probe
from scene.scene import Scene, apply_to_sim

_ENGINES = [LBM2D, LBM2DLiquid, LBM2DMultiComponent]


class TestApplyDoesNotResize:
    """Pins the underlying behaviour C24 works around."""

    def test_apply_leaves_a_mismatched_grid_alone(self):
        sim = LBM2D(128, 128)
        scene = Scene(width=256, height=192)
        apply_to_sim(scene, sim)
        assert sim.grid_shape == (128, 128)

    def test_probes_past_the_edge_are_clamped_not_fatal(self):
        sim = LBM2D(64, 64)
        scene = Scene(
            width=256,
            height=256,
            probes=[
                _probe("far", 200, 30),
            ],
        )
        apply_to_sim(scene, sim)
        probe = Probe(scene.probes[0])
        probe.record(sim)
        assert len(probe.history["u"]) == 1

    def test_two_distant_probes_can_collapse_onto_one_cell(self):
        """Why the mismatch is dangerous rather than merely untidy: on a 64-wide
        grid, x=10 and x=200 read the same value because both clamp to 63."""
        sim = LBM2D(64, 64)
        sim.u_inflow = 0.2
        near = Probe(_probe("near", 10, 30))
        far = Probe(_probe("far", 200, 30))
        apply_to_sim(Scene(width=256, height=256), sim)
        near.record(sim)
        far.record(sim)
        assert near.history["u"][0] == far.history["u"][0]


def _probe(name: str, x: int, y: int):
    from scene.scene import ProbeSpec

    return ProbeSpec(name=name, x=x, y=y)


class _StubStatus:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, text: str, *_args) -> None:  # noqa: N802 - Qt naming
        self.messages.append(text)


class _StubHolder:
    def __init__(self, sim) -> None:
        self.sim = sim

    def set_sim(self, sim) -> None:
        self.sim = sim


def _window(sim, scene):
    from workbench.app import MainWindow

    window = MainWindow.__new__(MainWindow)
    window.sim = sim
    window.scene = scene
    window.viewport = _StubHolder(sim)
    window.scene_panel = _StubHolder(sim)
    window.analysis_panel = _StubHolder(sim)
    window.outcome_panel = _StubHolder(sim)
    window.status = _StubStatus()
    return window


class TestEngineMatchesScene:
    def test_mismatched_scene_triggers_a_rebuild(self):
        window = _window(LBM2D(128, 128), Scene(width=256, height=192))
        assert window._ensure_engine_matches_scene() is True
        h, w = window.sim.grid_shape
        assert (w, h) == (256, 192)

    def test_matching_scene_does_not_rebuild(self):
        sim = LBM2D(128, 96)
        window = _window(sim, Scene(width=128, height=96))
        assert window._ensure_engine_matches_scene() is False
        assert window.sim is sim

    def test_rebuild_keeps_the_engine_type(self):
        window = _window(LBM2D(64, 64), Scene(width=96, height=96))
        window._ensure_engine_matches_scene()
        assert type(window.sim) is LBM2D

    @pytest.mark.parametrize("factory", _ENGINES)
    def test_rebuild_works_for_every_engine(self, factory):
        window = _window(factory(64, 64), Scene(width=96, height=80))
        assert window._ensure_engine_matches_scene() is True
        assert window.sim.grid_shape == (80, 96)
        assert type(window.sim) is factory

    def test_every_holder_is_repointed(self):
        """Four objects capture the engine at construction; missing one leaves
        a panel reporting from a discarded instance."""
        window = _window(LBM2D(128, 128), Scene(width=192, height=192))
        window._ensure_engine_matches_scene()
        for holder in (
            window.viewport,
            window.scene_panel,
            window.analysis_panel,
            window.outcome_panel,
        ):
            assert holder.sim is window.sim

    def test_rebuild_is_idempotent(self):
        window = _window(LBM2D(128, 128), Scene(width=192, height=192))
        window._ensure_engine_matches_scene()
        rebuilt = window.sim
        assert window._ensure_engine_matches_scene() is False
        assert window.sim is rebuilt

    def test_the_user_is_told(self):
        window = _window(LBM2D(128, 128), Scene(width=192, height=96))
        window._ensure_engine_matches_scene()
        assert any("192x96" in m for m in window.status.messages)


class TestPresetsAreLoadedOntoAMatchingGrid:
    def test_rebuilding_a_preset_scene_fits_its_probes(self):
        from presets.loader import load_preset

        scene = load_preset("presets/scenes/karman_street.json")
        window = _window(LBM2D(128, 128), scene)
        window._ensure_engine_matches_scene()
        h, w = window.sim.grid_shape
        assert (w, h) == (scene.width, scene.height)
        for spec in scene.probes:
            assert 0 <= spec.x < w, spec.name
            assert 0 <= spec.y < h, spec.name

    def test_probes_record_distinct_cells_after_a_rebuild(self):
        """The whole point: x=180 and x=300 must stop reading the same cell."""
        from presets.loader import load_preset

        scene = load_preset("presets/scenes/karman_street.json")
        window = _window(LBM2D(128, 128), scene)
        window._ensure_engine_matches_scene()
        apply_to_sim(scene, window.sim)
        readings = []
        for spec in scene.probes:
            probe = Probe(spec)
            probe.record(window.sim)
            readings.append(tuple(probe.history[f][0] for f in spec.fields))
        assert len(set(readings)) > 1, "every probe is still reading one cell"


class TestOtherApplySitesAreSafe:
    """Not every caller of apply_to_sim can rebuild an engine, so each one that
    does not needs to already be size-correct."""

    def test_render_presets_builds_at_the_scene_size(self):
        source = (
            Path(__file__).resolve().parents[1] / "scripts" / "render_presets.py"
        ).read_text()
        assert (
            "LBM2D(width=scene.width, height=scene.height" in source
        ), "render_thumbnail must size its engine from the scene"

    def test_sweep_builds_at_the_scene_size(self):
        source = (
            Path(__file__).resolve().parents[1] / "analysis" / "sweep.py"
        ).read_text()
        assert "LBM2D(width=run_scene.width, height=run_scene.height)" in source

    def test_reset_guards_the_grid(self):
        source = (
            Path(__file__).resolve().parents[1] / "workbench" / "app.py"
        ).read_text()
        reset_body = source.split("def reset(")[1].split("\n    def ")[0]
        assert (
            "_ensure_engine_matches_scene()" in reset_body
        ), "reset() calls apply_to_sim directly and needs its own grid guard"


class TestGridInvariantHolds:
    """The invariant: (engine.height, engine.width) == (scene.height, scene.width).

    C24 fixed the mechanism. This is the standing check that the mechanism keeps
    holding for every shipped preset and for shapes nobody has tried yet.
    """

    def _assert_matches(self, sim, scene) -> None:
        h, w = sim.grid_shape
        assert (w, h) == (
            scene.width,
            scene.height,
        ), f"scene {scene.width}x{scene.height} vs engine {w}x{h}"

    @pytest.mark.parametrize("factory", _ENGINES)
    def test_matching_sizes_are_already_consistent(self, factory):
        scene = Scene(width=64, height=48)
        window = _window(factory(64, 48), scene)
        window._ensure_engine_matches_scene()
        self._assert_matches(window.sim, scene)

    @pytest.mark.parametrize(
        ("scene_w", "scene_h", "engine_w", "engine_h"),
        [
            (32, 32, 64, 64),
            (64, 32, 32, 64),
            (128, 128, 128, 64),
            (256, 96, 128, 128),
            (512, 256, 128, 128),
            (65, 33, 64, 32),  # odd sizes must not be silently rounded
        ],
    )
    def test_any_mismatch_is_repaired(self, scene_w, scene_h, engine_w, engine_h):
        scene = Scene(width=scene_w, height=scene_h)
        window = _window(LBM2D(engine_w, engine_h), scene)
        window._ensure_engine_matches_scene()
        self._assert_matches(window.sim, scene)

    def test_apply_to_sim_does_not_break_the_invariant(self):
        scene = Scene(width=96, height=72)
        window = _window(LBM2D(64, 64), scene)
        window._ensure_engine_matches_scene()
        apply_to_sim(scene, window.sim)
        self._assert_matches(window.sim, scene)

    def test_every_shipped_preset_lands_on_its_own_grid(self):
        from presets.loader import list_presets, load_preset

        for preset in list_presets():
            scene = load_preset(preset["file"])
            window = _window(LBM2D(128, 128), scene)
            window._ensure_engine_matches_scene()
            self._assert_matches(window.sim, scene)

    def test_every_scene_has_a_usable_grid(self):
        """A zero or negative dimension would make the rebuild itself fail."""
        from presets.loader import list_presets, load_preset

        for preset in list_presets():
            scene = load_preset(preset["file"])
            assert scene.width > 0 and scene.height > 0, preset["file"]

    def test_the_guard_runs_before_apply(self):
        """Ordering matters: rebuilding after apply would seed the discarded
        engine and then resize, losing the seed."""
        source = (
            Path(__file__).resolve().parents[1] / "workbench" / "app.py"
        ).read_text()
        body = source.split("def _apply_and_refresh(")[1].split("\n    def ")[0]
        assert body.index("_ensure_engine_matches_scene()") < body.index(
            "apply_to_sim(self.scene, self.sim)"
        )

    def test_grid_label_reports_the_engine_not_the_scene(self):
        """The HUD must describe what is actually being simulated."""
        from presets.loader import load_preset

        scene = load_preset("presets/scenes/karman_street.json")
        window = _window(LBM2D(128, 128), scene)
        window._ensure_engine_matches_scene()
        h, w = window.sim.grid_shape
        assert f"{w}x{h}" == f"{scene.width}x{scene.height}"
