"""Probes must be able to record whatever the engine can produce.

`Probe.record` used to test the requested field against a hardcoded
`u`/`v`/`speed`/`pressure` chain and `continue` past anything else, so a probe
asking for a real field silently recorded nothing at all -- no error, no empty
trace visible in the UI, just a channel that never moved.

`presets/scenes/oil_water_separation.json` ships a probe requesting
`color`, `component1` and `component2`. All three were dead.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from scene.probe import (
    LOCAL_FIELDS,
    Probe,
    default_probe_fields,
    recordable_fields,
    validate_probe_fields,
    validate_scene_probes,
)
from scene.scene import ProbeSpec


def _probe(x=10, y=10, fields=("u", "v")) -> Probe:
    return Probe(ProbeSpec(name="p", x=x, y=y, fields=list(fields)))


def _stepped(sim, n=2):
    for _ in range(n):
        sim.step()
    return sim


class TestRecordableFields:
    def test_local_quantities_come_first(self):
        assert recordable_fields(LBM2D(32, 32))[: len(LOCAL_FIELDS)] == list(
            LOCAL_FIELDS
        )

    def test_every_engine_field_is_recordable(self):
        sim = LBM2D(32, 32)
        for name in sim.get_field_names():
            assert name in recordable_fields(sim)

    def test_multiphase_fields_are_recordable(self):
        sim = LBM2DMultiComponent(32, 32)
        for name in ("color", "component1", "component2", "phase"):
            assert name in recordable_fields(sim)

    def test_no_duplicates(self):
        names = recordable_fields(LBM2DMultiComponent(32, 32))
        assert len(names) == len(set(names))

    def test_local_speed_is_not_shadowed_by_the_engine_field(self):
        """get_field("speed") is a normalised display value; the probe series is
        a physical magnitude. Both are offered, and they must not collide."""
        names = recordable_fields(LBM2D(32, 32))
        assert names.count("speed") == 1


class TestRecordingEngineFields:
    def test_engine_field_records_a_value(self):
        sim = _stepped(LBM2D(32, 32))
        probe = _probe(fields=["density"])
        probe.record(sim)
        assert len(probe.history["density"]) == 1
        assert np.isfinite(probe.history["density"][0])

    def test_value_matches_the_field_array(self):
        sim = _stepped(LBM2D(32, 32))
        probe = _probe(x=12, y=8, fields=["pressure"])
        probe.record(sim)
        assert probe.history["pressure"][0] == pytest.approx(
            float(sim.get_field("pressure")[8, 12]), rel=1e-5
        )

    def test_multiphase_preset_channels_now_record(self):
        """The exact preset that shipped three dead channels."""
        sim = _stepped(LBM2DMultiComponent(32, 32))
        probe = _probe(x=6, y=6, fields=["color", "component1", "component2"])
        probe.record(sim)
        for name in ("color", "component1", "component2"):
            assert len(probe.history[name]) == 1, name
            assert probe.unavailable_fields == set()

    def test_speed_stays_physical_not_normalised(self):
        """Regression guard: the probe speed series must not silently switch to
        the engine's 0..1 display value."""
        sim = _stepped(LBM2D(32, 32))
        probe = _probe(x=5, y=5, fields=["speed"])
        probe.record(sim)
        physical = float(np.hypot(*sim.get_velocity_at(5, 5)))
        assert probe.history["speed"][0] == pytest.approx(physical, rel=1e-5)


class TestSharedCache:
    def test_cache_avoids_recomputing_the_field(self):
        sim = _stepped(LBM2D(32, 32))
        probes = [_probe(x=i, y=3, fields=["density"]) for i in range(4)]
        cache: dict[str, np.ndarray] = {}
        for probe in probes:
            probe.record(sim, cache)
        assert len(cache) == 1, "field should be fetched once for all probes"

    def test_cached_and_uncached_agree(self):
        sim = _stepped(LBM2D(32, 32))
        a = _probe(x=7, y=9, fields=["density"])
        b = _probe(x=7, y=9, fields=["density"])
        a.record(sim, {})
        b.record(sim)
        assert a.history["density"][0] == pytest.approx(b.history["density"][0])

    def test_one_cache_serves_all_probe_types(self):
        sim = _stepped(LBM2D(32, 32))
        cache: dict[str, np.ndarray] = {}
        _probe(fields=["density"]).record(sim, cache)
        _probe(fields=["vorticity", "density"]).record(sim, cache)
        assert set(cache) == {"density", "vorticity"}


class TestUnavailableFieldsAreReported:
    def test_missing_field_is_logged_once_not_silently_dropped(self, caplog):
        """Thermal is off, so `temperature` is unproducible. C20 made the engine
        refuse it; the probe must report that rather than quietly skipping."""
        sim = _stepped(LBM2D(32, 32))
        probe = _probe(fields=["temperature", "density"])
        with caplog.at_level("ERROR"):
            for _ in range(5):
                sim.step()
                probe.record(sim)
        errors = [r for r in caplog.records if "temperature" in r.getMessage()]
        assert len(errors) == 1, "should report once, then stop retrying"
        assert probe.unavailable_fields == {"temperature"}

    def test_the_available_channel_keeps_recording(self, caplog):
        """One bad field must not take the rest of the probe down with it."""
        sim = _stepped(LBM2D(32, 32))
        probe = _probe(fields=["temperature", "density"])
        with caplog.at_level("ERROR"):
            for _ in range(5):
                sim.step()
                probe.record(sim)
        assert len(probe.history["density"]) == 5
        assert len(probe.history["temperature"]) == 0

    def test_reported_field_is_excluded_from_recordable(self):
        sim = LBM2D(32, 32)
        sim.thermal_enabled = False
        assert "temperature" not in recordable_fields(sim)

    def test_clear_resets_the_reported_set(self):
        probe = _probe(fields=["density"])
        probe._unavailable.add("density")
        probe.clear()
        assert probe.unavailable_fields == set()


class TestLifecycle:
    def test_clear_empties_every_channel(self):
        sim = _stepped(LBM2D(32, 32))
        probe = _probe(fields=["u", "density"])
        probe.record(sim)
        assert probe.step == 1
        probe.clear()
        assert probe.step == 0
        for series in probe.history.values():
            assert len(series) == 0

    def test_history_is_bounded(self):
        from scene.probe import MAX_HISTORY

        sim = _stepped(LBM2D(32, 32))
        probe = _probe(fields=["density"])
        for _ in range(MAX_HISTORY + 25):
            sim.step()
            probe.record(sim)
        assert len(probe.history["density"]) == MAX_HISTORY

    def test_out_of_range_coordinates_are_clamped_not_fatal(self):
        sim = _stepped(LBM2D(32, 32))
        probe = _probe(x=999, y=999, fields=["u", "density"])
        probe.record(sim)
        assert len(probe.history["density"]) == 1


class TestEveryEngineStillWorks:
    @pytest.mark.parametrize("factory", [LBM2D, LBM2DLiquid, LBM2DMultiComponent])
    def test_all_declared_fields_are_recordable_on_every_engine(self, factory):
        sim = _stepped(factory(32, 32))
        names = recordable_fields(sim)
        probe = _probe(fields=list(names))
        probe.record(sim)
        for name in names:
            assert len(probe.history[name]) == 1, f"{name} did not record"
            assert np.isfinite(probe.history[name][0]), name


class TestFieldValidation:
    """C22: a probe asking for a field the engine cannot produce must be caught
    when the scene is applied, not left as a permanently flat trace."""

    def test_a_sound_probe_reports_nothing(self):
        sim = _stepped(LBM2D(32, 32))
        spec = ProbeSpec(name="ok", x=4, y=4, fields=["u", "density"])
        assert validate_probe_fields(sim, spec) == []

    def test_multiphase_probe_is_invalid_on_plain_lbm2d(self):
        sim = LBM2D(32, 32)
        spec = ProbeSpec(name="Phase probe", x=6, y=6, fields=["color", "component1"])
        assert validate_probe_fields(sim, spec) == ["color", "component1"]

    def test_the_same_probe_is_valid_on_its_own_engine(self):
        sim = LBM2DMultiComponent(32, 32)
        spec = ProbeSpec(name="Phase probe", x=6, y=6, fields=["color", "component1"])
        assert validate_probe_fields(sim, spec) == []

    def test_engine_swap_is_the_regression(self):
        """A scene authored on the multicomponent engine, then opened on LBM2D."""
        mc = _stepped(LBM2DMultiComponent(32, 32))
        spec = ProbeSpec(name="p", x=6, y=6, fields=["color", "component1"])
        assert validate_probe_fields(mc, spec) == []
        assert validate_probe_fields(LBM2D(32, 32), spec) == [
            "color",
            "component1",
        ]

    def test_scene_validation_names_the_probe_and_the_engine(self):
        spec = ProbeSpec(name="Phase probe", x=6, y=6, fields=["component1"])
        scene = SimpleNamespace(probes=[spec])
        problems = validate_scene_probes(LBM2D(32, 32), scene)
        assert len(problems) == 1
        assert "Phase probe" in problems[0]
        assert "component1" in problems[0]
        assert "LBM2D" in problems[0]

    def test_scene_with_no_probes_is_clean(self):
        assert validate_scene_probes(LBM2D(32, 32), SimpleNamespace(probes=[])) == []

    def test_scene_without_a_probes_attribute_is_clean(self):
        assert validate_scene_probes(LBM2D(32, 32), SimpleNamespace()) == []

    def test_unavailable_engine_field_is_caught(self):
        sim = LBM2D(32, 32)
        sim.thermal_enabled = False
        spec = ProbeSpec(name="p", x=1, y=1, fields=["temperature"])
        assert validate_probe_fields(sim, spec) == ["temperature"]


class TestPlacementDefaults:
    def test_defaults_are_all_recordable_on_lbm2d(self):
        sim = LBM2D(32, 32)
        assert (
            validate_probe_fields(
                sim, ProbeSpec(name="n", fields=default_probe_fields(sim))
            )
            == []
        )

    def test_defaults_are_all_recordable_on_every_engine(self):
        for factory in (LBM2D, LBM2DLiquid, LBM2DMultiComponent):
            sim = factory(32, 32)
            spec = ProbeSpec(name="n", fields=default_probe_fields(sim))
            assert validate_probe_fields(sim, spec) == [], factory.__name__

    def test_defaults_always_include_the_local_quantities(self):
        for factory in (LBM2D, LBM2DLiquid, LBM2DMultiComponent):
            names = default_probe_fields(factory(32, 32))
            for local in ("u", "v", "speed"):
                assert local in names, f"{factory.__name__} lost {local}"

    def test_defaults_are_unique(self):
        for factory in (LBM2D, LBM2DLiquid, LBM2DMultiComponent):
            names = default_probe_fields(factory(32, 32))
            assert len(names) == len(set(names))


class TestEveryPresetProbeIsValid:
    """Every shipped preset must declare fields that some engine can record.

    Note what is *not* available here: no preset records which engine it expects
    (`product.engine` is absent from all ten). The engine is chosen at startup by
    a CLI flag, so a preset cannot be validated against a declared engine at load
    time. That is why the app validates at apply time instead -- open
    oil_water_separation on a plain LBM2D and the mismatch is reported rather
    than showing three dead channels.
    """

    ENGINES = {
        "LBM2D": LBM2D,
        "LBM2DLiquid": LBM2DLiquid,
        "LBM2DMultiComponent": LBM2DMultiComponent,
    }

    def _presets(self):
        import json
        from pathlib import Path

        directory = Path(__file__).resolve().parents[1] / "presets" / "scenes"
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text())
            yield path.name, [ProbeSpec(**p) for p in data.get("probes", [])]

    def test_every_preset_probe_runs_on_at_least_one_engine(self):
        failures = []
        for filename, specs in self._presets():
            for spec in specs:
                if not any(
                    not validate_probe_fields(factory(32, 32), spec)
                    for factory in self.ENGINES.values()
                ):
                    failures.append(f"{filename}: {spec.name} {spec.fields}")
        assert not failures, "probe fields no engine can record:\n" + "\n".join(
            failures
        )

    def test_ordinary_presets_are_valid_on_plain_lbm2d(self):
        """Most presets should not depend on a special engine."""
        for filename, specs in self._presets():
            if "oil_water" in filename:
                continue
            sim = LBM2D(32, 32)
            for spec in specs:
                assert (
                    validate_probe_fields(sim, spec) == []
                ), f"{filename}: {spec.name}"

    def test_the_multiphase_preset_requires_the_multicomponent_engine(self):
        """Pins the engine pairing that the preset file itself does not declare."""
        specs = dict(self._presets())["oil_water_separation.json"]
        assert specs, "fixture expects a probe"
        for spec in specs:
            assert validate_probe_fields(LBM2DMultiComponent(32, 32), spec) == []
            assert validate_probe_fields(LBM2D(32, 32), spec), (
                "expected the multicomponent-only fields to be rejected "
                "on plain LBM2D"
            )

    def test_no_preset_declares_an_engine_yet(self):
        """Records the gap rather than asserting it is good: once presets carry
        `product.engine`, this test should be replaced by a per-preset check."""
        import json
        from pathlib import Path

        directory = Path(__file__).resolve().parents[1] / "presets" / "scenes"
        for path in sorted(directory.glob("*.json")):
            product = json.loads(path.read_text()).get("product", {})
            assert "engine" not in product, path.name
