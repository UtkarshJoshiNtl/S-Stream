"""Probes must be able to record whatever the engine can produce.

`Probe.record` used to test the requested field against a hardcoded
`u`/`v`/`speed`/`pressure` chain and `continue` past anything else, so a probe
asking for a real field silently recorded nothing at all -- no error, no empty
trace visible in the UI, just a channel that never moved.

`presets/scenes/oil_water_separation.json` ships a probe requesting
`color`, `component1` and `component2`. All three were dead.
"""

from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from scene.probe import LOCAL_FIELDS, Probe, recordable_fields
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
