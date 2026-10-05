"""One canonical pressure quantity, and units that are part of the field metadata.

PLAN.md C35 (P1-2). P1-2 counted five incompatible pressure definitions: `rho/3`,
gauge pressure, an EMA-normalized render, `rho - 1`, and percentile contour levels.
The colormap tooltip read "Pressure field (rho - 1)" while the viewport rendered an
EMA-normalized gauge, and `SimEngine.plot_pressure`'s docstring read "rho - 1" while
its body plotted `rho / 3`.

The factor-of-3 divergence had a single root cause. `get_pressure_gauge()` existed
only on `LBM2D`, so the analysis layer could not call it on any other backend
without `AttributeError`, and each engine compensated by inventing its own display
normalization:

    LBM2D            gauge       (rho - 1) / 3
    LBM3D, GPU       rho - 1                 <-- three times larger, same field

Defining the quantity once on `SimEngine` makes the divergence impossible by
construction rather than by convention. This file pins the definition and the
metadata; C36 routes the per-engine display paths through it, and C43 checks that
every consumer agrees.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from engines.base import SimEngine
from engines.lbm2d import LBM2D
from resources.colormaps import (
    CORE_FIELDS,
    FIELD_REGISTRY,
    field_quantity,
    field_unit,
)


def _engines() -> dict[str, SimEngine]:
    """Every engine constructible without a GPU, keyed by class name."""
    found: dict[str, SimEngine] = {"LBM2D": LBM2D(width=32, height=32)}

    from engines.lbm2d_liquid import LBM2DLiquid
    from engines.lbm2d_multicomponent import LBM2DMultiComponent

    found["LBM2DLiquid"] = LBM2DLiquid(width=24, height=24)
    found["LBM2DMultiComponent"] = LBM2DMultiComponent(width=24, height=24)

    from engines.lbm3d import LBM3D

    found["LBM3D"] = LBM3D(width=12, height=12, depth=6)
    return found


#: Engines whose rest state is genuinely uniform. The multiphase backends seed a
#: droplet (PLAN.md C10), so their rest state contains a phase interface and rho = 2
#: where there is liquid -- asserting zero there would be asserting the absence of
#: physics.
_SINGLE_PHASE = ("LBM2D", "LBM3D")


class TestTheQuantityIsDefinedOnce:
    def test_it_lives_on_the_base_class(self):
        assert "get_pressure_gauge" in vars(SimEngine), (
            "get_pressure_gauge must be a concrete method on SimEngine, not an "
            "LBM2D extension; that is what let every other backend diverge"
        )

    def test_no_engine_overrides_it(self):
        """A backend with a different equation of state changes the constants, not
        the formula. An override would reintroduce exactly the per-engine drift."""
        for name, sim in _engines().items():
            defining = next(
                c for c in type(sim).__mro__ if "get_pressure_gauge" in vars(c)
            )
            assert defining is SimEngine, f"{name} redefines the gauge pressure"

    def test_the_reference_constants_are_lattice_standard(self):
        assert SimEngine.SOUND_SPEED_SQUARED == pytest.approx(1 / 3)
        assert SimEngine.REFERENCE_DENSITY == pytest.approx(1.0)
        assert SimEngine.reference_pressure() == pytest.approx(1 / 3)

    def test_the_docstring_states_the_definition(self):
        """A definition nobody can find is the defect this whole item is about."""
        doc = inspect.getdoc(SimEngine.get_pressure_gauge) or ""
        assert "rho0" in doc
        assert "canonical" in doc.lower()


class TestEveryEngineAgreesOnTheValue:
    def test_every_engine_has_it(self):
        for name, sim in _engines().items():
            gauge = sim.get_pressure_gauge()
            assert isinstance(gauge, np.ndarray), name
            assert gauge.shape == sim.grid_shape, name

    def test_gauge_is_absolute_minus_the_reference(self):
        for name, sim in _engines().items():
            assert np.allclose(
                sim.get_pressure_gauge(),
                sim.get_pressure() - sim.reference_pressure(),
            ), name

    def test_the_resting_state_is_zero(self):
        """A gauge pressure that does not vanish at rest is not a gauge pressure.

        This is the property that makes the density deviation and the gauge
        pressure distinguishable, and it is the 1/3 that every ad-hoc `rho - 1`
        normalization silently threw away.
        """
        engines = _engines()
        for name in _SINGLE_PHASE:
            sim = engines[name]
            sim.initialize(rho=1.0, u=0.0, v=0.0)
            assert np.allclose(sim.get_pressure_gauge(), 0.0, atol=1e-6), name

    def test_a_multiphase_droplet_still_reports_the_gauge_value(self):
        """The multiphase engines seed rho = 2 inside the droplet, so their field is
        not zero at rest. That is physics, not drift: the *definition* still holds,
        which is what this checks.
        """
        for name in ("LBM2DLiquid", "LBM2DMultiComponent"):
            sim = _engines()[name]
            assert sim.get_pressure_gauge().max() > 0.3, name
            assert np.allclose(
                sim.get_pressure_gauge(),
                (sim.get_density() - 1.0) / 3.0,
            ), name

    def test_a_compression_reads_positive(self):
        sim = LBM2D(width=32, height=32)
        sim.rho[:] = 1.2
        assert np.all(sim.get_pressure_gauge() > 0)
        assert np.allclose(sim.get_pressure_gauge(), 0.2 / 3)

    def test_it_is_the_density_deviation_divided_by_three(self):
        """Spelled out because this is the relationship the old code violated."""
        for name, sim in _engines().items():
            rho = sim.get_density()
            assert np.allclose(
                sim.get_pressure_gauge(), (rho - 1.0) / 3.0
            ), f"{name}: gauge pressure must be (rho - 1)/3, not rho - 1"


class TestPlotPressureUsesIt:
    def test_the_docstring_and_the_body_agree(self):
        """They did not: the docstring said `rho - 1`, the body plotted `rho / 3`."""
        source = inspect.getsource(SimEngine.plot_pressure)
        assert "get_pressure_gauge()" in source
        doc = inspect.getdoc(SimEngine.plot_pressure) or ""
        assert "rho - 1" in doc and "p'" in doc
        assert "Absolute lattice" not in doc

    def test_a_resting_state_plots_flat_at_zero(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        sim = LBM2D(width=16, height=16)
        sim.initialize(rho=1.0, u=0.0, v=0.0)

        ax = sim.plot_pressure(colorbar=False)
        image = ax.get_images()[0]
        assert image.get_array().max() == pytest.approx(0.0, abs=1e-6)
        matplotlib.pyplot.close(ax.figure)


class TestFieldUnitsAreMetadata:
    @pytest.mark.parametrize("name", sorted(FIELD_REGISTRY))
    def test_every_field_declares_its_quantity(self, name):
        assert FIELD_REGISTRY[name].quantity, (
            f"{name} has no quantity in its metadata; an unstated definition is "
            "free to drift away from the code"
        )

    def test_the_physical_fields_declare_units(self):
        assert field_unit("pressure") == "lattice"
        assert field_unit("density") == "lattice"
        assert field_unit("speed") == "lattice/step"
        assert field_unit("vorticity") == "1/step"

    def test_dimensionless_fields_declare_no_unit(self):
        for name in ("smoke", "color", "phase"):
            assert field_unit(name) == "", name

    def test_unknown_fields_do_not_raise(self):
        """The field catalog is engine-dependent, so these are called with names
        that may not exist."""
        assert field_unit("nope") == ""
        assert field_quantity("nope") == ""

    def test_the_pressure_definition_names_the_canonical_symbols(self):
        """Both the metadata and the code must speak about `rho0` and `c_s^2`.

        Asserted on the symbols rather than by string-equality of a whole sentence,
        because the prose is expected to drift while the definition is not.
        """
        quantity = field_quantity("pressure")
        doc = inspect.getdoc(SimEngine.get_pressure_gauge) or ""
        for token in ("rho0", "c_s^2"):
            assert token in quantity, token
            assert token in doc, token

    def test_the_tooltip_no_longer_claims_rho_minus_one(self):
        """The specific false claim: it named a quantity the viewport never showed."""
        tooltip = FIELD_REGISTRY["pressure"].tooltip
        assert "field (rho - 1)" not in tooltip
        assert "p'" in tooltip

    def test_symmetry_matches_the_sign_of_the_quantity(self):
        """Gauge pressure and vorticity genuinely change sign; density and speed
        genuinely do not. The flag drives the colorbar, so it is a claim about the
        physics."""
        assert FIELD_REGISTRY["pressure"].symmetric is True
        assert FIELD_REGISTRY["density"].symmetric is False
        assert FIELD_REGISTRY["speed"].symmetric is False

    def test_every_core_field_is_registered(self):
        for name in CORE_FIELDS:
            assert name in FIELD_REGISTRY
            assert FIELD_REGISTRY[name].colormap in (
                "viridis",
                "plasma",
                "inferno",
                "coolwarm",
                "blues",
                "reds",
            )


class TestTheMetadataMatchesTheRegistry:
    def test_the_mode_to_colormap_table_agrees(self):
        from resources.colormaps import MODE_TO_CMAP

        for name, info in FIELD_REGISTRY.items():
            if name in MODE_TO_CMAP:
                assert MODE_TO_CMAP[name] == info.colormap, name

    def test_the_registry_is_frozen(self):
        """`frozen=True` means a consumer cannot annotate a field in place and have
        the change outlive the process."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            FIELD_REGISTRY["pressure"].unit = "Pa"  # type: ignore[misc]
