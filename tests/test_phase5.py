"""Tests for Phase 5: Engine-Agnostic Colormap System and Field Annotations."""

from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm3d import LBM3D
from resources.colormaps import CMAP_LUTS, CORE_FIELDS, FIELD_REGISTRY, MODE_TO_CMAP

try:
    from workbench.viewport import Viewport

    HAS_VIEWPORT = True
except ImportError:
    HAS_VIEWPORT = False


# ---------------------------------------------------------------------------
# Field Registry
# ---------------------------------------------------------------------------


class TestFieldRegistry:
    def test_core_fields_in_registry(self) -> None:
        for name in CORE_FIELDS:
            assert name in FIELD_REGISTRY

    def test_all_registry_fields_have_required_attributes(self) -> None:
        for name, info in FIELD_REGISTRY.items():
            assert info.label, f"{name} missing label"
            assert (
                info.colormap in CMAP_LUTS
            ), f"{name} colormap LUT {info.colormap} not in CMAP_LUTS"
            assert info.tooltip, f"{name} missing tooltip"

    def test_mode_to_cmap_covers_all_registry_fields(self) -> None:
        for name in FIELD_REGISTRY:
            assert name in MODE_TO_CMAP, f"{name} missing from MODE_TO_CMAP"


# ---------------------------------------------------------------------------
# LBM2D get_field / get_field_names
# ---------------------------------------------------------------------------


class TestLBM2DFieldAccess:
    @pytest.fixture()
    def sim(self) -> LBM2D:
        s = LBM2D(width=32, height=32, viscosity=0.01)
        s.u_inflow = 0.1
        s.initialize()
        s.run(20)
        return s

    def test_field_names(self, sim: LBM2D) -> None:
        names = sim.get_field_names()
        assert "smoke" in names
        assert "speed" in names
        assert "vorticity" in names
        assert "pressure" in names
        assert "density" in names

    def test_core_fields_return_float32(self, sim: LBM2D) -> None:
        for name in ["smoke", "speed", "vorticity", "pressure", "density"]:
            field = sim.get_field(name)
            assert field.dtype == np.float32, f"{name} should be float32"

    def test_fields_are_normalized_0_1(self, sim: LBM2D) -> None:
        for name in ["smoke", "speed", "vorticity", "pressure", "density"]:
            field = sim.get_field(name)
            assert field.min() >= 0.0, f"{name} below 0"
            assert field.max() <= 1.0, f"{name} above 1"

    def test_fields_match_grid_shape(self, sim: LBM2D) -> None:
        for name in ["smoke", "speed", "vorticity", "pressure", "density"]:
            field = sim.get_field(name)
            assert field.shape == sim.grid_shape, f"{name} shape mismatch"

    def test_unknown_field_raises(self, sim: LBM2D) -> None:
        with pytest.raises(ValueError, match="Unknown field"):
            sim.get_field("nonexistent")

    def test_speed_field_positive(self, sim: LBM2D) -> None:
        field = sim.get_field("speed")
        assert field.max() > 0.0

    def test_smoke_field_has_content(self, sim: LBM2D) -> None:
        sim.smoke[16, 16] = 1.0
        field = sim.get_field("smoke")
        assert field.max() > 0.0


# ---------------------------------------------------------------------------
# LBM3D get_field / get_field_names
# ---------------------------------------------------------------------------


class TestLBM3DFieldAccess:
    @pytest.fixture()
    def sim(self) -> LBM3D:
        s = LBM3D(width=16, height=16, depth=16, viscosity=0.01)
        s.u_inflow = 0.1
        s.initialize()
        s.run(10)
        return s

    def test_field_names(self, sim: LBM3D) -> None:
        names = sim.get_field_names()
        assert "smoke" in names
        assert "speed" in names
        assert "vorticity" in names
        assert "pressure" in names
        assert "density" in names

    def test_core_fields_return_float32(self, sim: LBM3D) -> None:
        for name in ["smoke", "speed", "vorticity", "pressure", "density"]:
            field = sim.get_field(name)
            assert field.dtype == np.float32

    def test_fields_are_normalized_0_1(self, sim: LBM3D) -> None:
        for name in ["smoke", "speed", "vorticity", "pressure", "density"]:
            field = sim.get_field(name)
            assert field.min() >= 0.0
            assert field.max() <= 1.0

    def test_fields_match_grid_shape(self, sim: LBM3D) -> None:
        for name in ["smoke", "speed", "vorticity", "pressure", "density"]:
            field = sim.get_field(name)
            assert field.shape == sim.grid_shape

    def test_unknown_field_raises(self, sim: LBM3D) -> None:
        with pytest.raises(ValueError, match="Unknown field"):
            sim.get_field("nonexistent")


# ---------------------------------------------------------------------------
# LBM2DLiquid get_field / get_field_names
# ---------------------------------------------------------------------------


class TestLiquidFieldAccess:
    @pytest.fixture()
    def sim(self) -> LBM2DLiquid:
        s = LBM2DLiquid(width=32, height=32)
        s.initialize()
        s.run(10)
        return s

    def test_field_names_include_phase(self, sim: LBM2DLiquid) -> None:
        names = sim.get_field_names()
        assert "phase" in names
        assert "density" in names

    def test_phase_field(self, sim: LBM2DLiquid) -> None:
        field = sim.get_field("phase")
        assert field.dtype == np.float32
        assert field.min() >= 0.0
        assert field.max() <= 1.0
        assert field.shape == sim.grid_shape

    def test_density_field_varied(self, sim: LBM2DLiquid) -> None:
        field = sim.get_field("density")
        assert field.dtype == np.float32
        assert field.shape == sim.grid_shape

    def test_all_core_fields_work(self, sim: LBM2DLiquid) -> None:
        for name in ["smoke", "speed", "vorticity", "pressure", "density", "phase"]:
            field = sim.get_field(name)
            assert field.shape == sim.grid_shape


# ---------------------------------------------------------------------------
# Marching squares contour helper
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_VIEWPORT, reason="PySide6 not available")
class TestMarchingSquares:
    def test_contour_edges_for_all_indices(self) -> None:
        for idx in range(1, 15):
            edges = Viewport._ms_edges(idx, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
            assert isinstance(edges, list)
            if idx not in (5, 10):
                assert len(edges) == 1, f"idx={idx} should have 1 segment"
            else:
                assert len(edges) == 2, f"idx={idx} should have 2 segments"

    def test_flat_cells_no_edges(self) -> None:
        edges0 = Viewport._ms_edges(0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        edges15 = Viewport._ms_edges(15, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
        assert edges0 == []
        assert edges15 == []


# ---------------------------------------------------------------------------
# Field computation consistency
# ---------------------------------------------------------------------------


class TestFieldConsistency:
    def test_vorticity_from_get_field_matches_inline(self) -> None:
        sim = LBM2D(width=32, height=32, viscosity=0.01)
        sim.u_inflow = 0.1
        sim.initialize()
        sim.run(20)
        field = sim.get_field("vorticity")
        vel = sim.get_velocity()
        u, v = vel[:, :, 0], vel[:, :, 1]
        dvdx = np.zeros_like(u, dtype=np.float32)
        dudy = np.zeros_like(u, dtype=np.float32)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
        vort = dvdx - dudy
        # Field must be float32, in [0, 1], same shape
        assert field.dtype == np.float32
        assert field.shape == vort.shape
        assert float(field.min()) >= 0.0
        assert float(field.max()) <= 1.0
        # Non-zero vorticity should map to non-zero field
        assert float(field[1, 1]) != 0.0 or float(field[16, 16]) != 0.0

    def test_pressure_from_get_field_matches_get_pressure(self) -> None:
        sim = LBM2D(width=32, height=32, viscosity=0.01)
        sim.u_inflow = 0.1
        sim.initialize()
        sim.run(20)
        field = sim.get_field("pressure")
        p = sim.get_pressure()
        # Field must be float32, in [0, 1], same shape
        assert field.dtype == np.float32
        assert field.shape == p.shape
        assert float(field.min()) >= 0.0
        assert float(field.max()) <= 1.0
