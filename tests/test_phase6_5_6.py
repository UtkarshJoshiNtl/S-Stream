"""Tests for Phase 6.5 (Jupyter Integration) and 6.6 (REST API)."""
from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d import LBM2D


# ---------------------------------------------------------------------------
# 6.5 Jupyter Integration — plot methods on SimEngine
# ---------------------------------------------------------------------------


class TestPlotMethods:
    @pytest.fixture()
    def sim(self) -> LBM2D:
        s = LBM2D(width=32, height=32, viscosity=0.02)
        s.u_inflow = 0.1
        s.initialize()
        s.run(10)
        return s

    def test_plot_field_returns_axes(self, sim: LBM2D) -> None:
        import matplotlib.pyplot as plt

        ax = sim.plot_field("smoke")
        assert ax is not None
        assert hasattr(ax, "set_title")
        plt.close("all")

    def test_plot_field_with_existing_axes(self, sim: LBM2D) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = sim.plot_field("speed", ax=ax)
        assert result is ax
        plt.close("all")

    def test_plot_velocity(self, sim: LBM2D) -> None:
        import matplotlib.pyplot as plt

        ax = sim.plot_velocity()
        assert ax is not None
        plt.close("all")

    def test_plot_pressure(self, sim: LBM2D) -> None:
        import matplotlib.pyplot as plt

        ax = sim.plot_pressure()
        assert ax is not None
        plt.close("all")

    def test_plot_smoke(self, sim: LBM2D) -> None:
        import matplotlib.pyplot as plt

        ax = sim.plot_smoke()
        assert ax is not None
        plt.close("all")

    def test_plot_vorticity(self, sim: LBM2D) -> None:
        import matplotlib.pyplot as plt

        ax = sim.plot_vorticity()
        assert ax is not None
        plt.close("all")

    def test_plot_unknown_field_raises(self, sim: LBM2D) -> None:
        with pytest.raises(ValueError, match="Unknown field"):
            sim.plot_field("nonexistent")


class TestReprMethods:
    @pytest.fixture()
    def sim(self) -> LBM2D:
        s = LBM2D(width=32, height=32, viscosity=0.02)
        s.u_inflow = 0.1
        s.initialize()
        s.run(10)
        return s

    def test_repr_png_returns_bytes(self, sim: LBM2D) -> None:
        png = sim._repr_png_()
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"

    def test_repr_html_returns_img_tag(self, sim: LBM2D) -> None:
        html = sim._repr_html_()
        assert html is not None
        assert '<img src="data:image/png;base64,' in html


# ---------------------------------------------------------------------------
# 6.5 sstream package import
# ---------------------------------------------------------------------------


class TestSStreamPackage:
    def test_import_sstream(self) -> None:
        import sstream
        assert hasattr(sstream, "LBM2D")
        assert hasattr(sstream, "LBM3D")
        assert hasattr(sstream, "LBM2DLiquid")
        assert hasattr(sstream, "SimEngine")
        assert hasattr(sstream, "ParticleTracer")

    def test_sstream_version(self) -> None:
        import sstream
        assert sstream.__version__ == "0.3.0"

    def test_sstream_lbm2d_works(self) -> None:
        import sstream
        sim = sstream.LBM2D(width=16, height=16)
        sim.run(5)
        assert sim.grid_shape == (16, 16)


# ---------------------------------------------------------------------------
# 6.6 REST API
# ---------------------------------------------------------------------------


class TestRESTAPI:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from engines.api import create_app

        sim = LBM2D(width=32, height=32, viscosity=0.02)
        sim.u_inflow = 0.1
        sim.initialize()
        app = create_app(sim)
        return TestClient(app), sim

    def test_root(self, client) -> None:
        http, sim = client
        r = http.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["engine"] == "LBM2D"
        assert "speed" in data["fields"]

    def test_engine_info(self, client) -> None:
        http, sim = client
        r = http.get("/engine")
        assert r.status_code == 200
        data = r.json()
        assert data["ndim"] == 2
        assert data["viscosity"] == pytest.approx(0.02)

    def test_run(self, client) -> None:
        http, sim = client
        r = http.post("/run", json={"steps": 10})
        assert r.status_code == 200
        data = r.json()
        assert data["steps"] == 10
        assert data["elapsed_s"] > 0

    def test_field(self, client) -> None:
        http, sim = client
        r = http.get("/field?type=speed")
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "speed"
        assert data["min"] >= 0.0
        assert data["max"] <= 1.0

    def test_field_array(self, client) -> None:
        http, sim = client
        r = http.get("/field/array?type=smoke")
        assert r.status_code == 200
        data = r.json()
        assert len(data["data"]) == 32
        assert len(data["data"][0]) == 32

    def test_field_unknown(self, client) -> None:
        http, sim = client
        r = http.get("/field?type=nonexistent")
        assert r.status_code == 400

    def test_field_png(self, client) -> None:
        http, sim = client
        r = http.get("/field/png?type=smoke")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:4] == b"\x89PNG"

    def test_velocity(self, client) -> None:
        http, sim = client
        r = http.get("/velocity")
        assert r.status_code == 200
        data = r.json()
        assert data["speed_mean"] >= 0.0

    def test_pressure(self, client) -> None:
        http, sim = client
        r = http.get("/pressure")
        assert r.status_code == 200
        data = r.json()
        assert "mean" in data

    def test_obstacles(self, client) -> None:
        http, sim = client
        r = http.get("/obstacles")
        assert r.status_code == 200

    def test_add_obstacle(self, client) -> None:
        http, sim = client
        r = http.post("/obstacle", json={"x": 16, "y": 16, "radius": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_clear_obstacles(self, client) -> None:
        http, sim = client
        http.post("/obstacle", json={"x": 16, "y": 16, "radius": 5})
        r = http.delete("/obstacles")
        assert r.status_code == 200

    def test_add_emitter(self, client) -> None:
        http, sim = client
        r = http.post("/emitter", json={"x": 2, "y": 16, "strength": 0.05})
        assert r.status_code == 200

    def test_clear_emitters(self, client) -> None:
        http, sim = client
        http.post("/emitter", json={"x": 2, "y": 16, "strength": 0.05})
        r = http.delete("/emitters")
        assert r.status_code == 200

    def test_probe(self, client) -> None:
        http, sim = client
        r = http.get("/probe?type=speed")
        assert r.status_code == 200
        data = r.json()
        assert "center" in data
        assert "mean" in data
