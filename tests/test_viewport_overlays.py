"""Overlay paint-path regression tests with obstacles present.

C13 found that ``_draw_overlay`` rebound its ``obs`` mask parameter to the last
``ObstacleSpec`` dataclass, so the quiver, streamline and force-arrow helpers
subscripted a dataclass instead of an array and raised out of ``paintGL``.

These tests drive the real overlay code with a real ``QPainter`` onto a raster
``QImage``, which needs no GL context, and assert that the mask each helper
receives is the mask the caller passed in.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from scene.scene import (
    CircleObstacle,
    EmitterSpec,
    RectObstacle,
    Scene,
    apply_to_sim,
)

pytest.importorskip("PySide6")

N = 48


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def scene_with_obstacles():
    return Scene(
        name="overlays",
        width=N,
        height=N,
        obstacles=[
            RectObstacle(name="block", x=18, y=20, w=10, h=8),
            CircleObstacle(name="ball", x=8, y=34, radius=4),
        ],
        emitters=[EmitterSpec(name="src", x=2, y=6)],
    )


@pytest.fixture
def sim(scene_with_obstacles):
    engine = LBM2D(width=N, height=N, viscosity=0.02)
    apply_to_sim(scene_with_obstacles, engine)
    for _ in range(4):
        engine.step()
    return engine


@pytest.fixture
def viewport(app, sim, scene_with_obstacles):
    from workbench.viewport import Viewport

    vp = Viewport()
    vp.resize(N * 6, N * 6)
    vp.setMinimumSize(N * 6, N * 6)
    vp.set_sim(sim)
    vp.set_scene(scene_with_obstacles)
    vp.set_show_hud(False)
    return vp


@contextlib.contextmanager
def _raster_paint(vp):
    """Redirect the overlay's internal QPainter onto a QImage.

    `_draw_overlay` constructs `QPainter(self)` and QOpenGLWidget is not a valid
    paint device under the offscreen platform, so the widget symbol is swapped
    for a shim that paints into a raster image instead. This keeps the real
    drawing code under test -- only the paint device changes.
    """
    from PySide6.QtGui import QImage, QPainter

    from workbench import viewport as vpmod

    img = QImage(vp.width(), vp.height(), QImage.Format.Format_RGB32)
    img.fill(0)
    real = vpmod.QPainter

    def shim(device=None):  # noqa: ARG001 - device is deliberately ignored
        return QPainter(img)

    shim.RenderHint = QPainter.RenderHint
    shim.Antialiasing = QPainter.RenderHint.Antialiasing
    vpmod.QPainter = shim
    try:
        yield img
    finally:
        vpmod.QPainter = real


def _paint(vp, vel, obs):
    """Run the real overlay path onto a raster image."""
    with _raster_paint(vp) as img:
        vp._draw_overlay(vel, obs)
    return img


@pytest.fixture
def mask(sim):
    return sim.get_obstacles()


def _spy(vp, name):
    """Capture the `obs` argument a helper is called with."""
    seen = []
    original = getattr(vp, name)

    def wrapper(*args, **kwargs):
        seen.append(args[2] if len(args) > 2 else kwargs.get("obs"))
        return original(*args, **kwargs)

    setattr(vp, name, wrapper)
    return seen


class TestOverlayMaskIsNotShadowed:
    @pytest.mark.parametrize(
        "flag,helper",
        [
            ("_show_quiver", "_draw_quiver"),
            ("_show_streamlines", "_draw_streamlines"),
            ("_show_force_arrows", "_draw_force_arrows"),
        ],
    )
    def test_helper_receives_the_array_mask(
        self, viewport, scene_with_obstacles, mask, flag, helper
    ):
        seen = _spy(viewport, helper)
        setattr(viewport, flag, True)
        _paint(viewport, viewport.sim.get_velocity_view(), mask)

        assert len(seen) == 1, f"{helper} was not called"
        received = seen[0]
        assert isinstance(
            received, np.ndarray
        ), f"{helper} received {type(received).__name__}, not the mask"
        assert received.shape == (N, N)
        assert received.any(), "the mask handed to the helper was empty"

    @pytest.mark.parametrize(
        "flag,helper",
        [
            ("_show_quiver", "_draw_quiver"),
            ("_show_streamlines", "_draw_streamlines"),
            ("_show_force_arrows", "_draw_force_arrows"),
        ],
    )
    def test_overlay_paints_without_raising(self, viewport, mask, flag, helper):
        setattr(viewport, flag, True)
        _paint(viewport, viewport.sim.get_velocity_view(), mask)

    def test_all_overlays_together(self, viewport, mask):
        for name in (
            "_show_quiver",
            "_show_streamlines",
            "_show_contours",
            "_show_force_arrows",
            "_show_hud",
        ):
            setattr(viewport, name, True)
        _paint(viewport, viewport.sim.get_velocity_view(), mask)

    def test_empty_scene_does_not_break_the_mask(self, viewport, mask):
        """The zero-obstacle loop leaves the parameter untouched, so this path
        was already correct; keep it pinned so the rename stays symmetric."""
        viewport.scene.obstacles = []
        seen = _spy(viewport, "_draw_quiver")
        viewport._show_quiver = True
        _paint(viewport, viewport.sim.get_velocity_view(), mask)
        assert isinstance(seen[0], np.ndarray)


class TestDragMomentumDeficit:
    """`_estimate_obstacle_force` summed velocity over the solid cells, where
    bounce-back leaves frozen populations. C13 replaced that with a momentum
    deficit over the body footprint: free stream minus actual, normalised by
    the driving inflow."""

    @staticmethod
    def _body():
        vel = np.zeros((N, N, 2))
        mask = np.zeros((N, N), dtype=bool)
        mask[20:28, 18:28] = True
        return vel, mask

    def test_undisturbed_body_has_no_drag(self):
        from workbench.viewport import Viewport

        vel, mask = self._body()
        vel[..., 0] = 0.05  # fluid passes through at the free-stream speed
        fx, fy = Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.05)
        assert fx == pytest.approx(0.0, abs=1e-12)
        assert fy == pytest.approx(0.0, abs=1e-12)

    def test_fully_stalled_body_is_a_unit_deficit(self):
        from workbench.viewport import Viewport

        vel, mask = self._body()
        vel[..., 0] = 0.05
        vel[mask] = 0.0  # bounce-back body removes all streamwise momentum
        fx, fy = Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.05)
        assert fx == pytest.approx(1.0, rel=1e-6)
        assert fy == pytest.approx(0.0, abs=1e-12)

    def test_deficit_is_scale_invariant(self):
        from workbench.viewport import Viewport

        vel, mask = self._body()
        vel[..., 0] = 0.10
        vel[mask] = 0.02
        fx = Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.10)[0]
        assert fx == pytest.approx(0.8, rel=1e-6)

    def test_drag_points_opposite_the_flow(self):
        from workbench.viewport import Viewport

        vel, mask = self._body()
        vel[..., 0] = 0.05
        vel[mask] = 0.01
        fx, _ = Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.05)
        assert fx > 0.0, "a body slower than free stream must push back on the flow"

    def test_lift_follows_the_cross_stream_momentum(self):
        from workbench.viewport import Viewport

        vel, mask = self._body()
        vel[..., 0] = 0.05
        vel[mask, 1] = -0.02  # body drags the fluid downward
        fx, fy = Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.05)
        assert fx == pytest.approx(0.0, abs=1e-12)
        assert fy == pytest.approx(0.4, rel=1e-6)

    def test_zero_inflow_does_not_divide_by_zero(self):
        from workbench.viewport import Viewport

        vel, mask = self._body()
        vel[..., 0] = 0.05
        fx, fy = Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.0)
        assert np.isfinite(fx) and np.isfinite(fy)

    def test_rect_that_tiles_its_own_box_is_still_measured(self):
        """The sampling window is exactly the obstacle here, so a formula that
        looked for fluid inside the body would find none and report zero."""
        from workbench.viewport import Viewport

        vel, mask = self._body()
        vel[..., 0] = 0.05
        vel[mask] = 0.0
        fx, _ = Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.05)
        assert fx == pytest.approx(1.0, rel=1e-6)

    def test_circular_footprint_ignores_fluid_in_the_box_corners(self):
        from workbench.viewport import Viewport

        vel = np.zeros((N, N, 2))
        mask = np.zeros((N, N), dtype=bool)
        yy, xx = np.ogrid[:N, :N]
        mask[(xx - 24) ** 2 + (yy - 24) ** 2 <= 16] = True
        vel[..., 0] = 0.05
        vel[mask] = 0.0
        # Box is 2r x 2r; the fluid in the four corners must not be averaged in.
        fx, _ = Viewport._estimate_obstacle_force(vel, mask, 20, 20, 8, 8, 0.05)
        assert fx == pytest.approx(1.0, rel=1e-6)

    def test_no_body_in_the_window_reports_zero(self):
        from workbench.viewport import Viewport

        vel = np.zeros((N, N, 2))
        mask = np.zeros((N, N), dtype=bool)
        vel[..., 0] = 0.05
        assert Viewport._estimate_obstacle_force(vel, mask, 18, 20, 10, 8, 0.05) == (
            0.0,
            0.0,
        )
