"""Overlay cache invalidation on scene, engine and window changes.

C29. The streamline, contour and particle overlays each cache pixel coordinates
that were computed from `_canvas_rect()` and the grid shape, and are only
recomputed on every 6th/4th frame. `invalidate_caches()` only cleared
`_tex_init`, so every overlay survived a domain change and a window resize: the
viewport drew a wake from the previous grid at the previous scale, five frames
out of six. The colorbar pixmap had the same problem in a worse place -- it is
keyed on the colormap name but sized from the canvas height, so a resize left a
colorbar of the wrong height pinned to the top of the canvas.

Each test below is written to fail against that implementation.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from engines.lbm2d import LBM2D
from scene.scene import Scene, apply_to_sim

pytest.importorskip("PySide6")

N = 40


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _scene(width=N, height=N):
    return Scene(name="cache", width=width, height=height)


@pytest.fixture
def viewport(app):
    from workbench.viewport import Viewport

    scene = _scene()
    vp = Viewport()
    vp.resize(N * 6, N * 6)
    vp.set_sim(LBM2D(width=N, height=N, viscosity=0.02))
    apply_to_sim(scene, vp.sim)
    vp.set_scene(scene)
    # Offscreen there is no GL context, and delivering a resize event can repaint
    # synchronously. The field upload is the only GL call in that path, and it is
    # not what any test here asserts.
    vp._upload_smoke = lambda: None
    return vp


_OVERLAY_CACHES = (
    "_streamline_cache",
    "_contour_cache",
    "_particle_cache",
    "_colorbar_pixmap",
)


def _populate_caches(vp):
    """Give every overlay cache a distinguishable non-None payload.

    Painting is not needed: the point is that a *populated* cache survives, which
    is the condition under which the stale draw happens.
    """
    vp._streamline_cache = [[(0, 0), (1, 1)]]
    vp._contour_cache = ([(0.0, 0.0, 1.0, 1.0)], 6.0, 6.0, 0.0, 0.0)
    vp._particle_cache = ([(0, 0, 1, 1)], [(2, 2)])
    vp._colorbar_pixmap = object()
    vp._colorbar_cmap = "viridis"


@pytest.mark.parametrize("attr", _OVERLAY_CACHES)
def test_full_invalidate_clears_every_overlay_cache(viewport, attr):
    _populate_caches(viewport)
    viewport.invalidate_caches()
    assert getattr(viewport, attr) is None, (
        f"invalidate_caches() left {attr} populated, so the overlay is drawn "
        "from the previous domain for 5 frames out of 6"
    )


@pytest.mark.parametrize("attr", _OVERLAY_CACHES)
def test_set_scene_clears_every_overlay_cache(viewport, attr):
    """A preset load reaches the viewport through set_scene; a load that only
    refreshed the texture would leave the previous scene's overlays drawn."""
    _populate_caches(viewport)
    viewport.set_scene(_scene(N * 2, N * 2))
    assert getattr(viewport, attr) is None


def test_colormap_key_is_cleared_too(viewport):
    """The colorbar pixmap is keyed on name only, so it must be dropped with its
    key or the stale pixmap is reused under a matching name."""
    _populate_caches(viewport)
    viewport.invalidate_caches()
    assert viewport._colorbar_cmap is None


def test_field_texture_is_cleared(viewport):
    """The GL texture is sized to the grid, which is the original reason this
    method existed."""
    viewport._tex_init = True
    viewport.invalidate_caches()
    assert viewport._tex_init is False


def _resize(vp, w, h):
    """Change the widget size and deliver the matching resize event.

    The offscreen platform does not route a real resize to a QOpenGLWidget, so the
    event is delivered explicitly. `resize()` first, because the event alone does
    not move `_canvas_rect()`. Note that a real size change repaints synchronously
    here, which legitimately refills the caches -- so tests that assert on the
    caches being *empty* use `_deliver_resize` instead.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    old = vp.size()
    vp.resize(w, h)
    vp.resizeEvent(QResizeEvent(QSize(w, h), old))


def _deliver_resize(vp):
    """Deliver a resize event without moving the canvas.

    Qt skips the repaint when the size is unchanged, which isolates the override
    from the paint path.
    """
    from PySide6.QtGui import QResizeEvent

    size = vp.size()
    vp.resizeEvent(QResizeEvent(size, size))


@contextlib.contextmanager
def _no_paint(vp):
    """Suppress the repaint a resize event schedules synchronously offscreen.

    Only the overlay drawing is suppressed; `resizeEvent` itself still runs, which is
    the code under test.
    """
    original = vp._draw_overlay
    vp._draw_overlay = lambda *a, **kw: None
    try:
        yield vp
    finally:
        vp._draw_overlay = original


class TestResizeInvalidatesOverlayGeometry:
    def test_resize_clears_the_overlay_caches(self, viewport):
        _populate_caches(viewport)
        with _no_paint(viewport):
            _deliver_resize(viewport)
        assert viewport._streamline_cache is None
        assert viewport._contour_cache is None
        assert viewport._particle_cache is None

    def test_resize_clears_the_colorbar(self, viewport):
        """The pixmap height is derived from the canvas height but its cache key is
        only the colormap name, so a resize silently kept the old bar height."""
        _populate_caches(viewport)
        with _no_paint(viewport):
            _deliver_resize(viewport)
        assert viewport._colorbar_pixmap is None
        assert viewport._colorbar_cmap is None

    def test_resize_does_not_drop_the_field_texture(self, viewport):
        """The texture is sized to the grid, not the widget, so a window resize
        must not force a reallocation on every resize frame."""
        viewport._tex_init = True
        try:
            with _no_paint(viewport):
                _deliver_resize(viewport)
            assert viewport._tex_init is True
        finally:
            # No GL context exists offscreen; leaving the flag set makes the widget
            # destructor try to delete a texture it never created.
            viewport._tex_init = False


class TestCacheFreshnessIsNotFrameCounterOnly:
    """A populated cache must never outlive the domain that produced it."""

    def test_grid_change_after_rebuild_clears_the_cache(self, viewport):
        """Rebuilding the engine at the scene's grid (the C24 path) is a domain
        change even though no resizeEvent fires."""
        _populate_caches(viewport)
        viewport.set_scene(_scene(N * 2, N * 2))
        viewport.invalidate_caches()
        assert viewport._contour_cache is None
        assert viewport.sim.grid_shape == (N, N)

    def test_toggling_an_overlay_keeps_a_valid_cache(self, viewport):
        """The inverse of the case above, pinned so the invalidation is not widened
        into a per-toggle recomputation: a toggle does not change the domain, so
        reusing the cache is correct."""
        _populate_caches(viewport)
        viewport.set_show_streamlines(False)
        viewport.set_show_streamlines(True)
        assert viewport._streamline_cache is not None


@contextlib.contextmanager
def _raster_painter(vp):
    """A real QPainter onto a QImage, discarded afterwards."""
    from PySide6.QtGui import QImage, QPainter

    img = QImage(max(1, vp.width()), max(1, vp.height()), QImage.Format.Format_RGB32)
    img.fill(0)
    painter = QPainter(img)
    try:
        yield painter
    finally:
        painter.end()


class TestPaintAfterResizeUsesTheNewGeometry:
    """The user-visible consequence of the resize hook."""

    @staticmethod
    def _expected_bar_height(vp):
        cr = vp._canvas_rect()
        return max(40, min(180, int(cr.height()) - 40))

    def test_contours_repaint_at_the_resized_scale(self, viewport):
        """Paint once to populate the cache, resize, then paint on a frame the
        frame-counter would otherwise have served from cache: the cache must
        already describe the new canvas."""
        viewport._show_contours = True
        viewport._overlay_frame = 0
        with _raster_painter(viewport) as painter:
            viewport._draw_pressure_contours(painter)
        first = viewport._contour_cache
        assert first is not None

        _resize(viewport, N * 10, N * 9)
        viewport._overlay_frame = 1  # a cached frame if the resize left one behind
        with _raster_painter(viewport) as painter:
            viewport._draw_pressure_contours(painter)

        cache = viewport._contour_cache
        assert cache is not None
        cr = viewport._canvas_rect()
        assert cache[1] == pytest.approx(cr.width() / N)
        assert cache[2] == pytest.approx(cr.height() / N)

    def test_colorbar_follows_the_new_canvas_height(self, viewport):
        """The pixmap is keyed on the colormap name alone but sized from the canvas
        height, so a resize reused a bar of the wrong height. This one survives even
        a synchronous repaint, because the cache key did not change."""
        _resize(viewport, N * 2, N * 2)
        with _raster_painter(viewport) as painter:
            viewport._draw_colorbar(painter)
        small = viewport._colorbar_pixmap
        assert small is not None
        small_h = small.height()
        assert small_h < 180, "the fixture is too large to see the height change"

        _resize(viewport, N * 10, N * 10)
        with _raster_painter(viewport) as painter:
            viewport._draw_colorbar(painter)

        grown = viewport._colorbar_pixmap
        assert grown is not None
        assert grown.height() == self._expected_bar_height(viewport)
        assert grown.height() != small_h, (
            "the colorbar pixmap kept the pre-resize height; it is cached on the "
            "colormap name, which a resize does not change"
        )

    def test_contours_do_not_replay_a_cache_from_another_domain(self, viewport):
        """Same thing from the other direction: with the cache dropped, the overlay
        recomputes even on a frame the counter would otherwise have skipped."""
        viewport._show_contours = True
        viewport._overlay_frame = 0
        with _raster_painter(viewport) as painter:
            viewport._draw_pressure_contours(painter)
        stale = viewport._contour_cache

        viewport.set_scene(_scene(N * 2, N * 2))
        viewport.invalidate_caches()
        viewport._overlay_frame = 1  # would hit the cached branch if not dropped
        with _raster_painter(viewport) as painter:
            viewport._draw_pressure_contours(painter)
        assert viewport._contour_cache is not None
        assert viewport._contour_cache is not stale
