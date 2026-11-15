"""Renderer helpers: timeline, sampling, frame plan, CLI guard (no GPU)."""

import numpy as np
import pytest

from sstream import render
from sstream.render import (
    GridSampler,
    SmokePlume,
    Timeline,
    frame_times,
    load_velocity,
    origin_of,
)

SPACING = (0.05, 0.05, 0.05)
SHAPE = (160, 48, 36)  # covers the tunnel bounds of meta_synth at SPACING


def meta_synth() -> dict:
    return {
        "tunnel": {
            "bounds": [0.0, 6.0, -1.2, 1.2, 0.0, 1.8],
            "spacing": list(SPACING),
        },
        "placement": {
            "placed_bounds": [[2.4, 3.6], [-0.3, 0.3], [0.0, 0.3]],
        },
    }


def test_origin_of_uses_first_cell_centre():
    origin = origin_of(meta_synth())
    assert origin == pytest.approx((0.025, -1.175, 0.025))


def test_bracket_clamps_and_interpolates():
    tl = Timeline([0, 100, 200], 1e-4)
    assert tl.bracket(-1.0) == (0, 0, 0.0)
    assert tl.bracket(0.03) == (2, 2, 0.0)
    i0, i1, a = tl.bracket(0.015)
    assert (i0, i1) == (1, 2)
    assert a == pytest.approx(0.5)


def test_frame_times_preview_and_full():
    tl = Timeline([0, 100, 200], 1e-4)
    assert frame_times(tl, preview=True) == [0.0, 0.02]
    full = frame_times(tl, preview=False)
    assert len(full) == render.FRAMES_PER_SNAPSHOT_GAP * 2 + 1
    assert full[0] == 0.0 and full[-1] == pytest.approx(0.02)


def test_grid_sampler_is_linear_exact():
    s = 0.05
    field = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for i in range(4):
        field[i, :, :, 0] = (i + 0.5) * s
    sampler = GridSampler(field, (s, s, s), (s / 2, 0.0, 0.0))
    got = sampler(np.array([[0.75 * s, 0.0, 0.0]]))
    # halfway between cell centres 0 and 1 of a linear field
    assert got[0, 0] == pytest.approx(0.75 * s, rel=1e-3)


def test_load_velocity_applies_lattice_to_si_factor(tmp_path):
    raw = np.zeros((2, 2, 2, 3), dtype=np.float32)
    raw[..., 0] = 0.06
    path = tmp_path / "velocity_step_0000001.npz"
    np.savez_compressed(path, velocity=raw)
    field = load_velocity(path, 500.0)
    assert field[0, 0, 0, 0] == pytest.approx(30.0)


def test_smoke_survives_zero_field_and_grows_trails():
    plume = SmokePlume(meta_synth(), np.zeros(SHAPE, dtype=np.uint8))
    plume.emit()
    n0 = len(plume.positions)
    assert n0 == render.SMOKE_EMIT_PER_FRAME
    field = np.zeros(SHAPE + (3,), dtype=np.float32)
    sampler = GridSampler(field, SPACING, origin_of(meta_synth()))
    plume.advect(sampler, 1e-3)
    assert len(plume.positions) == 2 * n0  # killed none, emitted again
    grown = sum(1 for t in plume.trails if len(t) == 2)
    fresh = sum(1 for t in plume.trails if len(t) == 1)
    assert grown == n0 and fresh == render.SMOKE_EMIT_PER_FRAME


def test_smoke_dies_inside_solid():
    plume = SmokePlume(meta_synth(), np.ones(SHAPE, dtype=np.uint8))
    plume.emit()
    field = np.zeros(SHAPE + (3,), dtype=np.float32)
    sampler = GridSampler(field, SPACING, origin_of(meta_synth()))
    plume.advect(sampler, 1e-3)
    # everything died in the solid; only fresh newborns remain, trail-less
    assert len(plume.positions) == render.SMOKE_EMIT_PER_FRAME
    assert all(len(t) == 1 for t in plume.trails)


def test_cli_requires_a_layer():
    with pytest.raises(SystemExit) as exc:
        render._parse_args(["snap", "--out", "x.mp4"])
    assert exc.value.code == 2
