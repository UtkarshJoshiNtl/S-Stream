"""Snapshot planning and the disk guard (no GPU needed)."""

import pytest

from sstream import snapshots
from sstream.snapshots import (
    FLOW_THROUGH_COUNT,
    check_disk,
    snapshot_bytes,
    snapshot_steps,
    steps_per_flow_through,
)


def test_steps_per_flow_through():
    # 6 m tunnel, 30 m/s freestream, dt = 9.375e-5 s (preview grid)
    assert steps_per_flow_through(6.0, 30.0, 9.375e-5) == pytest.approx(2133.3, rel=1e-3)
    with pytest.raises(ValueError):
        steps_per_flow_through(6.0, 0.0, 1e-3)


def test_snapshot_steps_use_the_flow_through_window():
    steps = snapshot_steps(30_000, 800.0, 5)
    # window = 1600 steps at the end
    assert steps[0] == 28_400
    assert steps[-1] == 30_000
    assert len(steps) == 5


def test_snapshot_steps_never_land_in_the_first_half():
    # preview-like: two flow-throughs exceed the whole run; only the
    # second half may be sampled.
    steps = snapshot_steps(2_200, 2_133.0, 6)
    assert steps[0] >= 1_100
    assert steps[-1] == 2_200
    assert len(steps) == 6


def test_snapshot_steps_zero_requested_writes_nothing():
    assert snapshot_steps(100, 10.0, 0) == []


def test_snapshot_steps_all_inside_the_run():
    steps = snapshot_steps(50, 10.0, 4)
    assert steps[0] >= 1
    assert steps[-1] == 50


def test_snapshot_bytes_scales_with_count():
    one = snapshot_bytes(1_000_000, 1)
    ten = snapshot_bytes(1_000_000, 10)
    assert ten == 10 * one
    # ~6 B per cell (3 components x 2 B) plus a margin
    assert one >= 1_000_000 * 6


def test_check_disk_refuses_when_free_space_is_short(tmp_path):
    with pytest.raises(ValueError, match="refusing snapshots"):
        check_disk(tmp_path, needed=10**30)


def test_check_disk_passes_for_reasonable_estimates(tmp_path):
    assert check_disk(tmp_path, needed=1) >= 1
