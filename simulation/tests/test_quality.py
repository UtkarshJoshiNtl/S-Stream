"""Quality presets, the cell scan, and the VRAM guard (no GPU needed)."""

import pytest

from sstream.quality import (
    BYTES_PER_CELL,
    VRAM_FRACTION_LIMIT,
    cells_streamwise_for_target,
    check_vram,
    format_bytes,
    free_device_bytes,
    grid_cells,
    resolve_target_cells,
    vram_estimate_bytes,
)
from sstream.schema import TunnelSpec


def test_quality_and_cells_resolve_to_a_target():
    assert resolve_target_cells("low", None) == 1_000_000
    assert resolve_target_cells("ultra", None) == 17_000_000
    assert resolve_target_cells("high", 12345) == 12345  # --cells wins
    assert resolve_target_cells(None, None) is None  # preset grid
    with pytest.raises(ValueError):
        resolve_target_cells("bogus", None)
    with pytest.raises(ValueError):
        resolve_target_cells(None, 4)


@pytest.mark.parametrize("target", [50_000, 1_000_000, 3_000_000, 17_000_000])
def test_cells_streamwise_is_the_largest_grid_that_fits(target):
    tunnel = TunnelSpec()
    nx = cells_streamwise_for_target(tunnel, target)
    assert grid_cells(tunnel.with_resolution(nx)) <= target
    assert grid_cells(tunnel.with_resolution(nx + 1)) > target


def test_vram_estimate_is_bytes_per_cell_times_cells():
    assert vram_estimate_bytes(1_000_000) == int(1_000_000 * BYTES_PER_CELL)


def test_free_device_bytes_env_override(monkeypatch):
    monkeypatch.setenv("SSTREAM_FREE_MEM_BYTES", "123456789")
    assert free_device_bytes("cuda") == 123456789


def test_free_device_bytes_unknown_off_cuda(monkeypatch):
    monkeypatch.delenv("SSTREAM_FREE_MEM_BYTES", raising=False)
    assert free_device_bytes("cpu") is None


def test_check_vram_passes_and_refuses(monkeypatch):
    monkeypatch.setenv("SSTREAM_FREE_MEM_BYTES", str(1_000_000_000))
    assert check_vram(1_000_000, "cuda") == (
        vram_estimate_bytes(1_000_000),
        1_000_000_000,
    )
    with pytest.raises(ValueError, match=f"{VRAM_FRACTION_LIMIT:.0%}"):
        check_vram(10_000_000, "cuda")


def test_check_vram_none_when_free_unknown(monkeypatch):
    monkeypatch.delenv("SSTREAM_FREE_MEM_BYTES", raising=False)
    assert check_vram(1_000_000, "cpu") is None


def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert format_bytes(184_645_180) == "184.6 MB"
    assert format_bytes(3_500_000_000) == "3.5 GB"
