"""Quality presets, the cell budget, and the VRAM guard.

``--quality`` picks a target cell count (~1M/3M/9M/17M for
low/medium/high/ultra) and ``--cells`` overrides it with an exact
number; either way the streamwise resolution is the largest whole-cell
grid that fits the target, so the cross-section stays square-ish and
the tunnel length is kept.

The VRAM estimate is ``BYTES_PER_CELL`` bytes per cell — populations
plus velocity/density/force fields plus the solid mask — and it is
compared against the *actual* free memory of the Warp device, read
through the CUDA driver API.  Nothing is hardcoded: if the driver is
unreachable the guard is skipped and the run proceeds unguarded.
"""

from __future__ import annotations

import ctypes
import os

from .schema import TunnelSpec

QUALITY_CELLS = {
    "low": 1_000_000,
    "medium": 3_000_000,
    "high": 9_000_000,
    "ultra": 17_000_000,
}

#: Rough bytes of device memory per lattice cell (19 populations plus
#: rho/u/force/solid fields and solver scratch).
BYTES_PER_CELL = 185.0

#: Refuse a run whose estimate would exceed this fraction of free VRAM.
VRAM_FRACTION_LIMIT = 0.90

#: Steps timed before the estimate is printed (the warm-up).
WARMUP_STEPS = 50

_FREE_MEM_ENV = "SSTREAM_FREE_MEM_BYTES"


def resolve_target_cells(quality: str | None, cells: int | None) -> int | None:
    """Target cell count from ``--quality``/``--cells`` (None = use preset)."""
    if cells is not None:
        if cells < 8:
            raise ValueError("cells must be at least 8")
        return int(cells)
    if quality is not None:
        if quality not in QUALITY_CELLS:
            raise ValueError(f"quality must be one of {sorted(QUALITY_CELLS)}")
        return QUALITY_CELLS[quality]
    return None


def grid_cells(tunnel: TunnelSpec) -> int:
    return tunnel.grid_shape[0] * tunnel.grid_shape[1] * tunnel.grid_shape[2]


def cells_streamwise_for_target(tunnel: TunnelSpec, target: int) -> int:
    """Largest streamwise resolution whose whole grid fits ``target`` cells."""
    if grid_cells(tunnel.with_resolution(8)) > target:
        return 8
    lo, hi = 8, 8
    while grid_cells(tunnel.with_resolution(hi)) <= target:
        lo = hi
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if grid_cells(tunnel.with_resolution(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def vram_estimate_bytes(cells: int) -> int:
    return int(cells * BYTES_PER_CELL)


def free_device_bytes(device: str) -> int | None:
    """Free memory of the Warp device in bytes, or None if unavailable.

    Warp 1.18 exposes no free-memory query, so the CUDA driver API is
    called directly.  ``SSTREAM_FREE_MEM_BYTES`` overrides the reading,
    which is what the tests use on machines without a GPU.
    """
    override = os.environ.get(_FREE_MEM_ENV)
    if override:
        return int(override)
    if not device.startswith("cuda"):
        return None
    library = None
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            library = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if library is None:
        return None
    # CUDA 12 has no default context; retain and select the primary one or
    # cuMemGetInfo_v2 returns CUDA_ERROR_INVALID_CONTEXT.
    try:
        index = int(device.split(":", 1)[1]) if ":" in device else 0
    except ValueError:
        index = 0
    dev = ctypes.c_int()
    ctx = ctypes.c_void_p()
    try:
        if library.cuInit(0) != 0:
            return None
        if library.cuDeviceGet(ctypes.byref(dev), index) != 0:
            return None
        if library.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != 0:
            return None
        if library.cuCtxSetCurrent(ctx) != 0:
            return None
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        rc = library.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        library.cuCtxDetach(ctx)
        if rc != 0:
            return None
        return int(free.value)
    except (ctypes.ArgumentError, ValueError):
        return None


def check_vram(cells: int, device: str) -> tuple[int, int] | None:
    """Refuse ``cells`` if the estimate exceeds 90% of free device memory.

    Returns ``(estimate, free)`` when the free memory is known, else
    None.  Raises ValueError with a clear message when the run would
    not fit.
    """
    free = free_device_bytes(device)
    if free is None:
        return None
    estimate = vram_estimate_bytes(cells)
    if estimate > VRAM_FRACTION_LIMIT * free:
        raise ValueError(
            f"estimated VRAM {format_bytes(estimate)} for {cells:,} cells "
            f"exceeds {VRAM_FRACTION_LIMIT:.0%} of the {format_bytes(free)} "
            f"free on {device}; lower --quality or --cells"
        )
    return estimate, free


def format_bytes(count: float) -> str:
    """Human-readable bytes, one decimal, SI suffixes."""
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if count < 1000.0 or suffix == "TB":
            return f"{count:.1f} {suffix}" if suffix != "B" else f"{int(count)} B"
        count /= 1000.0
    return f"{count:.1f} TB"
