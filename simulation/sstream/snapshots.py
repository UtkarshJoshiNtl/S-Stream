"""Streaming field snapshots for a solve.

Snapshots never sit in RAM as a collection: each one is converted to
float16 and written to its own ``velocity_step_NNNNNNN.npz`` the moment
its step finishes, so peak memory is one field plus one file.  ``meta``
is written once at the end and points at ``solid.npy`` for the mask.

Snapshot steps are spaced evenly over the last
:data:`FLOW_THROUGH_COUNT` flow-throughs (a flow-through is the time
the freestream needs to cross the tunnel, ``length / speed / dt``),
which skips the startup transient by construction.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .lattice import TAU_MIN
from .quality import format_bytes

#: How many flow-throughs the snapshot window covers.
FLOW_THROUGH_COUNT = 2

#: Per-file safety margin on the byte estimate.
FILE_OVERHEAD = 8192
#: Estimate factor for zip headers and alignment.
ESTIMATE_FACTOR = 1.05
#: Refuse when free space is below this multiple of the estimate.
DISK_MARGIN = 1.2

VELOCITY_DTYPE = np.dtype("<f2")  # float16, little-endian, portable


def get_velocity_dtype(dtype_str: str = "float16") -> np.dtype:
    """Get numpy dtype from string."""
    if dtype_str == "float16":
        return np.dtype("<f2")
    elif dtype_str == "float32":
        return np.dtype("<f4")
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def steps_per_flow_through(tunnel_length: float, speed: float, dt: float) -> float:
    """Lattice steps for the freestream to cross the tunnel once."""
    if speed <= 0.0 or dt <= 0.0:
        raise ValueError("speed and dt must be positive for a flow-through")
    return tunnel_length / speed / dt


def snapshot_steps(total_steps: int, per_flow_through: float, count: int) -> list[int]:
    """``count`` steps, evenly spaced over the last flow-through window.

    The window is the last :data:`FLOW_THROUGH_COUNT` flow-throughs, but
    never earlier than the second half of the run: the startup transient is
    always skipped (the same convention the force average uses), even when
    a flow-through is longer than the preset's step count.
    """
    if count <= 0:
        return []
    window = int(FLOW_THROUGH_COUNT * per_flow_through)
    start = max(total_steps // 2, total_steps - window)
    start = max(1, start)
    raw = np.unique(np.linspace(start, total_steps, num=count).round().astype(np.int64))
    return [int(step) for step in raw]


def snapshot_bytes(cells: int, count: int, dtype_str: str = "float16") -> int:
    """Estimated bytes on disk for ``count`` velocity fields."""
    dtype = get_velocity_dtype(dtype_str)
    per_file = int(cells * 3 * dtype.itemsize * ESTIMATE_FACTOR) + FILE_OVERHEAD
    return per_file * count


def check_disk(directory: Path, needed: int) -> int:
    """Refuse with a clear message when free disk is below ``DISK_MARGIN``×."""
    free = shutil.disk_usage(directory).free
    if free < DISK_MARGIN * needed:
        raise ValueError(
            f"refusing snapshots: free disk {format_bytes(free)} in {directory} "
            f"is below {DISK_MARGIN}× the {format_bytes(needed)} needed"
        )
    return free


def write_velocity(directory: Path, step: int, velocity: np.ndarray, dtype_str: str = "float16", density: np.ndarray | None = None) -> int:
    """Write one velocity snapshot; returns bytes on disk.

    If density is provided, it is saved alongside velocity in the same NPZ.
    """
    path = directory / f"velocity_step_{step:07d}.npz"
    dtype = get_velocity_dtype(dtype_str)
    data = {
        "velocity": np.ascontiguousarray(velocity, dtype=dtype),
        "step": np.int64(step),
    }
    if density is not None:
        data["density"] = np.ascontiguousarray(density, dtype=dtype)
    np.savez(path, **data)
    return path.stat().st_size


def write_solid_mask(directory: Path, solid: np.ndarray) -> Path:
    """Write the boolean solid mask as uint8; returns the path."""
    path = directory / "solid.npy"
    np.save(path, np.ascontiguousarray(solid, dtype=np.uint8))
    return path


def write_meta(
    directory: Path,
    tunnel,
    placement: dict,
    snapshot_steps: list[int],
    units,
    smagorinsky: float,
) -> Path:
    """Write ``meta.json`` once: solid mask, spacing, tunnel, placement."""
    meta = {
        "model": placement.get("model"),
        "placement": placement,
        "tunnel": {
            "length": tunnel.length,
            "width": tunnel.width,
            "height": tunnel.height,
            "floor_z": tunnel.floor_z,
            "cells_streamwise": tunnel.cells_streamwise,
            "grid_shape": list(tunnel.grid_shape),
            "spacing": list(tunnel.grid_spacing),
            "bounds": list(tunnel.bounds),
        },
        "solid_mask": {
            "file": "solid.npy",
            "dtype": "uint8",
            "shape": list(tunnel.grid_shape),
        },
        "snapshots": {
            "files": [f"velocity_step_{step:07d}.npz" for step in snapshot_steps],
            "steps": [int(step) for step in snapshot_steps],
            "velocity_dtype": str(VELOCITY_DTYPE),
            "flow_throughs": FLOW_THROUGH_COUNT,
        },
        "lattice": {
            "u_lattice": units.u_lattice,
            "tau": units.tau,
            "tau_min": TAU_MIN,
            "dt": units.dt,
            "dx": units.dx,
            "smagorinsky": smagorinsky,
        },
    }
    path = directory / "meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def placement_dict(path, spec, build) -> dict:
    """Everything needed to rebuild the placed model in a renderer."""
    placed = asdict(spec)
    placed["model"] = str(path)
    placed["placed_scale"] = build.scale
    bounds = np.asarray(build.mesh.bounds, dtype=float).reshape(3, 2)
    placed["placed_bounds"] = bounds.tolist()
    return placed
