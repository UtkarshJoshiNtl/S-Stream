"""Import, normalize, orient, and place a surface inside the tunnel.

The transform is deliberately two-stage:

1. A free setup rotation (pitch/roll/yaw) stands an arbitrary model upright.
2. A world-Z yaw rotates it relative to the wind.

The model is then dropped onto the floor and clamped inside the tunnel.  All
geometry is returned in tunnel metres; no lattice units appear here.
"""

from __future__ import annotations

import sys

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

from .schema import MeshSpec, MeshTransform, TunnelSpec

SUPPORTED_SUFFIXES = {".obj", ".stl"}


def orient_to_wind(mesh: pv.PolyData) -> pv.PolyData:
    """Rotate the model so its longest extent lies along +X (the wind)
    and its shortest extent along +Z (up).

    The mapping is always a proper rotation (det = +1): when the axis
    permutation is odd, the new Y axis is negated to restore handedness,
    so the model is never mirrored.  Callers can opt out with
    ``MeshSpec.auto_orient = False``.
    """
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    extent = bounds[:, 1] - bounds[:, 0]
    order = np.argsort(extent)  # ascending: shortest first
    perm = np.array([order[2], order[1], order[0]], dtype=np.int64)
    points = np.asarray(mesh.points)[:, perm]
    if np.linalg.det(np.eye(3)[perm]) < 0.0:
        points[:, 1] *= -1.0
    out = mesh.copy()
    out.points = points
    return out


@dataclass
class ModelBuild:
    """The placed surface and the small facts callers need to report."""

    mesh: pv.PolyData
    scale: float
    warnings: list[str] = field(default_factory=list)

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.mesh.bounds, dtype=float).reshape(3, 2)


def _as_triangle_surface(data: pv.DataSet) -> pv.PolyData:
    """Reduce any dataset to triangles and drop non-surface cell types."""
    surface = data.triangulate().clean()
    faces = np.asarray(surface.faces)
    if faces.size == 0:
        return surface
    return pv.PolyData(np.asarray(surface.points), faces)


def _trim_outliers(mesh: pv.PolyData, fraction: float = 0.001) -> pv.PolyData:
    """Drop a handful of far-away vertices that would ruin the bounding box.

    Scanned models sometimes contain a tiny detached lidar speck or a stray
    vertex hundreds of millimetres from the body.  Clipping the extreme 0.1%
    on each axis keeps the model usable.  If the clip would remove more than
    2% of the surface, it is abandoned and the model is left untouched.
    """
    points = np.asarray(mesh.points)
    faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:]
    if len(points) < 1000 or len(faces) == 0:
        return mesh
    low = np.quantile(points, fraction, axis=0)
    high = np.quantile(points, 1.0 - fraction, axis=0)
    keep = np.all((points >= low) & (points <= high), axis=1)
    if keep.mean() < 0.98 or not keep.any():
        return mesh
    cells = np.all(keep[faces], axis=1)
    if not cells.any():
        return mesh
    kept_faces = faces[cells]
    used = np.unique(kept_faces)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    rebuilt = remap[kept_faces]
    faces_out = np.column_stack((np.full(len(rebuilt), 3), rebuilt)).ravel()
    return pv.PolyData(points[used], faces_out).clean()


def load_surface(path: str | Path) -> pv.PolyData:
    """Read an OBJ or STL file as one clean triangle surface."""
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported model format: {source.suffix or '(none)'}")
    if not source.is_file():
        raise FileNotFoundError(source)

    data = pv.read(str(source))
    if isinstance(data, pv.MultiBlock):
        data = data.combine(merge_points=True)
    if not isinstance(data, pv.PolyData):
        data = data.extract_surface()
    mesh = _trim_outliers(_as_triangle_surface(data))
    if mesh.n_points < 3 or mesh.n_cells < 1:
        raise ValueError("The file did not contain a triangle surface.")
    if not np.isfinite(np.asarray(mesh.points)).all():
        raise ValueError("The model contains invalid coordinates.")
    return mesh


def normalized(mesh: pv.PolyData, target_size: float) -> tuple[pv.PolyData, float]:
    """Scale the longest bounding-box side to ``target_size`` and centre it."""
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    extent = bounds[:, 1] - bounds[:, 0]
    longest = float(extent.max())
    if longest <= 1e-9:
        raise ValueError("The model has zero size.")
    scale = float(target_size) / longest
    centre = (bounds[:, 0] + bounds[:, 1]) * 0.5
    out = mesh.copy()
    out.points = (np.asarray(out.points) - centre) * scale
    return out, scale


def setup_rotated(mesh: pv.PolyData, transform: MeshTransform) -> pv.PolyData:
    """Apply the free 'stand it up' rotation about the model centre.

    Order is pitch (X), yaw (Y), roll (Z).  The mesh is already centred by
    :func:`normalized`, so the origin is the bounding-box centre.
    """
    out = mesh.copy()
    if transform.pitch:
        out = out.rotate_x(transform.pitch, point=(0.0, 0.0, 0.0))
    if transform.yaw:
        out = out.rotate_y(transform.yaw, point=(0.0, 0.0, 0.0))
    if transform.roll:
        out = out.rotate_z(transform.roll, point=(0.0, 0.0, 0.0))
    return out


def dropped_to_floor(mesh: pv.PolyData, floor_z: float, position_z: float = 0.0) -> pv.PolyData:
    z_min = float(np.asarray(mesh.bounds, dtype=float).reshape(3, 2)[2, 0])
    target_z = floor_z + position_z
    if abs(z_min - target_z) <= 1e-12:
        return mesh.copy()
    return mesh.translate((0.0, 0.0, target_z - z_min))


def yawed_about_bottom_center(mesh: pv.PolyData, angle: float) -> pv.PolyData:
    """Rotate about world Z through the bounding-box bottom centre."""
    if not angle:
        return mesh.copy()
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    centre = (bounds[:, 0] + bounds[:, 1]) * 0.5
    return mesh.rotate_z(angle, point=(centre[0], centre[1], bounds[2, 0]))


def clamp_inside(
    mesh: pv.PolyData,
    tunnel: TunnelSpec,
    position: tuple[float, float],
    margin: float = 0.05,
) -> tuple[pv.PolyData, list[str]]:
    """Move the model to ``position`` and keep its bbox inside the tunnel."""
    warnings: list[str] = []
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    half = (bounds[:, 1] - bounds[:, 0]) * 0.5

    targets = []
    for axis, desired, low, high in (
        (0, position[0], tunnel.x_min + margin, tunnel.x_max - margin),
        (1, position[1], tunnel.y_min + margin, tunnel.y_max - margin),
    ):
        lo, hi = low + half[axis], high - half[axis]
        if lo > hi:
            warnings.append(
                f"Model is wider than the tunnel on {'XY'[axis]}; using the centre."
            )
            targets.append((low + high) * 0.5)
        else:
            targets.append(float(np.clip(desired, lo, hi)))
    if bounds[2, 1] > tunnel.z_max + 1e-9:
        warnings.append("Model is taller than the tunnel; the top is outside.")

    current = bounds[:, 0] + half
    moved = mesh.translate((targets[0] - current[0], targets[1] - current[1], 0.0))
    return moved, warnings


def build_model(
    surface: pv.PolyData,
    spec: MeshSpec,
    tunnel: TunnelSpec,
) -> ModelBuild:
    """Run the full import-to-placement contract for one surface.

    When ``spec.auto_orient`` is set (the default), the model is first
    rotated so its longest extent lies along the wind (+X) and its
    shortest extent points up (+Z); the user's pitch/yaw/roll and the
    tunnel yaw then compose on top of that.
    """
    if spec.auto_orient:
        surface = orient_to_wind(surface)
    mesh, base_scale = normalized(surface, spec.target_size)
    # Apply scale multiplier from transform
    if spec.transform.scale != 1.0:
        mesh = mesh.scale([spec.transform.scale] * 3, inplace=False)
        base_scale *= spec.transform.scale
    mesh = setup_rotated(mesh, spec.transform)
    mesh = dropped_to_floor(mesh, tunnel.z_min, spec.transform.position_z)
    mesh = yawed_about_bottom_center(mesh, spec.transform.tunnel_yaw)
    mesh, warnings = clamp_inside(
        mesh, tunnel, (spec.transform.position_x, spec.transform.position_y), spec.margin
    )
    return ModelBuild(mesh=mesh, scale=base_scale, warnings=warnings)


def resource_root() -> Path:
    """Directory holding the bundled ``assets/`` folder (PyInstaller aware)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


DEFAULT_MODEL = resource_root() / "assets" / "3d-files" / "car-model.obj"
