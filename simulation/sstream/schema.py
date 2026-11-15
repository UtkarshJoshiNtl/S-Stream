"""Plain data for the wind tunnel.

Coordinate convention: X is the wind direction, Z is up, and the floor is
``TunnelSpec.floor_z``.  Nothing in this module imports Qt, Warp, or PyVista,
so the specs can be tested and serialized without a display or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite

ANGLE_SNAP = 15.0
ANGLE_FINE = 1.0
POSITION_SNAP = 0.1
POSITION_FINE = 0.01

ORIENT_MODES = ("auto-vehicle", "upright", "none")
UP_AXES = ("+Y", "+Z")


def snap_value(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step``.

    A non-positive step means "no snapping".  Half-way values follow
    Python's ``round`` (nearest even multiple).
    """
    if not isfinite(value) or step <= 0.0:
        return float(value)
    return round(value / step) * step


@dataclass(frozen=True)
class TunnelSpec:
    """A rectangular, open-circuit wind tunnel in metres.

    ``cells_streamwise`` is the number of lattice cells along X, not a cubic
    resolution.  Cells are cubes, so the cross-section is rounded to whole
    cells and may be a few millimetres smaller than the nominal tunnel.

    Dimensions can be set directly or computed from a model length L using
    ``from_model_length`` for a tunnel that is 4L long, 1.5L wide, and 1.0L tall.
    """

    length: float = 6.0
    width: float = 2.4
    height: float = 1.8
    floor_z: float = 0.0
    cells_streamwise: int = 256

    @classmethod
    def from_model_length(cls, model_length: float, cells_streamwise: int = 256) -> "TunnelSpec":
        """Create a tunnel sized relative to model length L: 4L long, 1.5L wide, 1.0L tall."""
        return cls(
            length=4.0 * model_length,
            width=1.5 * model_length,
            height=1.0 * model_length,
            floor_z=0.0,
            cells_streamwise=cells_streamwise,
        )

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Tunnel dimensions must be positive")
        if self.cells_streamwise < 8:
            raise ValueError("cells_streamwise must be at least 8")

    @property
    def x_min(self) -> float:
        return 0.0

    @property
    def x_max(self) -> float:
        return self.length

    @property
    def y_min(self) -> float:
        return -self.width * 0.5

    @property
    def y_max(self) -> float:
        return self.width * 0.5

    @property
    def z_min(self) -> float:
        return self.floor_z

    @property
    def z_max(self) -> float:
        return self.floor_z + self.height

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    @property
    def cell_size(self) -> float:
        return self.length / self.cells_streamwise

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        size = self.cell_size
        return (
            self.cells_streamwise,
            max(1, round(self.width / size)),
            max(1, round(self.height / size)),
        )

    @property
    def grid_spacing(self) -> tuple[float, float, float]:
        size = self.cell_size
        return (size, size, size)

    @property
    def grid_origin(self) -> tuple[float, float, float]:
        """Centre of the first cell, so every point is a cell centre."""
        size = self.cell_size
        return (self.x_min + size * 0.5, self.y_min + size * 0.5, self.z_min + size * 0.5)

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.grid_shape
        return nx * ny * nz

    def with_resolution(self, cells_streamwise: int) -> TunnelSpec:
        return replace(self, cells_streamwise=int(cells_streamwise))


@dataclass(frozen=True)
class FluidSpec:
    """Air properties and the reference values used for reporting."""

    speed: float = 30.0
    density: float = 1.225
    kinematic_viscosity: float = 1.5e-5
    reference_length: float = 1.2

    def __post_init__(self) -> None:
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive")

    @property
    def reynolds(self) -> float:
        return self.speed * self.reference_length / self.kinematic_viscosity

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.density * self.speed * self.speed


@dataclass(frozen=True)
class MeshTransform:
    """Two-stage placement: stand the model up, then rotate it in the tunnel.

    ``pitch``/``roll``/``yaw`` are the free setup rotation about X/Y/Z and are
    applied before the model is dropped onto the floor.  ``tunnel_yaw`` is
    always about world Z, so the wind direction never becomes ambiguous.
    ``scale`` is a multiplier applied after normalization.
    """

    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    tunnel_yaw: float = 0.0
    position_x: float = 3.0
    position_y: float = 0.0
    position_z: float = 0.0
    scale: float = 1.0

    def snapped(
        self,
        angle_step: float = ANGLE_SNAP,
        position_step: float = POSITION_SNAP,
    ) -> MeshTransform:
        return replace(
            self,
            pitch=snap_value(self.pitch, angle_step),
            roll=snap_value(self.roll, angle_step),
            yaw=snap_value(self.yaw, angle_step),
            tunnel_yaw=snap_value(self.tunnel_yaw, angle_step),
            position_x=snap_value(self.position_x, position_step),
            position_y=snap_value(self.position_y, position_step),
            position_z=snap_value(self.position_z, position_step),
        )


@dataclass(frozen=True)
class MeshSpec:
    """An imported surface plus the placement the user chose for it.

    ``orient`` picks the alignment mode: ``auto-vehicle`` maps the longest
    extent to the wind and the shortest to up, ``upright`` keeps the model's
    up axis (``up``) and puts the wind along the longer horizontal extent,
    ``none`` uses the file as imported.  ``fraction`` scales the longest
    extent to that fraction of the tunnel length.
    """

    path: str | None = None
    margin: float = 0.05
    orient: str = "auto-vehicle"
    up: str = "+Y"
    flip: bool = False
    pitch: float = 0.0
    yaw: float = 0.0
    fraction: float = 0.2
    transform: MeshTransform = field(default_factory=MeshTransform)

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError("margin cannot be negative")
        if self.orient not in ORIENT_MODES:
            raise ValueError(f"orient must be one of {ORIENT_MODES}")
        if self.up not in UP_AXES:
            raise ValueError(f"up must be one of {UP_AXES}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
