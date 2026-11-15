"""Turn a placed surface into the boolean mask the solver will see.

VTK's polydata-to-image-stencil path is used instead of writing our own
triangle rasterizer.  It is the same mature code path ParaView uses, and it
keeps this project small.  Open meshes are detected, repaired when possible,
and otherwise reported instead of silently producing a bogus solid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkFeatureEdges
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter
from vtkmodules.vtkImagingStencil import vtkImageStencil, vtkPolyDataToImageStencil
from vtkmodules.util.numpy_support import vtk_to_numpy

from .schema import TunnelSpec

VTK_UNSIGNED_CHAR = 3


@dataclass
class VoxelResult:
    """A boolean solid mask and the grid it belongs to."""

    solid: np.ndarray
    boundary: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    dimensions: tuple[int, int, int]
    watertight: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def solid_count(self) -> int:
        return int(self.solid.sum())

    @property
    def empty(self) -> bool:
        return self.solid_count == 0

    def to_image(self) -> pv.ImageData:
        """Package the mask as an ImageData whose points are cell centres."""
        image = pv.ImageData(
            dimensions=self.dimensions,
            spacing=self.spacing,
            origin=self.origin,
        )
        image.point_data["solid"] = self.solid.astype(np.uint8).ravel(order="F")
        image.point_data["boundary"] = self.boundary.astype(np.uint8).ravel(order="F")
        return image


def _boundary_edges(mesh: pv.PolyData) -> int:
    edges = vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return int(edges.GetOutput().GetNumberOfCells())


def _fill_holes(mesh: pv.PolyData, hole_size: float) -> pv.PolyData:
    filler = vtkFillHolesFilter()
    filler.SetInputData(mesh)
    filler.SetHoleSize(max(hole_size, 0.0))
    filler.Update()
    return pv.wrap(filler.GetOutput()).triangulate().clean()


def _rasterize(
    mesh: pv.PolyData,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    dimensions: tuple[int, int, int],
) -> np.ndarray:
    image = vtkImageData()
    image.SetOrigin(*origin)
    image.SetSpacing(*spacing)
    image.SetDimensions(*dimensions)
    image.AllocateScalars(VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(255)

    stencil = vtkPolyDataToImageStencil()
    stencil.SetInputData(mesh)
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(*spacing)
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()

    cutter = vtkImageStencil()
    cutter.SetInputData(image)
    cutter.SetStencilConnection(stencil.GetOutputPort())
    cutter.SetBackgroundValue(0)
    cutter.ReverseStencilOff()
    cutter.Update()

    values = vtk_to_numpy(cutter.GetOutput().GetPointData().GetScalars())
    # VTK stores x fastest; reshape as (z, y, x) then transpose to (x, y, z).
    return values.reshape(dimensions[::-1]).transpose(2, 1, 0) > 0


def _erode(mask: np.ndarray) -> np.ndarray:
    """Six-neighbour erosion; boundary cells are treated as outside."""
    solid = np.asarray(mask, dtype=bool)
    out = solid.copy()
    for axis in range(3):
        lead = [slice(None)] * 3
        trail = [slice(None)] * 3
        lead[axis] = slice(1, None)
        trail[axis] = slice(None, -1)
        out[tuple(lead)] &= solid[tuple(trail)]
        out[tuple(trail)] &= solid[tuple(lead)]
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = 0
        last[axis] = -1
        out[tuple(first)] = False
        out[tuple(last)] = False
    return out


def _flood_outside(solid: np.ndarray) -> np.ndarray:
    """Cells reachable from the domain boundary without crossing solid.

    Iterative 6-neighbour dilation from the boundary fluid cells.  Used on
    non-watertight surfaces: the rasterized shell already blocks cell-scale
    tunnels, and anything the flood cannot reach is treated as the body's
    interior instead of being silently kept as fluid.  Leaks larger than a
    cell still let fluid through, which is the honest outcome for a shell
    that is open at grid scale.
    """
    solid = np.asarray(solid, dtype=bool)
    outside = np.zeros_like(solid)
    outside[0] = ~solid[0]
    outside[-1] = ~solid[-1]
    outside[:, 0] = ~solid[:, 0]
    outside[:, -1] = ~solid[:, -1]
    outside[:, :, 0] = ~solid[:, :, 0]
    outside[:, :, -1] = ~solid[:, :, -1]
    while True:
        grown = outside.copy()
        for axis in range(3):
            hi, lo = [slice(None)] * 3, [slice(None)] * 3
            hi[axis] = slice(1, None)
            lo[axis] = slice(None, -1)
            grown[tuple(hi)] |= outside[tuple(lo)]
            grown[tuple(lo)] |= outside[tuple(hi)]
        grown &= ~solid
        if np.array_equal(grown, outside):
            return outside
        outside = grown


def voxelize(mesh: pv.PolyData, tunnel: TunnelSpec) -> VoxelResult:
    """Rasterize ``mesh`` onto ``tunnel``'s grid.

    The model must already have been placed by :func:`sstream.mesh.build_model`.
    """
    warnings: list[str] = []
    working = mesh
    open_edges = _boundary_edges(working)
    watertight = open_edges == 0
    if not watertight:
        extent = np.asarray(working.bounds, dtype=float).reshape(3, 2)
        span = float((extent[:, 1] - extent[:, 0]).max())
        working = _fill_holes(working, span * 0.15)
        watertight = _boundary_edges(working) == 0
        if watertight:
            warnings.append("Mesh had open edges; small holes were filled.")
        else:
            warnings.append(
                "Mesh is not watertight after closing small holes; "
                "the body is treated as solid."
            )

    dimensions = tunnel.grid_shape
    solid = _rasterize(working, tunnel.grid_origin, tunnel.grid_spacing, dimensions)
    if not watertight:
        outside = _flood_outside(solid)
        interior = ~solid & ~outside
        if interior.any():
            solid = solid | interior
            warnings.append(
                "Mesh is not watertight; cells unreachable from outside were "
                "filled so the body acts as solid."
            )
    if not solid.any():
        warnings.append("The model did not cover any voxels. Check its scale.")
    boundary = solid & ~_erode(solid)
    return VoxelResult(
        solid=solid,
        boundary=boundary,
        origin=tunnel.grid_origin,
        spacing=tunnel.grid_spacing,
        dimensions=dimensions,
        watertight=watertight,
        warnings=warnings,
    )
