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


def _fill_streamwise(solid: np.ndarray) -> np.ndarray:
    """Fill gaps between the first and last solid cell in each wind column.

    An open scan or a car shell often has cracks that the rasterizer sees as
    tunnels.  A steady streamwise flow cannot pass through a real body, so
    closing each (y, z) column makes the body behave as a solid.  Applying
    this only to non-watertight surfaces keeps clean models untouched.
    """
    solid = np.asarray(solid, dtype=bool)
    occupied = solid.any(axis=0)
    if not occupied.any():
        return solid.copy()
    first = np.argmax(solid, axis=0)
    last = solid.shape[0] - 1 - np.argmax(solid[::-1], axis=0)
    x = np.arange(solid.shape[0])[:, None, None]
    return solid | ((x >= first[None]) & (x <= last[None]) & occupied[None])


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
                "Mesh is not watertight; gaps along the wind direction were "
                "filled so the body acts as a solid."
            )

    dimensions = tunnel.grid_shape
    solid = _rasterize(working, tunnel.grid_origin, tunnel.grid_spacing, dimensions)
    if not watertight:
        solid = _fill_streamwise(solid)
        warnings.append("Streamwise gaps filled to prevent wind tunnel artifacts")
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
