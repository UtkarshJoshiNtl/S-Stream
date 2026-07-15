"""Free-surface flow (lattice volume-of-fluid style) — Phase D Experimental.

Tracks a liquid volume fraction ``phi`` in [0, 1] advected with the velocity
field. Gas cells (phi≈0) are treated with atmosphere pressure; liquid cells
use standard LBM. This is a minimal scaffolding, not a full free-surface
solver — validation is advection mass conservation of phi.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _advect_phi_nb(
    phi: np.ndarray,
    phi_new: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    height: int,
    width: int,
) -> None:
    for y in prange(height):
        for x in range(width):
            x_src = x - u[y, x]
            y_src = y - v[y, x]
            if x_src < 0.0:
                x_src = 0.0
            elif x_src > width - 1.001:
                x_src = width - 1.001
            if y_src < 0.0:
                y_src = 0.0
            elif y_src > height - 1.001:
                y_src = height - 1.001
            x0 = int(x_src)
            y0 = int(y_src)
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)
            fx = x_src - x0
            fy = y_src - y0
            v00 = phi[y0, x0]
            v10 = phi[y0, x1]
            v01 = phi[y1, x0]
            v11 = phi[y1, x1]
            phi_new[y, x] = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10
            phi_new[y, x] += (1 - fx) * fy * v01 + fx * fy * v11
            if phi_new[y, x] < 0.0:
                phi_new[y, x] = 0.0
            elif phi_new[y, x] > 1.0:
                phi_new[y, x] = 1.0


class FreeSurfaceTracker:
    """Volume fraction tracker coupled to an LBM2D-like engine."""

    def __init__(self, height: int, width: int) -> None:
        self.phi = np.zeros((height, width), dtype=np.float32)
        self._buf = np.zeros_like(self.phi)

    def fill_bottom(self, fill_height: int) -> None:
        self.phi[:] = 0.0
        self.phi[:fill_height, :] = 1.0

    def step(self, u: np.ndarray, v: np.ndarray) -> None:
        h, w = self.phi.shape
        _advect_phi_nb(self.phi, self._buf, u, v, h, w)
        self.phi, self._buf = self._buf, self.phi

    def mass(self) -> float:
        return float(np.sum(self.phi))
