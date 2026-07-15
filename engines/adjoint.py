"""Adjoint / shape-optimization hooks — Phase D Experimental.

Provides a thin differentiable-style interface for “optimize this shape”
workflows. Full autograd via Lettuce/PyTorch is optional; this module defines
the objective API and a finite-difference gradient estimator that works on
CPU BGK for small design vectors (e.g. cylinder radius, offset).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engines.lbm2d import LBM2D


@dataclass
class DesignVariable:
    name: str
    value: float
    lo: float
    hi: float


def cylinder_drag_objective(
    sim_factory: Callable[[], LBM2D],
    diameter: float,
    steps: int = 2000,
) -> float:
    """Run a short channel sim and return Cd (lower is better)."""
    from analysis.physics import drag_coefficient

    sim = sim_factory()
    sim.run(steps, physics_only=True)
    return float(drag_coefficient(sim, diameter=diameter))


def finite_difference_gradient(
    objective: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Central-difference gradient for small design vectors."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (objective(xp) - objective(xm)) / (2.0 * eps)
    return g
