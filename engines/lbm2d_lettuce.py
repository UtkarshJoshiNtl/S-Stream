"""Lettuce (PyTorch) GPU backend for S-Stream.

This engine provides an alternative GPU backend using Lettuce's PyTorch-based LBM.
Lettuce is MIT licensed: https://github.com/lettuce-project/lettuce

Legal Reference: Can incorporate directly with attribution (MIT license).
"""

from __future__ import annotations

import numpy as np

from engines.particle_tracer import ParticleTracer

try:
    import torch
    from lettuce import Lattice, UnitConversion
    from lettuce import D2Q9 as LettuceD2Q9
    from lettuce import BGKCollision as LettuceBGK
    from lettuce import StandardStreaming as LettuceStreaming
    from lettuce import BoundaryCondition as LettuceBC
    from lettuce import NoSlipBoundary as LettuceNoSlip
    from lettuce import EquilibriumBoundary as LettuceEquilibrium

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from engines.base import SimEngine
from engines.smoke_mixin import SmokeMixin


class LettuceBoundary:
    """Wrapper for Lettuce boundary conditions."""

    def __init__(self, bc_type: str, **kwargs):
        self.bc_type = bc_type
        self.kwargs = kwargs


class LBM2DLettuce(SimEngine, SmokeMixin):
    """D2Q9 Lattice Boltzmann fluid simulation using Lettuce (PyTorch GPU).

    This engine leverages Lettuce's PyTorch-based LBM for automatic CUDA
    kernel optimization and 3D GPU simulation capabilities.

    Legal: Lettuce (MIT license) - can incorporate directly with attribution.
    Reference: https://github.com/lettuce-project/lettuce
    """

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        viscosity: float = 0.02,
        device: str = "cuda:0",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and Lettuce are required for this backend. "
                "Install with: pip install sstream[gpu]"
            )

        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.15
        self.device = device

        # Initialize Lettuce lattice
        self._lattice = LettuceD2Q9(
            dtype=torch.float32,
            device=torch.device(device),
        )

        # Initialize fields
        self._rho = torch.ones(
            (height, width), dtype=torch.float32, device=torch.device(device)
        )
        self._u = torch.zeros(
            (2, height, width), dtype=torch.float32, device=torch.device(device)
        )
        self._f = self._lattice.equilibrium(
            self._rho.unsqueeze(0),
            self._u,
        )

        # Obstacles (CPU for compatibility, will be moved to GPU as needed)
        self.obstacles = np.zeros((height, width), dtype=np.bool_)

        # Smoke (CPU for compatibility with SmokeMixin)
        self.smoke = np.zeros((height, width), dtype=np.float32)
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, float]] = []

        # Coordinate arrays for advection
        self._x_coords = np.arange(width, dtype=np.float32)
        self._y_coords = np.arange(height, dtype=np.float32)
        self.xp = np

        self._particle_tracer = ParticleTracer(width, height, trail_length=20)

        # Initialize collision operator
        self._collision = LettuceBGK(
            lattice=self._lattice,
            tau=self._lattice.omega_from_viscosity(viscosity),
        )

        # Initialize streaming
        self._streaming = LettuceStreaming(self._lattice)

        # Boundary conditions
        self._boundary_conditions = []

        self.initialize(rho=1.0, u=0.1, v=0.0)

    # --- SimEngine interface ---

    @property
    def ndim(self) -> int:
        return 2

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return (self.height, self.width)

    @property
    def omega(self) -> float:
        return self._lattice.omega_from_viscosity(self.viscosity)

    def initialize(
        self, rho: float = 1.0, u: float = 0.1, v: float = 0.0, w: float = 0.0
    ) -> None:
        self._rho[:] = rho
        self._u[0, :, :] = u
        self._u[1, :, :] = v
        self._f = self._lattice.equilibrium(
            self._rho.unsqueeze(0),
            self._u,
        )
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

    def step(self) -> None:
        # Apply collision
        self._f = self._collision(self._f)

        # Apply streaming
        self._f = self._streaming(self._f)

        # Apply boundary conditions
        for bc in self._boundary_conditions:
            self._f = bc(self._f)

        # Compute macroscopic quantities
        self._rho = self._lattice.rho(self._f)
        self._u = self._lattice.u(self._f)

        # Apply smoke operations (CPU-based for compatibility)
        self.apply_emitters()
        self.advect_smoke()
        self.diffuse_smoke()
        self.smoke[self.obstacles] = 0.0
        self.decay_smoke()

        vel = self.get_velocity()
        self._particle_tracer.step(vel)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def get_density(self) -> np.ndarray:
        return self._rho.cpu().numpy()

    def get_velocity(self) -> np.ndarray:
        # Lettuce returns (2, H, W), we need (H, W, 2)
        u_np = self._u.cpu().numpy()
        return np.transpose(u_np, (1, 2, 0))

    def get_smoke(self) -> np.ndarray:
        return self.smoke.copy()

    def get_obstacles(self) -> np.ndarray:
        return self.obstacles.copy()

    def get_obstacles_mut(self) -> np.ndarray:
        return self.obstacles

    def get_f(self) -> np.ndarray:
        # Lettuce returns (9, H, W) in torch tensor
        return self._f.cpu().numpy()

    def get_pressure(self) -> np.ndarray:
        return self._rho.cpu().numpy() - 1.0

    def get_field_names(self) -> list[str]:
        return ["smoke", "speed", "vorticity", "pressure", "density"]

    def get_field(self, name: str) -> np.ndarray:
        vel = self.get_velocity()
        u, v = vel[:, :, 0], vel[:, :, 1]
        if name == "smoke":
            field = self.smoke.copy()
            mx = max(float(np.percentile(field, 98)), 0.001)
            return np.clip(field / mx, 0, 1).astype(np.float32)
        if name == "speed":
            speed = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)
            mx = max(self.u_inflow * 1.5, float(np.percentile(speed, 98)), 0.001)
            return np.clip(speed / mx, 0, 1).astype(np.float32)
        if name == "vorticity":
            dvdx = np.zeros_like(u, dtype=np.float32)
            dudy = np.zeros_like(u, dtype=np.float32)
            dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
            dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
            vort = dvdx - dudy
            mx = max(float(np.percentile(np.abs(vort), 98)), 0.001)
            return np.clip(vort / mx * 0.5 + 0.5, 0, 1).astype(np.float32)
        if name == "pressure":
            p = (self.get_density() - 1.0).astype(np.float32)
            mx = max(float(np.percentile(np.abs(p), 98)), 0.001)
            return np.clip(p / mx * 0.5 + 0.5, 0, 1).astype(np.float32)
        if name == "density":
            rho = self.get_density()
            lo, hi = float(np.min(rho)), float(np.max(rho))
            if hi - lo < 0.001:
                return np.full_like(rho, 0.5, dtype=np.float32)
            return np.clip((rho - lo) / (hi - lo), 0, 1).astype(np.float32)
        raise ValueError(f"Unknown field: {name!r}. Available: {self.get_field_names()}")

    def get_emitter_count(self) -> int:
        return len(self.emitters)

    def get_particle_tracer(self) -> ParticleTracer:
        return self._particle_tracer

    def add_obstacle(self, x: int, y: int, radius: int = 5) -> None:
        y_grid, x_grid = np.ogrid[: self.height, : self.width]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius**2
        self.obstacles[mask] = True

    def clear_obstacles(self) -> None:
        self.obstacles[:] = False

    def add_emitter(self, x: int, y: int, strength: float = 0.05) -> None:
        self.emitters.append((x, y, strength))

    def clear_emitters(self) -> None:
        self.emitters.clear()
