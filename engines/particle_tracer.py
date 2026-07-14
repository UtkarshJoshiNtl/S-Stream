"""Lagrangian particle tracer for passive tracer advection.

Particles are advected by the flow field using second-order Heun's method
(RK2), which provides significantly better accuracy than simple Euler
integration for capturing streamline structure.

Particles are stored as a contiguous (N, 2) float32 array for 2D or
(N, 3) float32 array for 3D, with positions in continuous grid coordinates.
"""
from __future__ import annotations

import numpy as np


class ParticleTracer:
    """Lagrangian particle tracer with RK2 advection.

    Attributes:
        positions: (N, 2) or (N, 3) array of continuous grid coordinates.
        trail_length: Number of past positions kept per particle for rendering.
        trails: (trail_length, N, 2) circular buffer of past positions.
        max_particles: Hard cap on particle count.
    """

    def __init__(
        self,
        width: int,
        height: int,
        depth: int | None = None,
        trail_length: int = 20,
        max_particles: int = 50000,
    ) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.trail_length = max(1, trail_length)
        self.max_particles = max_particles
        self.is_3d = depth is not None

        self.positions: np.ndarray = np.empty(
            (0, 3 if self.is_3d else 2), dtype=np.float32
        )
        self._trail_buf: np.ndarray | None = None
        if self.trail_length > 1:
            self._trail_buf = np.empty(
                (self.trail_length, 0, 3 if self.is_3d else 2), dtype=np.float32
            )
        self._step_count: int = 0

    @property
    def count(self) -> int:
        return len(self.positions)

    def get_positions(self) -> np.ndarray:
        """Return a copy of current particle positions."""
        return self.positions.copy()

    def get_trails(self) -> np.ndarray | None:
        """Return a copy of the trail buffer, or None if trails disabled."""
        if self._trail_buf is None or self._trail_buf.shape[1] == 0:
            return None
        return self._trail_buf.copy()

    def add_particles(self, x: float, y: float, count: int = 10, z: float = 0.0) -> int:
        """Add particles near (x, y) with small random jitter.

        Returns the number of particles actually added (may be fewer if at cap).
        """
        if count <= 0:
            return 0
        available = self.max_particles - self.count
        if available <= 0:
            return 0
        count = min(count, available)

        rng = np.random.default_rng()
        jitter_x = rng.uniform(-0.4, 0.4, count).astype(np.float32)
        jitter_y = rng.uniform(-0.4, 0.4, count).astype(np.float32)

        if self.is_3d:
            jitter_z = rng.uniform(-0.4, 0.4, count).astype(np.float32)
            new_pos = np.column_stack([
                np.full(count, x, dtype=np.float32) + jitter_x,
                np.full(count, y, dtype=np.float32) + jitter_y,
                np.full(count, z, dtype=np.float32) + jitter_z,
            ])
        else:
            new_pos = np.column_stack([
                np.full(count, x, dtype=np.float32) + jitter_x,
                np.full(count, y, dtype=np.float32) + jitter_y,
            ])

        self.positions = np.vstack([self.positions, new_pos]) if self.count > 0 else new_pos
        self._sync_trail_buffer()
        return count

    def add_particles_line(
        self, x1: float, y1: float, x2: float, y2: float, count: int = 20
    ) -> int:
        """Add particles uniformly along a line segment.

        Returns the number of particles actually added.
        """
        if count <= 0:
            return 0
        available = self.max_particles - self.count
        if available <= 0:
            return 0
        count = min(count, available)

        ts = np.linspace(0, 1, count, dtype=np.float32)
        xs = x1 + (x2 - x1) * ts
        ys = y1 + (y2 - y1) * ts

        if self.is_3d:
            new_pos = np.column_stack([xs, ys, np.zeros(count, dtype=np.float32)])
        else:
            new_pos = np.column_stack([xs, ys])

        self.positions = np.vstack([self.positions, new_pos]) if self.count > 0 else new_pos
        self._sync_trail_buffer()
        return count

    def add_particles_random(self, count: int = 100) -> int:
        """Add particles at random positions within the domain.

        Returns the number of particles actually added.
        """
        if count <= 0:
            return 0
        available = self.max_particles - self.count
        if available <= 0:
            return 0
        count = min(count, available)

        rng = np.random.default_rng()
        if self.is_3d:
            new_pos = np.column_stack([
                rng.uniform(0, self.width - 1, count).astype(np.float32),
                rng.uniform(0, self.height - 1, count).astype(np.float32),
                rng.uniform(0, self.depth - 1, count).astype(np.float32),
            ])
        else:
            new_pos = np.column_stack([
                rng.uniform(0, self.width - 1, count).astype(np.float32),
                rng.uniform(0, self.height - 1, count).astype(np.float32),
            ])

        self.positions = np.vstack([self.positions, new_pos]) if self.count > 0 else new_pos
        self._sync_trail_buffer()
        return count

    def step(self, vel_field: np.ndarray) -> int:
        """Advect all particles by one timestep using Heun's method (RK2).

        Args:
            vel_field: Velocity field, shape (H, W, 2) for 2D or (D, H, W, 3) for 3D.
                       Velocities are in grid cells per timestep.

        Returns the number of particles removed (out-of-bounds or obstacle).
        """
        if self.count == 0:
            return 0

        self._step_count += 1
        n = self.count

        # --- RK2 (Heun's method) ---
        # Stage 1: Euler predictor
        v1 = self._interpolate_velocity(vel_field, self.positions)
        pred = self.positions + v1

        # Stage 2: corrector using predicted positions
        v2 = self._interpolate_velocity(vel_field, pred)
        self.positions = self.positions + 0.5 * (v1 + v2)

        # Update trails
        if self._trail_buf is not None and n > 0:
            idx = (self._step_count - 1) % self.trail_length
            self._trail_buf[idx] = self.positions

        # Remove out-of-bounds particles
        removed = self._remove_oob()
        return removed

    def clear(self) -> None:
        """Remove all particles."""
        ndim = 3 if self.is_3d else 2
        self.positions = np.empty((0, ndim), dtype=np.float32)
        if self._trail_buf is not None:
            self._trail_buf = np.empty(
                (self.trail_length, 0, ndim), dtype=np.float32
            )
        self._step_count = 0

    def set_trail_length(self, length: int) -> None:
        """Change trail length, preserving existing trail data if possible."""
        length = max(1, length)
        if length == self.trail_length:
            return
        ndim = 3 if self.is_3d else 2
        old_buf = self._trail_buf
        self.trail_length = length
        if length > 1:
            self._trail_buf = np.empty((length, self.count, ndim), dtype=np.float32)
            if old_buf is not None and old_buf.shape[1] > 0:
                copy_len = min(old_buf.shape[0], length)
                self._trail_buf[:copy_len] = old_buf[:copy_len]
        else:
            self._trail_buf = None

    def _interpolate_velocity(
        self, vel_field: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Bilinear (2D) or trilinear (3D) interpolation of velocity at particle positions."""
        if self.is_3d:
            return self._interp_3d(vel_field, positions)
        return self._interp_2d(vel_field, positions)

    def _interp_2d(self, vel: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Bilinear interpolation for 2D velocity field (H, W, 2)."""
        h, w = vel.shape[:2]
        x = np.clip(pos[:, 0], 0.0, w - 1.001)
        y = np.clip(pos[:, 1], 0.0, h - 1.001)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)

        v00 = vel[y0, x0]  # (N, 2)
        v10 = vel[y0, x1]
        v01 = vel[y1, x0]
        v11 = vel[y1, x1]

        return (
            v00 * (1 - fx[:, None]) * (1 - fy[:, None])
            + v10 * fx[:, None] * (1 - fy[:, None])
            + v01 * (1 - fx[:, None]) * fy[:, None]
            + v11 * fx[:, None] * fy[:, None]
        )

    def _interp_3d(self, vel: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation for 3D velocity field (D, H, W, 3)."""
        d, h, w = vel.shape[:3]
        x = np.clip(pos[:, 0], 0.0, w - 1.001)
        y = np.clip(pos[:, 1], 0.0, h - 1.001)
        z = np.clip(pos[:, 2], 0.0, d - 1.001)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        z0 = np.floor(z).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        z1 = np.minimum(z0 + 1, d - 1)

        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        fz = (z - z0).astype(np.float32)

        v000 = vel[z0, y0, x0]
        v100 = vel[z0, y0, x1]
        v010 = vel[z0, y1, x0]
        v110 = vel[z0, y1, x1]
        v001 = vel[z1, y0, x0]
        v101 = vel[z1, y0, x1]
        v011 = vel[z1, y1, x0]
        v111 = vel[z1, y1, x1]

        fx_ = fx[:, None]
        fy_ = fy[:, None]
        fz_ = fz[:, None]

        return (
            v000 * (1 - fx_) * (1 - fy_) * (1 - fz_)
            + v100 * fx_ * (1 - fy_) * (1 - fz_)
            + v010 * (1 - fx_) * fy_ * (1 - fz_)
            + v110 * fx_ * fy_ * (1 - fz_)
            + v001 * (1 - fx_) * (1 - fy_) * fz_
            + v101 * fx_ * (1 - fy_) * fz_
            + v011 * (1 - fx_) * fy_ * fz_
            + v111 * fx_ * fy_ * fz_
        )

    def _remove_oob(self) -> int:
        """Remove particles outside domain bounds. Returns count removed."""
        if self.count == 0:
            return 0
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        margin = 0.5
        if self.is_3d:
            z = self.positions[:, 2]
            mask = (
                (x >= -margin) & (x < self.width - 1 + margin)
                & (y >= -margin) & (y < self.height - 1 + margin)
                & (z >= -margin) & (z < self.depth - 1 + margin)
            )
        else:
            mask = (
                (x >= -margin) & (x < self.width - 1 + margin)
                & (y >= -margin) & (y < self.height - 1 + margin)
            )

        removed = self.count - int(mask.sum())
        if removed > 0:
            self.positions = self.positions[mask]
            if self._trail_buf is not None:
                self._trail_buf = self._trail_buf[:, mask]
        return removed

    def _sync_trail_buffer(self) -> None:
        """Resize trail buffer to match current particle count."""
        if self._trail_buf is None:
            return
        ndim = 3 if self.is_3d else 2
        n = self.count
        if self._trail_buf.shape[1] != n:
            new_buf = np.empty((self.trail_length, n, ndim), dtype=np.float32)
            copy_len = min(self._trail_buf.shape[0], self.trail_length)
            if self._trail_buf.shape[1] > 0:
                new_buf[:copy_len, :min(n, self._trail_buf.shape[1])] = (
                    self._trail_buf[:copy_len, :min(n, self._trail_buf.shape[1])]
                )
            self._trail_buf = new_buf
