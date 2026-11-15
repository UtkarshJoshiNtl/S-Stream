"""A passive dye field for showing where the wind has been.

The dye is a scalar carried by a semi-Lagrangian step: trace each cell backwards
along the velocity, sample the previous dye field there, and blend.  It is
unconditionally stable at the timesteps used here, and it costs one small
kernel per solver step.

Stripes are emitted on the inlet plane.  A stripe that disappears into a wake
is the clearest possible "the flow is disturbed here" cue, and it is honest:
the dye does not push the fluid back.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.func
def _trilinear(
    field: wp.array3d(dtype=wp.float32),
    point: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    x = wp.clamp(point[0], 0.0, float(nx - 1))
    y = wp.clamp(point[1], 0.0, float(ny - 1))
    z = wp.clamp(point[2], 0.0, float(nz - 1))

    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    z0 = int(wp.floor(z))
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)
    tx = x - float(x0)
    ty = y - float(y0)
    tz = z - float(z0)

    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@wp.kernel
def _advect_kernel(
    dye_in: wp.array3d(dtype=wp.float32),
    dye_out: wp.array3d(dtype=wp.float32),
    velocity: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.uint8),
    stripe: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    decay: float,
):
    x, y, z = wp.tid()
    if solid[x, y, z] != wp.uint8(0):
        dye_out[x, y, z] = 0.0
        return
    u = velocity[x, y, z]
    start = wp.vec3(float(x) - u[0], float(y) - u[1], float(z) - u[2])
    value = _trilinear(dye_in, start, nx, ny, nz) * decay
    if x == 0:
        value = stripe[y, z]
    dye_out[x, y, z] = value


class DyeField:
    """A double-buffered scalar field tied to one solver grid."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str,
        stripe: int = 8,
        decay: float = 0.9995,
    ) -> None:
        if stripe < 1:
            raise ValueError("stripe must be at least one cell")
        ny, nz = shape[1], shape[2]
        yy, zz = np.indices((ny, nz))
        pattern = (((yy // stripe) + (zz // stripe)) % 2 == 0).astype(np.float32)
        self.device = device
        self.pattern = pattern
        self._stripe = wp.array(pattern, dtype=wp.float32, device=device)
        self.a = wp.zeros(shape, dtype=wp.float32, device=device)
        self.b = wp.zeros(shape, dtype=wp.float32, device=device)
        self.decay = decay
        self.steps = 0

    def step(self, velocity, solid) -> None:
        """Advect one lattice step using the solver's device velocity/solid."""
        nx, ny, nz = self.a.shape
        wp.launch(
            _advect_kernel,
            dim=(nx, ny, nz),
            inputs=[self.a, self.b, velocity, solid, self._stripe, nx, ny, nz, self.decay],
            device=self.device,
        )
        self.a, self.b = self.b, self.a
        self.steps += 1

    def numpy(self) -> np.ndarray:
        return self.a.numpy()
