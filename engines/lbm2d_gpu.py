from __future__ import annotations

import cupy as cp
import numpy as np

from engines.base import SimEngine
from engines.lbm_common import LATTICE_2D
from engines.smoke_mixin import SmokeMixin

_KERNEL_SRC = r"""
extern "C" __global__
void fused_step(
    float* f_out, const float* f_in,
    float* rho, float* u, float* v,
    const bool* obstacles,
    const int* opp, const float* w, const int* cx, const int* cy,
    float omega, float u_inflow,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int stride_yx = width;
    int stride_f = height * width;

    float fi[9];
    for (int i = 0; i < 9; i++) {
        int sx = x - cx[i];
        int sy = y - cy[i];
        if (sx < 0) sx += width;
        else if (sx >= width) sx -= width;
        if (sy < 0) sy += height;
        else if (sy >= height) sy -= height;
        fi[i] = f_in[i * stride_f + sy * stride_yx + sx];
    }

    if (obstacles[y * width + x]) {
        for (int i = 0; i < 9; i++) {
            int oi = opp[i];
            if (i < oi) {
                float tmp = fi[i]; fi[i] = fi[oi]; fi[oi] = tmp;
            }
        }
    }

    if (x == 0) {
        float u2_in = u_inflow * u_inflow;
        for (int i = 0; i < 9; i++) {
            float cu = cx[i] * u_inflow;
            fi[i] = w[i] * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*u2_in);
        }
    }

    if (y == 0 || y == height - 1) {
        for (int i = 0; i < 9; i++) {
            int oi = opp[i];
            if (i < oi) {
                float tmp = fi[i]; fi[i] = fi[oi]; fi[oi] = tmp;
            }
        }
    }

    float r = 0.0f, u_vel = 0.0f, v_vel = 0.0f;
    for (int i = 0; i < 9; i++) {
        r += fi[i];
        u_vel += fi[i] * cx[i];
        v_vel += fi[i] * cy[i];
    }
    float rho_safe = r > 0.0f ? r : 1.0f;
    u_vel /= rho_safe;
    v_vel /= rho_safe;

    float u2 = u_vel*u_vel + v_vel*v_vel;
    float om = 1.0f - omega;
    for (int i = 0; i < 9; i++) {
        float cu = cx[i]*u_vel + cy[i]*v_vel;
        float feq = w[i]*r*(1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*u2);
        f_out[i * stride_f + y * stride_yx + x] = fi[i]*om + feq*omega;
    }

    rho[y * width + x] = r;
    u[y * width + x] = u_vel;
    v[y * width + x] = v_vel;
}
"""


class LBM2DGPU(SimEngine, SmokeMixin):
    """D2Q9 Lattice Boltzmann fluid simulation (GPU / CuPy)."""

    def __init__(
        self, width: int = 128, height: int = 128, viscosity: float = 0.02
    ) -> None:
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.u_inflow = 0.15

        self.lattice = LATTICE_2D
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )

        self.f = cp.zeros((9, height, width), dtype=cp.float32)
        self._f_swap = cp.empty_like(self.f)

        self.rho = cp.ones((height, width), dtype=cp.float32)
        self.u = cp.zeros((height, width), dtype=cp.float32)
        self.v = cp.zeros((height, width), dtype=cp.float32)

        self.obstacles = cp.zeros((height, width), dtype=cp.bool_)

        self.smoke = cp.zeros((height, width), dtype=cp.float32)
        self.smoke_diffusion = 0.05
        self.smoke_decay = 0.999
        self.emitters: list[tuple[int, int, float]] = []

        self._x_coords = cp.arange(width, dtype=cp.float32)
        self._y_coords = cp.arange(height, dtype=cp.float32)

        self._cx = cp.array(self.lattice.cx, dtype=cp.int32)
        self._cy = cp.array(self.lattice.cy, dtype=cp.int32)
        self._opp = cp.array(self.lattice.opp, dtype=cp.int32)
        self._w = cp.array(self.lattice.w, dtype=cp.float32)
        self.xp = cp

        self._kernel = cp.RawKernel(_KERNEL_SRC, "fused_step")

        # CUDA streams for overlapping compute and transfer
        self._stream_compute = cp.cuda.Stream()
        self._stream_transfer = cp.cuda.Stream()

        self.initialize(rho=1.0, u=0.1, v=0.0)
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        self.step()
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
        return self.lattice.omega_from_viscosity(self.viscosity)

    def initialize(
        self, rho: float = 1.0, u: float = 0.1, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        cu = (
            self._cx[:, cp.newaxis, cp.newaxis] * self.u[cp.newaxis, :, :]
            + self._cy[:, cp.newaxis, cp.newaxis] * self.v[cp.newaxis, :, :]
        )
        u2 = self.u**2 + self.v**2
        self.f = (
            self._w[:, cp.newaxis, cp.newaxis]
            * self.rho[cp.newaxis, :, :]
            * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2[cp.newaxis, :, :])
        )
        self.smoke[:] = 0.0
        self.emitters.clear()
        self.clear_obstacles()

    def step(self) -> None:
        # Use optimized block size for better occupancy
        # Try 32x8 = 256 threads per block (same as 16x16 but different shape)
        threads = (32, 8)
        blocks = (
            (self.width + 31) // 32,
            (self.height + 7) // 8,
        )

        # Launch compute kernel on compute stream
        with self._stream_compute:
            self._kernel(
                blocks,
                threads,
                (
                    self._f_swap,
                    self.f,
                    self.rho,
                    self.u,
                    self.v,
                    self.obstacles,
                    self._opp,
                    self._w,
                    self._cx,
                    self._cy,
                    self.omega,
                    self.u_inflow,
                    self.height,
                    self.width,
                ),
            )
            self._f_swap[:, :, -1] = self._f_swap[:, :, -2]
            self.f, self._f_swap = self._f_swap, self.f

        self.apply_emitters()
        self.advect_smoke()
        self.diffuse_smoke()
        self.smoke[self.obstacles] = 0.0
        self.decay_smoke()

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def get_density(self) -> np.ndarray:
        return cp.asnumpy(self.rho)

    def get_velocity(self) -> np.ndarray:
        return cp.asnumpy(cp.stack([self.u, self.v], axis=2))

    def get_velocity_at(self, x: int, y: int) -> tuple[float, float]:
        return float(cp.asnumpy(self.u[y, x])), float(cp.asnumpy(self.v[y, x]))

    def get_smoke(self) -> np.ndarray:
        return cp.asnumpy(self.smoke)

    def get_obstacles(self) -> np.ndarray:
        return cp.asnumpy(self.obstacles)

    def get_obstacles_mut(self) -> np.ndarray:
        return self.obstacles

    def get_f(self) -> np.ndarray:
        return self.f

    def get_pressure(self) -> np.ndarray:
        return cp.asnumpy(self.rho - 1.0)

    def get_emitter_count(self) -> int:
        return len(self.emitters)

    def add_obstacle(self, x: int, y: int, radius: int = 5) -> None:
        y_grid, x_grid = cp.ogrid[: self.height, : self.width]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius**2
        self.obstacles[mask] = True

    def clear_obstacles(self) -> None:
        self.obstacles[:] = False

    def add_emitter(self, x: int, y: int, strength: float = 0.05) -> None:
        self.emitters.append((x, y, strength))
