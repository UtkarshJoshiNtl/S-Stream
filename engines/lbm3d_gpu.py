"""CuPy D3Q19 fused BGK engine (Experimental).

Closed domain: bounce-back on all six faces; no smoke for v1.
Memory target: 64³ interactive (~40 MB for double-buffered f + macros).
At 96³, f double-buffer alone is ~130 MB — still fine on consumer GPUs.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from engines.base import SimEngine
from engines.lbm_common import LATTICE_3D_Q19

_KERNEL_SRC = r"""
extern "C" __global__
void fused_step_d3q19(
    float* f_out, const float* f_in,
    float* rho, float* u, float* v, float* w_vel,
    const bool* obstacles,
    const int* opp, const float* wgt,
    const int* cx, const int* cy, const int* cz,
    float omega,
    int depth, int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = depth * height * width;
    if (idx >= n) return;

    int x = idx % width;
    int y = (idx / width) % height;
    int z = idx / (width * height);
    int stride_f = n;

    float fi[19];
    for (int i = 0; i < 19; i++) {
        int sx = x - cx[i];
        int sy = y - cy[i];
        int sz = z - cz[i];
        if (sx < 0) sx += width;
        else if (sx >= width) sx -= width;
        if (sy < 0) sy += height;
        else if (sy >= height) sy -= height;
        if (sz < 0) sz += depth;
        else if (sz >= depth) sz -= depth;
        int sidx = sz * height * width + sy * width + sx;
        fi[i] = f_in[i * stride_f + sidx];
    }

    bool wall = (x == 0 || x == width - 1 ||
                 y == 0 || y == height - 1 ||
                 z == 0 || z == depth - 1);
    if (wall || obstacles[idx]) {
        for (int i = 0; i < 19; i++) {
            int oi = opp[i];
            if (i < oi) {
                float tmp = fi[i]; fi[i] = fi[oi]; fi[oi] = tmp;
            }
        }
    }

    float r = 0.0f, uu = 0.0f, vv = 0.0f, ww = 0.0f;
    for (int i = 0; i < 19; i++) {
        r += fi[i];
        uu += fi[i] * (float)cx[i];
        vv += fi[i] * (float)cy[i];
        ww += fi[i] * (float)cz[i];
    }
    float rho_safe = r > 0.0f ? r : 1.0f;
    uu /= rho_safe;
    vv /= rho_safe;
    ww /= rho_safe;

    float u2 = uu*uu + vv*vv + ww*ww;
    float om = 1.0f - omega;
    for (int i = 0; i < 19; i++) {
        float cu = (float)cx[i]*uu + (float)cy[i]*vv + (float)cz[i]*ww;
        float feq = wgt[i]*r*(1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*u2);
        f_out[i * stride_f + idx] = fi[i]*om + feq*omega;
    }

    rho[idx] = r;
    u[idx] = uu;
    v[idx] = vv;
    w_vel[idx] = ww;
}
"""


class LBM3DGPU(SimEngine):
    """D3Q19 Lattice Boltzmann on CuPy — closed box, fused BGK."""

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        depth: int = 64,
        viscosity: float = 0.02,
    ) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.viscosity = viscosity
        self.u_inflow = 0.05
        self.smoke_diffusion = 0.0
        self.smoke_decay = 1.0

        self.lattice = LATTICE_3D_Q19
        self.lattice.assert_stable(
            viscosity, self.lattice.omega_from_viscosity(viscosity)
        )

        shape = (depth, height, width)
        self.f = cp.zeros((19, *shape), dtype=cp.float32)
        self._f_swap = cp.empty_like(self.f)
        self.rho = cp.ones(shape, dtype=cp.float32)
        self.u = cp.zeros(shape, dtype=cp.float32)
        self.v = cp.zeros(shape, dtype=cp.float32)
        self.w_vel = cp.zeros(shape, dtype=cp.float32)
        self.obstacles = cp.zeros(shape, dtype=cp.bool_)
        self._smoke_host = np.zeros(shape, dtype=np.float32)
        self.emitters: list = []

        self._cx = cp.array(self.lattice.cx, dtype=cp.int32)
        self._cy = cp.array(self.lattice.cy, dtype=cp.int32)
        self._cz = cp.array(self.lattice.cz, dtype=cp.int32)
        self._opp = cp.array(self.lattice.opp, dtype=cp.int32)
        self._w = cp.array(self.lattice.w, dtype=cp.float32)

        self._kernel = cp.RawKernel(_KERNEL_SRC, "fused_step_d3q19")
        self._stream_compute = cp.cuda.Stream()
        self._stream_transfer = cp.cuda.Stream()

        self._ema_speed_max = 0.001
        self._ema_pres_max = 0.001
        self._ema_alpha = 0.05

        self.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        self.step()
        self.initialize(rho=1.0, u=0.0, v=0.0, w=0.0)

    def _asnumpy(self, arr: cp.ndarray) -> np.ndarray:
        with self._stream_transfer:
            out = cp.asnumpy(arr)
        self._stream_transfer.synchronize()
        return out

    @property
    def ndim(self) -> int:
        return 3

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return (self.depth, self.height, self.width)

    @property
    def omega(self) -> float:
        return self.lattice.omega_from_viscosity(self.viscosity)

    def initialize(
        self, rho: float = 1.0, u: float = 0.0, v: float = 0.0, w: float = 0.0
    ) -> None:
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.w_vel[:] = w
        rho_h = self._asnumpy(self.rho)
        u_h = self._asnumpy(self.u)
        v_h = self._asnumpy(self.v)
        w_h = self._asnumpy(self.w_vel)
        feq = self.lattice.equilibrium(rho_h, u_h, v_h, w_h)
        self.f = cp.asarray(feq, dtype=cp.float32)
        self._f_swap = cp.empty_like(self.f)
        self.clear_obstacles()
        self.emitters.clear()

    def step(self, physics_only: bool = False) -> None:
        # No smoke path yet; physics_only kept for CLI / ABC API compatibility.
        _ = physics_only
        n = self.depth * self.height * self.width
        threads = 256
        blocks = (n + threads - 1) // threads
        with self._stream_compute:
            self._kernel(
                (blocks,),
                (threads,),
                (
                    self._f_swap,
                    self.f,
                    self.rho,
                    self.u,
                    self.v,
                    self.w_vel,
                    self.obstacles,
                    self._opp,
                    self._w,
                    self._cx,
                    self._cy,
                    self._cz,
                    float(self.omega),
                    self.depth,
                    self.height,
                    self.width,
                ),
            )
            self.f, self._f_swap = self._f_swap, self.f

    def run(self, steps: int, physics_only: bool = False) -> None:
        for _ in range(steps):
            self.step(physics_only=physics_only)

    def get_density(self) -> np.ndarray:
        return self._asnumpy(self.rho)

    def get_velocity(self) -> np.ndarray:
        return self._asnumpy(cp.stack([self.u, self.v, self.w_vel], axis=3))

    def get_smoke(self) -> np.ndarray:
        return self._smoke_host.copy()

    def get_obstacles(self) -> np.ndarray:
        return self._asnumpy(self.obstacles)

    def get_obstacles_mut(self) -> np.ndarray:
        return self.obstacles

    def get_f(self) -> np.ndarray:
        return self._asnumpy(self.f)

    def get_pressure(self) -> np.ndarray:
        return self._asnumpy(self.rho / 3.0)

    def get_field_names(self) -> list[str]:
        return ["smoke", "speed", "vorticity", "pressure", "density"]

    def get_field(self, name: str) -> np.ndarray:
        a = self._ema_alpha
        if name == "smoke":
            return np.zeros(self.grid_shape, dtype=np.float32)
        vel = self.get_velocity()
        u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
        if name == "speed":
            speed = np.sqrt(u * u + v * v + w * w).astype(np.float32)
            cur_max = max(float(np.max(speed)), 0.001)
            self._ema_speed_max = (1 - a) * self._ema_speed_max + a * cur_max
            mx = max(self.u_inflow * 1.5, self._ema_speed_max, 0.001)
            return np.clip(speed / mx, 0, 1).astype(np.float32)
        if name == "vorticity":
            # Mid-plane z-component proxy for viz
            mid = self.depth // 2
            uu, vv = u[mid], v[mid]
            dvdx = np.zeros_like(uu, dtype=np.float32)
            dudy = np.zeros_like(uu, dtype=np.float32)
            dvdx[:, 1:-1] = (vv[:, 2:] - vv[:, :-2]) * 0.5
            dudy[1:-1, :] = (uu[2:, :] - uu[:-2, :]) * 0.5
            vort = dvdx - dudy
            out = np.zeros(self.grid_shape, dtype=np.float32)
            mx = max(float(np.max(np.abs(vort))), 0.001)
            out[mid] = np.clip(vort / mx * 0.5 + 0.5, 0, 1)
            return out
        if name == "pressure":
            p = (self.get_density() - 1.0).astype(np.float32)
            cur_max = max(float(np.max(np.abs(p))), 0.001)
            self._ema_pres_max = (1 - a) * self._ema_pres_max + a * cur_max
            return np.clip(p / self._ema_pres_max * 0.5 + 0.5, 0, 1).astype(np.float32)
        if name == "density":
            rho = self.get_density()
            lo, hi = float(np.min(rho)), float(np.max(rho))
            if hi - lo < 0.001:
                return np.full_like(rho, 0.5, dtype=np.float32)
            return np.clip((rho - lo) / (hi - lo), 0, 1).astype(np.float32)
        raise ValueError(
            f"Unknown field: {name!r}. Available: {self.get_field_names()}"
        )

    def add_obstacle(self, x: int, y: int, z: int, radius: int = 5) -> None:
        zg, yg, xg = cp.ogrid[: self.depth, : self.height, : self.width]
        mask = (xg - x) ** 2 + (yg - y) ** 2 + (zg - z) ** 2 <= radius**2
        self.obstacles[mask] = True

    def clear_obstacles(self) -> None:
        self.obstacles[:] = False

    def add_emitter(self, *args, strength: float = 0.05) -> None:
        pass  # no smoke in v1

    def clear_emitters(self) -> None:
        self.emitters.clear()

    def get_emitter_count(self) -> int:
        return 0
