"""D3Q19 lattice-Boltzmann solver.

The scheme is deliberately one step per launch: pull streaming with half-way
bounce-back, then BGK collision with a Guo body force and optional
Smagorinsky eddy viscosity.  Splitting the kernels would cost more than it
buys, and the single launch is the part that has to be fast.

Boundaries
    ``periodic`` is a per-axis flag.  A non-periodic face that is not the
    inlet or outlet is a no-slip wall: streaming from outside the grid is
    replaced by the opposite population from the current cell, which places
    the wall half a cell outside the face.

    With ``inlet_speed > 0`` the x=0 face is reset to equilibrium every step
    and the x=nx-1 face copies its neighbour (zero-gradient outlet).  ``tau``
    is a relaxation time, not a physical viscosity; see
    :class:`sstream.lattice.LatticeUnits` for the conversion.

Force direction
    ``wall_force`` is the momentum handed to the *solid* by the fluid, so a
    car's drag is the x-component summed over its cells.  In a channel the
    walls receive the momentum the body force feeds in, which makes the
    force balance test meaningful.

Warp restrictions that shaped this file: kernels may not read Python tuples
(direction and weight tables are device arrays), and the direction loop must
be a plain ``range(19)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .lattice import CS2, DIRECTIONS, OPPOSITE, WEIGHTS

_INITIALISED = False


def _ensure_warp() -> None:
    global _INITIALISED
    if not _INITIALISED:
        wp.init()
        _INITIALISED = True


def pick_device(preferred: str | None = None) -> str:
    """Return a usable Warp device, preferring CUDA when it is present."""
    _ensure_warp()
    if preferred:
        return preferred
    return "cuda" if wp.get_cuda_devices() else "cpu"


@dataclass
class LbmConfig:
    """Everything the solver needs that is not the solid mask."""

    shape: tuple[int, int, int]
    tau: float = 0.6
    smagorinsky: float = 0.0
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    periodic: tuple[bool, bool, bool] = (True, False, True)
    inlet_speed: float = 0.0
    device: str | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError("shape must have three axes of at least two cells")
        if self.tau <= 0.5:
            raise ValueError("tau must be greater than 0.5 or the scheme is unstable")
        if self.smagorinsky < 0.0:
            raise ValueError("smagorinsky cannot be negative")
        if self.inlet_speed < 0.0:
            raise ValueError("inlet_speed cannot be negative")
        if self.inlet_speed > 0.0 and self.periodic[0]:
            raise ValueError("an inlet and x-periodicity describe two different tunnels")

    @property
    def use_inlet(self) -> bool:
        return self.inlet_speed > 0.0 and not self.periodic[0]


@wp.kernel
def _init_kernel(
    f: wp.array4d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.uint8),
    dirs: wp.array1d(dtype=wp.vec3i),
    weights: wp.array1d(dtype=wp.float32),
    rho_field: wp.array3d(dtype=wp.float32),
    u_field: wp.array3d(dtype=wp.vec3),
    ux: float,
    uy: float,
    uz: float,
    cs2: float,
):
    x, y, z = wp.tid()
    rho_field[x, y, z] = 1.0
    u_field[x, y, z] = wp.vec3(ux, uy, uz)
    if solid[x, y, z] != wp.uint8(0):
        rho_field[x, y, z] = 1.0
        u_field[x, y, z] = wp.vec3(0.0, 0.0, 0.0)
        for i in range(19):
            f[x, y, z, i] = 0.0
        return
    speed2 = ux * ux + uy * uy + uz * uz
    for i in range(19):
        c = dirs[i]
        cu = float(c[0]) * ux + float(c[1]) * uy + float(c[2]) * uz
        f[x, y, z, i] = weights[i] * (
            1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * speed2 / cs2
        )


@wp.kernel
def _inlet_outlet_kernel(
    f: wp.array4d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.uint8),
    dirs: wp.array1d(dtype=wp.vec3i),
    weights: wp.array1d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    inlet_speed: float,
    cs2: float,
):
    y, z = wp.tid()
    if solid[0, y, z] == wp.uint8(0):
        speed2 = inlet_speed * inlet_speed
        for i in range(19):
            c = dirs[i]
            cu = float(c[0]) * inlet_speed
            f[0, y, z, i] = weights[i] * (
                1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * speed2 / cs2
            )
    if solid[nx - 1, y, z] == wp.uint8(0):
        for i in range(19):
            f[nx - 1, y, z, i] = f[nx - 2, y, z, i]


@wp.kernel
def _force_sum_kernel(
    force: wp.array3d(dtype=wp.vec3),
    total: wp.array1d(dtype=wp.float32),
):
    x, y, z = wp.tid()
    f = force[x, y, z]
    wp.atomic_add(total, 0, f[0])
    wp.atomic_add(total, 1, f[1])
    wp.atomic_add(total, 2, f[2])


@wp.kernel
def _step_kernel(
    f_in: wp.array4d(dtype=wp.float32),
    f_out: wp.array4d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.uint8),
    dirs: wp.array1d(dtype=wp.vec3i),
    opposite: wp.array1d(dtype=wp.int32),
    weights: wp.array1d(dtype=wp.float32),
    rho_field: wp.array3d(dtype=wp.float32),
    u_field: wp.array3d(dtype=wp.vec3),
    force_field: wp.array3d(dtype=wp.vec3),
    nx: int,
    ny: int,
    nz: int,
    tau: float,
    cs2: float,
    smagorinsky: float,
    gx: float,
    gy: float,
    gz: float,
    per_x: int,
    per_y: int,
    per_z: int,
):
    x, y, z = wp.tid()
    if solid[x, y, z] != wp.uint8(0):
        rho_field[x, y, z] = 1.0
        u_field[x, y, z] = wp.vec3(0.0, 0.0, 0.0)
        force_field[x, y, z] = wp.vec3(0.0, 0.0, 0.0)
        return

    fx = float(0.0)
    fy = float(0.0)
    fz = float(0.0)

    # --- streaming with half-way bounce-back -----------------------------
    for i in range(19):
        c = dirs[i]
        sx = x - c[0]
        sy = y - c[1]
        sz = z - c[2]
        if per_x == 1:
            if sx < 0:
                sx += nx
            if sx >= nx:
                sx -= nx
        if per_y == 1:
            if sy < 0:
                sy += ny
            if sy >= ny:
                sy -= ny
        if per_z == 1:
            if sz < 0:
                sz += nz
            if sz >= nz:
                sz -= nz
        blocked = int(0)
        solid_hit = int(0)
        if sx < 0 or sx >= nx or sy < 0 or sy >= ny or sz < 0 or sz >= nz:
            blocked = 1
        if blocked == 0:
            if solid[sx, sy, sz] != wp.uint8(0):
                blocked = 1
                solid_hit = 1
        if blocked == 1:
            bounced = f_in[x, y, z, opposite[i]]
            f_out[x, y, z, i] = bounced
            # Fluid momentum changes by -2 c f_bounced, so this is the
            # momentum handed to the solid.  Reflections off the domain
            # boundary (floor, tunnel walls, inlet, outlet) are not the
            # model, so they are deliberately not counted.
            if solid_hit == 1:
                fx -= 2.0 * float(c[0]) * bounced
                fy -= 2.0 * float(c[1]) * bounced
                fz -= 2.0 * float(c[2]) * bounced
        else:
            f_out[x, y, z, i] = f_in[sx, sy, sz, i]

    # --- moments ----------------------------------------------------------
    rho = float(0.0)
    ux = float(0.0)
    uy = float(0.0)
    uz = float(0.0)
    for i in range(19):
        f = f_out[x, y, z, i]
        c = dirs[i]
        rho += f
        ux += f * float(c[0])
        uy += f * float(c[1])
        uz += f * float(c[2])
    ux /= rho
    uy /= rho
    uz /= rho

    # --- Smagorinsky eddy viscosity --------------------------------------
    tau_eff = tau
    if smagorinsky > 0.0:
        pxx = float(0.0)
        pyy = float(0.0)
        pzz = float(0.0)
        pxy = float(0.0)
        pxz = float(0.0)
        pyz = float(0.0)
        speed2 = ux * ux + uy * uy + uz * uz
        for i in range(19):
            c = dirs[i]
            cu = float(c[0]) * ux + float(c[1]) * uy + float(c[2]) * uz
            feq = weights[i] * rho * (
                1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * speed2 / cs2
            )
            ne = f_out[x, y, z, i] - feq
            pxx += float(c[0]) * float(c[0]) * ne
            pyy += float(c[1]) * float(c[1]) * ne
            pzz += float(c[2]) * float(c[2]) * ne
            pxy += float(c[0]) * float(c[1]) * ne
            pxz += float(c[0]) * float(c[2]) * ne
            pyz += float(c[1]) * float(c[2]) * ne
        pi_norm = wp.sqrt(
            pxx * pxx
            + pyy * pyy
            + pzz * pzz
            + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz)
        )
        disc = tau * tau + 18.0 * wp.sqrt(2.0) * smagorinsky * smagorinsky * pi_norm / rho
        tau_eff = 0.5 * (tau + wp.sqrt(disc))
        if tau_eff < 0.5001:
            tau_eff = 0.5001

    # --- Guo-forced BGK collision ----------------------------------------
    uxe = ux + gx / (2.0 * rho)
    uye = uy + gy / (2.0 * rho)
    uze = uz + gz / (2.0 * rho)
    speed2 = uxe * uxe + uye * uye + uze * uze
    source_scale = 1.0 - 0.5 / tau_eff
    for i in range(19):
        c = dirs[i]
        f = f_out[x, y, z, i]
        cx = float(c[0])
        cy = float(c[1])
        cz = float(c[2])
        cue = cx * uxe + cy * uye + cz * uze
        feq = weights[i] * rho * (
            1.0 + cue / cs2 + 0.5 * cue * cue / (cs2 * cs2) - 0.5 * speed2 / cs2
        )
        cu = cx * ux + cy * uy + cz * uz
        src = source_scale * weights[i] * (
            ((cx - ux) * gx + (cy - uy) * gy + (cz - uz) * gz) / cs2
            + (cu / cs2) * (cx * gx + cy * gy + cz * gz) / cs2
        )
        f_out[x, y, z, i] = f - (f - feq) / tau_eff + src

    rho_field[x, y, z] = rho
    u_field[x, y, z] = wp.vec3(ux, uy, uz)
    force_field[x, y, z] = wp.vec3(fx, fy, fz)


class LbmSolver:
    """A D3Q19 solver over a fixed grid and solid mask.

    ``solid`` is a boolean array shaped like ``config.shape``; the solver keeps
    its own device copy, so the caller may reuse the array.  All public arrays
    are returned as NumPy copies.
    """

    def __init__(self, config: LbmConfig, solid: np.ndarray | None = None) -> None:
        _ensure_warp()
        self.config = config
        self.device = pick_device(config.device)
        self.steps = 0

        nx, ny, nz = config.shape
        mask = np.zeros(config.shape, dtype=np.uint8)
        if solid is not None:
            mask = np.ascontiguousarray(np.asarray(solid, dtype=bool)).astype(np.uint8)
            if mask.shape != config.shape:
                raise ValueError(f"solid mask {mask.shape} does not match grid {config.shape}")

        self._dirs = wp.array(DIRECTIONS, dtype=wp.vec3i, device=self.device)
        self._opposite = wp.array(OPPOSITE, dtype=wp.int32, device=self.device)
        self._weights = wp.array(WEIGHTS, dtype=wp.float32, device=self.device)
        self.solid = wp.array(mask, dtype=wp.uint8, device=self.device)
        self._solid_cells = int(mask.sum())
        self._fluid = mask == 0

        self.f = wp.zeros((nx, ny, nz, 19), dtype=wp.float32, device=self.device)
        self.f_new = wp.zeros((nx, ny, nz, 19), dtype=wp.float32, device=self.device)
        self.rho = wp.zeros((nx, ny, nz), dtype=wp.float32, device=self.device)
        self.u = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=self.device)
        self.force = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=self.device)
        self._force_total = wp.zeros(3, dtype=wp.float32, device=self.device)

        self.initialize()

    # -- state -------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.config.shape

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.config.shape))

    @property
    def fluid_cells(self) -> int:
        return self.cell_count - self._solid_cells

    def initialize(self, ux: float = 0.0, uy: float = 0.0, uz: float = 0.0) -> None:
        """Fill the domain with a uniform equilibrium at rest by default."""
        wp.launch(
            _init_kernel,
            dim=self.shape,
            inputs=[self.f, self.solid, self._dirs, self._weights, self.rho, self.u, ux, uy, uz, CS2],
            device=self.device,
        )
        self.steps = 0

    def step(self) -> None:
        cfg = self.config
        gx, gy, gz = cfg.body_force
        wp.launch(
            _step_kernel,
            dim=self.shape,
            inputs=[
                self.f,
                self.f_new,
                self.solid,
                self._dirs,
                self._opposite,
                self._weights,
                self.rho,
                self.u,
                self.force,
                cfg.shape[0],
                cfg.shape[1],
                cfg.shape[2],
                cfg.tau,
                CS2,
                cfg.smagorinsky,
                gx,
                gy,
                gz,
                1 if cfg.periodic[0] else 0,
                1 if cfg.periodic[1] else 0,
                1 if cfg.periodic[2] else 0,
            ],
            device=self.device,
        )
        if cfg.use_inlet:
            wp.launch(
                _inlet_outlet_kernel,
                dim=(cfg.shape[1], cfg.shape[2]),
                inputs=[
                    self.f_new,
                    self.solid,
                    self._dirs,
                    self._weights,
                    cfg.shape[0],
                    cfg.shape[1],
                    cfg.shape[2],
                    cfg.inlet_speed,
                    CS2,
                ],
                device=self.device,
            )
        self.f, self.f_new = self.f_new, self.f
        self.steps += 1

    def run(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.step()

    # -- readback ----------------------------------------------------------
    def velocity(self) -> np.ndarray:
        """Fluid velocity as ``(nx, ny, nz, 3)`` float32."""
        return self.u.numpy()

    def density(self) -> np.ndarray:
        return self.rho.numpy()

    def wall_force(self) -> np.ndarray:
        """Momentum handed to the solid as ``(nx, ny, nz, 3)`` float32."""
        return self.force.numpy()

    def total_wall_force(self) -> np.ndarray:
        """Sum the per-cell force on the GPU, then copy three floats back."""
        self._force_total.zero_()
        wp.launch(
            _force_sum_kernel,
            dim=self.shape,
            inputs=[self.force, self._force_total],
            device=self.device,
        )
        return self._force_total.numpy().astype(np.float64)

    def total_mass(self) -> float:
        """Sum of density over fluid cells; a conservation check."""
        density = self.rho.numpy()
        return float(density[self._fluid].sum(dtype=np.float64))

    def max_speed(self) -> float:
        values = self.u.numpy()
        return float(np.sqrt((values * values).sum(axis=-1)).max())
