"""D3Q19 lattice-Boltzmann solver (structure-of-arrays layout).

One launch per step: pull streaming with half-way bounce-back, then BGK
collision with a Guo body force and optional Smagorinsky eddy viscosity.

Memory layout
    Populations are stored as ``f[q, x, y, z]`` (SoA).  Warp maps the last
    launch axis to the fastest-varying thread index, so neighbouring threads
    (consecutive ``z``) touch consecutive addresses for every population and
    global loads/stores coalesce.  The old ``f[x, y, z, q]`` layout put
    neighbouring threads 76 bytes apart.  The 19 streamed populations are
    loaded once into a register vector and reused for the moments, the
    Smagorinsky stress and the collision, instead of re-reading global memory.

Boundaries
    ``periodic`` is a per-axis flag.  A non-periodic face that is not the
    inlet or outlet has a wall treatment selected by ``wall_modes`` (a mapping
    from ``x0/x1/y0/y1/z0/z1`` to ``"no_slip"`` or ``"free_slip"``).  The
    default is no-slip half-way bounce-back: streaming from outside the grid
    is replaced by the opposite population from the current cell, which places
    the wall half a cell outside the face.  ``"free_slip"`` uses specular
    reflection (only the normal component flips), which conserves tangential
    momentum.  The x faces are overwritten by the inlet/outlet kernel when
    ``inlet_speed > 0``.

    With ``inlet_speed > 0`` the x=0 face is reset to equilibrium every step.
    The x=nx-1 face either copies its neighbour (``outlet_mode="zero_gradient"``,
    the historical behaviour) or is rebuilt as equilibrium at rho=1 with the
    neighbour's velocity (``outlet_mode="fixed_density"``).  ``tau`` is a
    relaxation time, not a physical viscosity; see
    :class:`sstream.lattice.LatticeUnits` for the conversion.

Force direction
    ``wall_force`` is the momentum handed to the *solid* by the fluid, so a
    car's drag is the x-component summed over its cells.

Warp restrictions that shaped this file: kernels may not read Python tuples
(direction and weight tables are device arrays), and the direction loop must
be a plain ``range(19)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .lattice import CS2, DIRECTIONS, OPPOSITE, TAU_MIN, WEIGHTS

_INITIALISED = False

vec19 = wp.types.vector(length=19, dtype=wp.float32)

# Boundary configuration.  Faces are named by axis and low/high side; the
# order below is the order of the integer codes passed to the kernels.
WALL_FACES = ("x0", "x1", "y0", "y1", "z0", "z1")
WALL_MODE_CODES = {"no_slip": 0, "free_slip": 1}
OUTLET_MODE_CODES = {"zero_gradient": 0, "fixed_density": 1}

# Wind-tunnel defaults chosen from the empty-tunnel sweep in
# docs/diag_data/tunnel_variant_summary.md.  The floor stays no-slip because
# the validation bodies rest on it; the sides and roof are specular so they do
# not build a slowly growing boundary layer, and the outlet pins rho = 1 so
# mass cannot accumulate in a closed tunnel.
TUNNEL_WALL_MODES = {
    "y0": "free_slip",
    "y1": "free_slip",
    "z0": "no_slip",
    "z1": "free_slip",
}
TUNNEL_OUTLET_MODE = "fixed_density"


def _ensure_warp() -> None:
    global _INITIALISED
    if not _INITIALISED:
        wp.init()
        _INITIALISED = True


def pick_device() -> str:
    """Return a usable Warp device: CUDA when present, else CPU."""
    _ensure_warp()
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
    # Per-face wall treatment.  Keys are a subset of WALL_FACES; omitted faces
    # keep the historical half-way bounce-back (no-slip) behaviour.  The x
    # faces are only relevant when ``inlet_speed == 0``: with an inlet the
    # inlet/outlet kernel overwrites them every step.
    wall_modes: dict[str, str] = field(default_factory=dict)
    # "zero_gradient" copies the upstream populations to the outlet;
    # "fixed_density" rebuilds equilibrium at rho=1 with the upstream velocity.
    outlet_mode: str = "zero_gradient"
    # Subtract the rest-state equilibrium population w_i * rho_ref from each
    # bounce-back force link.  Without it a body resting on a wall sees an
    # unbalanced ambient pressure of cs2*rho ~ 0.33 (a dynamic pressure of
    # ~0.0012), which swamps Cl/Cz.  The net drag of a closed body is
    # unchanged to floating-point precision.
    gauge_pressure: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError("shape must have three axes of at least two cells")
        if self.tau < TAU_MIN:
            raise ValueError(
                f"tau must be at least TAU_MIN ({TAU_MIN}) or the scheme is unstable"
            )
        if self.smagorinsky < 0.0:
            raise ValueError("smagorinsky cannot be negative")
        if self.inlet_speed < 0.0:
            raise ValueError("inlet_speed cannot be negative")
        if self.inlet_speed > 0.0 and self.periodic[0]:
            raise ValueError("an inlet and x-periodicity describe two different tunnels")

        self.wall_modes = dict(self.wall_modes)
        unknown = set(self.wall_modes) - set(WALL_FACES)
        if unknown:
            raise ValueError(f"unknown wall faces {sorted(unknown)}; expected {WALL_FACES}")
        for face, mode in self.wall_modes.items():
            if mode not in WALL_MODE_CODES:
                raise ValueError(
                    f"wall mode for '{face}' must be one of {sorted(WALL_MODE_CODES)}, got {mode!r}"
                )
        if self.outlet_mode not in OUTLET_MODE_CODES:
            raise ValueError(
                f"outlet_mode must be one of {sorted(OUTLET_MODE_CODES)}, got {self.outlet_mode!r}"
            )
        self._wall_codes = tuple(
            WALL_MODE_CODES[self.wall_modes.get(face, "no_slip")] for face in WALL_FACES
        )
        self._outlet_code = OUTLET_MODE_CODES[self.outlet_mode]

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
        u_field[x, y, z] = wp.vec3(0.0, 0.0, 0.0)
        for i in range(19):
            f[i, x, y, z] = 0.0
        return
    speed2 = ux * ux + uy * uy + uz * uz
    for i in range(19):
        c = dirs[i]
        cu = float(c[0]) * ux + float(c[1]) * uy + float(c[2]) * uz
        f[i, x, y, z] = weights[i] * (
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
    outlet_mode: int,
    cs2: float,
):
    y, z = wp.tid()
    if solid[0, y, z] == wp.uint8(0):
        speed2 = inlet_speed * inlet_speed
        for i in range(19):
            c = dirs[i]
            cu = float(c[0]) * inlet_speed
            f[i, 0, y, z] = weights[i] * (
                1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * speed2 / cs2
            )
    if solid[nx - 1, y, z] == wp.uint8(0):
        if outlet_mode == 1:
            # Fixed-density equilibrium outlet: take the velocity from the
            # upstream cell but reset the density to one.  A zero-gradient
            # outlet copies the accumulated density as well, which lets mass
            # pile up in a closed tunnel (see docs/DIAGNOSIS_SPHERE.md).
            rho = float(0.0)
            ux = float(0.0)
            uy = float(0.0)
            uz = float(0.0)
            for i in range(19):
                fn = f[i, nx - 2, y, z]
                c = dirs[i]
                rho += fn
                ux += fn * float(c[0])
                uy += fn * float(c[1])
                uz += fn * float(c[2])
            if rho > 0.0:
                ux /= rho
                uy /= rho
                uz /= rho
                speed2 = ux * ux + uy * uy + uz * uz
                for i in range(19):
                    c = dirs[i]
                    cu = float(c[0]) * ux + float(c[1]) * uy + float(c[2]) * uz
                    f[i, nx - 1, y, z] = weights[i] * (
                        1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * speed2 / cs2
                    )
            else:
                for i in range(19):
                    f[i, nx - 1, y, z] = f[i, nx - 2, y, z]
        else:
            for i in range(19):
                f[i, nx - 1, y, z] = f[i, nx - 2, y, z]


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
    tau_min: float,
    smagorinsky: float,
    gx: float,
    gy: float,
    gz: float,
    per_x: int,
    per_y: int,
    per_z: int,
    spec_x: wp.array1d(dtype=wp.int32),
    spec_y: wp.array1d(dtype=wp.int32),
    spec_z: wp.array1d(dtype=wp.int32),
    wall_x0: int,
    wall_x1: int,
    wall_y0: int,
    wall_y1: int,
    wall_z0: int,
    wall_z1: int,
    gauge_pressure: int,
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
    fl = vec19()

    # --- streaming with half-way bounce-back (into registers) ------------
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
            if solid_hit == 1:
                bounced = f_in[opposite[i], x, y, z]
                fl[i] = bounced
                # Fluid momentum changes by -2 c (f_bounced - w_i rho_ref),
                # i.e. subtract the rest-state equilibrium population w_i
                # times the reference density (rho=1) from the bounce-back term.
                # This removes the unbalanced ambient pressure cs2*rho acting on
                # walls and gives gauge drag; for a closed body it leaves the net
                # force unchanged (to floating-point precision).
                if gauge_pressure == 1:
                    fx -= 2.0 * float(c[0]) * (bounced - float(weights[i]))
                    fy -= 2.0 * float(c[1]) * (bounced - float(weights[i]))
                    fz -= 2.0 * float(c[2]) * (bounced - float(weights[i]))
                else:
                    fx -= 2.0 * float(c[0]) * bounced
                    fy -= 2.0 * float(c[1]) * bounced
                    fz -= 2.0 * float(c[2]) * bounced
            else:
                # Domain boundary.  No-slip is the historical full
                # bounce-back; free-slip is specular reflection (only the
                # normal velocity component flips), which conserves
                # tangential momentum.  If any violated face is no-slip we
                # use full bounce-back; otherwise the specular maps compose.
                ref = i
                bounce = int(0)
                if sy < 0:
                    if wall_y0 == 0:
                        bounce = 1
                    else:
                        ref = spec_y[ref]
                elif sy >= ny:
                    if wall_y1 == 0:
                        bounce = 1
                    else:
                        ref = spec_y[ref]
                if sz < 0:
                    if wall_z0 == 0:
                        bounce = 1
                    else:
                        ref = spec_z[ref]
                elif sz >= nz:
                    if wall_z1 == 0:
                        bounce = 1
                    else:
                        ref = spec_z[ref]
                if sx < 0:
                    if wall_x0 == 0:
                        bounce = 1
                    else:
                        ref = spec_x[ref]
                elif sx >= nx:
                    if wall_x1 == 0:
                        bounce = 1
                    else:
                        ref = spec_x[ref]
                if bounce == 1:
                    ref = opposite[i]
                fl[i] = f_in[ref, x, y, z]
        else:
            fl[i] = f_in[i, sx, sy, sz]

    # --- moments ----------------------------------------------------------
    rho = float(0.0)
    ux = float(0.0)
    uy = float(0.0)
    uz = float(0.0)
    for i in range(19):
        f = fl[i]
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
            ne = fl[i] - feq
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
        if tau_eff < tau_min:
            tau_eff = tau_min

    # --- Guo-forced BGK collision ----------------------------------------
    uxe = ux + gx / (2.0 * rho)
    uye = uy + gy / (2.0 * rho)
    uze = uz + gz / (2.0 * rho)
    speed2 = uxe * uxe + uye * uye + uze * uze
    source_scale = 1.0 - 0.5 / tau_eff
    for i in range(19):
        c = dirs[i]
        f = fl[i]
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
        f_out[i, x, y, z] = f - (f - feq) / tau_eff + src

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
        self.device = config.device if config.device else pick_device()
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
        # Specular maps: direction index after reflecting one coordinate.
        # D3Q19 is closed under these reflections, so every entry exists.
        by_direction = {tuple(int(v) for v in d): i for i, d in enumerate(DIRECTIONS)}
        self._spec_x = self._specular_map(by_direction, 0)
        self._spec_y = self._specular_map(by_direction, 1)
        self._spec_z = self._specular_map(by_direction, 2)
        self.solid = wp.array(mask, dtype=wp.uint8, device=self.device)
        self._solid_cells = int(mask.sum())
        self._fluid = mask == 0

        # SoA: population index first, z fastest.
        self.f = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=self.device)
        self.f_new = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=self.device)
        self.rho = wp.zeros((nx, ny, nz), dtype=wp.float32, device=self.device)
        self.u = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=self.device)
        self.force = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=self.device)
        self._force_total = wp.zeros(3, dtype=wp.float32, device=self.device)

        self.initialize()

    def _specular_map(self, by_direction: dict[tuple[int, int, int], int], axis: int):
        """Population index after reflecting the given coordinate axis."""
        mapped = []
        for d in DIRECTIONS:
            c = [int(v) for v in d]
            c[axis] = -c[axis]
            mapped.append(by_direction[tuple(c)])
        return wp.array(np.asarray(mapped, dtype=np.int32), dtype=wp.int32, device=self.device)

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
                TAU_MIN,
                cfg.smagorinsky,
                gx,
                gy,
                gz,
                1 if cfg.periodic[0] else 0,
                1 if cfg.periodic[1] else 0,
                1 if cfg.periodic[2] else 0,
                self._spec_x,
                self._spec_y,
                self._spec_z,
                *cfg._wall_codes,
                cfg.gauge_pressure,
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
                    cfg._outlet_code,
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

    def plane_speed(self, x: int) -> float:
        """Mean speed over a y-z plane, recomputed from live populations.

        Used for the Cd freestream probe: it must be the actual local flow,
        not the nominal inlet speed (nor the cached field at an overwritten
        boundary plane).
        """
        x = int(np.clip(x, 0, self.shape[0] - 1))
        f = np.asarray(self.f[:, x, :, :].numpy(), dtype=np.float64)
        rho = f.sum(axis=0)
        speed2 = np.zeros_like(rho)
        for k in range(3):
            component = np.einsum(
                "i,iyz->yz", DIRECTIONS[:, k].astype(np.float64), f
            ) / rho
            speed2 += component * component
        return float(np.sqrt(speed2).mean())

    def total_mass(self) -> float:
        """Sum of density over fluid cells; a conservation check."""
        density = self.rho.numpy()
        return float(density[self._fluid].sum(dtype=np.float64))

    def max_speed(self) -> float:
        values = self.u.numpy()
        return float(np.sqrt((values * values).sum(axis=-1)).max())
