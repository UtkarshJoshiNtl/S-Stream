from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engines.particle_tracer import ParticleTracer
    from scene.scene import Scene


class SimEngine(ABC):
    """
    Abstract interface that every simulation backend must implement.
    The UI layer talks exclusively to this interface — no engine subclass
    is ever referenced directly from ui/.
    """

    # --- Pressure reference ---

    #: Reference (at-rest) density in lattice units. Standard LBM normalises to 1.
    REFERENCE_DENSITY: float = 1.0

    #: Lattice sound speed squared. D2Q9 and D3Q19 both use c_s^2 = 1/3, so the
    #: isothermal pressure is p = c_s^2 * rho for *every* engine on the lattice.
    SOUND_SPEED_SQUARED: float = 1.0 / 3.0

    @classmethod
    def reference_pressure(cls) -> float:
        """Pressure of the reference state, p0 = c_s^2 * rho0 = 1/3."""
        return cls.SOUND_SPEED_SQUARED * cls.REFERENCE_DENSITY

    # --- Dimensionality ---

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Dimensionality of the simulation domain (2 or 3)."""
        ...

    @property
    @abstractmethod
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the scalar field arrays, e.g. (H, W) or (D, H, W)."""
        ...

    # --- Simulation control ---

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one timestep."""
        ...

    @abstractmethod
    def run(self, steps: int) -> None:
        """Advance the simulation by *steps* timesteps."""
        ...

    @abstractmethod
    def initialize(
        self, rho: float = 1.0, u: float = 0.1, v: float = 0.0, w: float = 0.0
    ) -> None:
        """
        Reset the simulation to a uniform state.
        Specific subclasses may accept additional keyword arguments.
        """
        ...

    def reset_for_scene(self, scene: Scene) -> None:
        """Reset to the initial condition a scene should start from.

        Each engine owns its own initial condition. The default suits
        driven (inflow) domains, where the scene's inlet velocity is the
        physically meaningful seed. Engines that are closed-domain or
        seeded with a phase structure must override this: forcing
        ``rho=1.0, u=scene.u_inflow`` on them overwrites the background
        density and start the flow moving before the physics does.

        Callers must use this rather than calling :meth:`initialize`
        directly when applying a scene.
        """
        self.initialize(rho=1.0, u=scene.u_inflow, v=0.0)

    # --- Tunable physical parameters (UI sliders write to these) ---
    # Concrete subclasses MUST set these as instance attributes in __init__.

    viscosity: float
    u_inflow: float
    smoke_diffusion: float
    smoke_decay: float

    # --- Observables (the UI reads these each frame) ---

    @abstractmethod
    def get_density(self) -> np.ndarray:
        """Return a copy of the density field as a NumPy array."""
        ...

    @abstractmethod
    def get_velocity(self) -> np.ndarray:
        """
        Return a copy of the velocity field as a NumPy array.
        Shape: (H, W, 2) for 2D, (D, H, W, 3) for 3D.
        """
        ...

    def get_velocity_view(self) -> np.ndarray:
        """Return a pre-allocated velocity buffer (no copy).

        The returned array is reused between calls — callers must not store
        references across frames. Prefer this over get_velocity() when the
        data is only read (e.g. rendering, particle advection).
        """
        return self.get_velocity()

    def get_velocity_at(self, x: int, y: int) -> tuple[float, float]:
        """Return velocity at a single grid cell without full-array allocation.

        Subclasses should override this for O(1) access. Default falls back to
        get_velocity() which allocates a full copy.
        """
        vel = self.get_velocity()
        return float(vel[y, x, 0]), float(vel[y, x, 1])

    @abstractmethod
    def get_smoke(self) -> np.ndarray:
        """Return a copy of the smoke (passive scalar) field as a NumPy array."""
        ...

    # --- Engine-agnostic field access (Phase 5) ---

    @abstractmethod
    def get_field_names(self) -> list[str]:
        """Return the list of field names this engine supports."""
        ...

    @abstractmethod
    def get_field(self, name: str) -> np.ndarray:
        """
        Return a normalized [0, 1] float32 scalar field for visualization.

        Supported fields (all engines): smoke, speed, vorticity, pressure, density.
        Additional fields: phase (liquid), temperature (thermal).

        Raises ValueError for unsupported field names.
        """
        ...

    # --- Physical (un-normalized) fields ---

    def speed_magnitude(self) -> np.ndarray:
        """|u| from the velocity field, in lattice units per step.

        Only the horizontal components are summed, matching the `speed` render: the
        last axis is x and the one before it is y, so a 3D velocity contributes its
        last two components and its vertical one is not counted as horizontal speed.
        """
        vel = np.asarray(self.get_velocity())
        n_comp = vel.shape[-1]
        if n_comp < 2:
            raise ValueError(
                f"Expected a velocity with at least 2 components, got shape {vel.shape}"
            )
        return np.sqrt(
            sum(
                np.square(vel[..., i].astype(np.float64))
                for i in range(n_comp - 2, n_comp)
            )
        )

    def vorticity(self) -> np.ndarray:
        """omega_z = dv/dx - du/dy, the single definition of the curl.

        Axis order is the same for 2D and 3D: x is always the last axis and y the one
        before it, because both engines store velocity as (..., y, x). Central
        differences, zero-padded at the boundaries.

        This used to be written out six times inside the per-engine `get_field`
        methods. Six copies of a derivative is six chances to disagree about which
        axis is x, and the rendered images would then differ between backends for a
        reason nobody could see.
        """
        vel = np.asarray(self.get_velocity())
        u, v = vel[..., -2].astype(np.float64), vel[..., -1].astype(np.float64)

        dvdx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        dvdx[..., 1:-1] = (v[..., 2:] - v[..., :-2]) * 0.5
        dudy[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) * 0.5
        return dvdx - dudy

    def get_field_raw(self, name: str) -> np.ndarray:
        """The *physical* field, un-normalized, with its own magnitude intact.

        `get_field` exists to be drawn: it returns a `[0, 1]` render whose scale has
        been divided out against an EMA-smoothed running extreme. Statistics,
        thresholds and exports all need the opposite — the value in lattice units —
        and asking `get_field` for it means reading a normalized number and either
        mislabelling it or reverse-engineering a scale that is not there.

        Raises KeyError for a field this engine cannot physically produce, naming
        the engine and the field. Consumers must handle that: the alternative, which
        this method exists to replace, is falling back to *some other* field and
        reporting it under the selected field's name.
        """
        if name == "smoke":
            return np.asarray(self.get_smoke())
        if name == "density":
            return np.asarray(self.get_density())
        if name == "pressure":
            # Gauge, the canonical display quantity (PLAN.md C35/C36).
            return self.get_pressure_gauge()
        if name == "speed":
            return self.speed_magnitude()
        if name == "vorticity":
            return self.vorticity()
        raise KeyError(
            f"{type(self).__name__} has no physical field {name!r}; "
            f"it provides {', '.join(self.get_field_names())}"
        )

    # --- Obstacles ---

    @abstractmethod
    def add_obstacle(self, *args: int, radius: int = 5) -> None:
        """
        Add an obstacle at a given grid position.
        2D: add_obstacle(x, y, radius=5)
        3D: add_obstacle(x, y, z, radius=5)
        """
        ...

    @abstractmethod
    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        ...

    @abstractmethod
    def get_obstacles(self) -> np.ndarray:
        """Return a copy of the obstacle mask as a bool NumPy array."""
        ...

    @abstractmethod
    def get_obstacles_mut(self) -> np.ndarray:
        """Return the mutable obstacle mask directly (no copy)."""
        ...

    # --- Internal state access (for analysis) ---

    @abstractmethod
    def get_f(self) -> np.ndarray:
        """Return the distribution arrays f (shape: (9, H, W))."""
        ...

    @abstractmethod
    def get_pressure(self) -> np.ndarray:
        """Absolute lattice pressure p = c_s^2 * rho (c_s^2 = 1/3).

        Absolute, not gauge: this is the thermodynamic quantity the isothermal
        equation of state gives, and it is what boundary conditions that prescribe
        an outlet pressure (`rho = 3p`) are stated in. It carries the 1/3 offset of
        the reference state, so it is the wrong thing to *display* — see
        `get_pressure_gauge`, which is the canonical display and analysis quantity.
        """
        ...

    def get_pressure_gauge(self) -> np.ndarray:
        """Gauge pressure p' = p - p0 = c_s^2 * (rho - rho0), in lattice units.

        **This is the canonical pressure quantity.** Every display, contour, probe
        reading, statistic, metric and report reads this and not `get_pressure`.

        It is defined once, here, rather than per engine. It used to exist only on
        `LBM2D`, so an analysis routine reaching for it on any other backend raised
        `AttributeError`, and each engine compensated by inventing its own display
        normalization: `LBM2D` used the gauge, `LBM3D` and the GPU backends used
        `rho - 1`. Those differ by exactly a factor of 3 — `(rho - 1)/3` versus
        `rho - 1` — so the same physical field read differently depending on which
        backend produced it. Subclasses must not override this; if a backend has a
        genuinely different equation of state it should change
        `SOUND_SPEED_SQUARED` and `REFERENCE_DENSITY`, which propagates here.
        """
        return self.get_pressure() - self.reference_pressure()

    # --- Emitters ---

    @abstractmethod
    def add_emitter(self, *args, strength: float = 0.05) -> None:
        """
        Add a smoke emitter at a given grid position.
        2D: add_emitter(x, y, strength=0.05)
        3D: add_emitter(x, y, z, strength=0.05)
        """
        ...

    @abstractmethod
    def clear_emitters(self) -> None:
        """Remove all emitters."""
        ...

    @abstractmethod
    def get_emitter_count(self) -> int:
        """Return the number of active emitters."""
        ...

    # --- Particle tracer ---

    def get_particle_tracer(self) -> ParticleTracer | None:
        """Return the particle tracer, or None if not supported."""
        return None

    # --- Plotting (Jupyter / matplotlib integration) ---

    def plot_field(
        self,
        name: str,
        *,
        cmap: str = "viridis",
        title: str | None = None,
        figsize: tuple[int, int] = (6, 5),
        colorbar: bool = True,
        ax: object | None = None,
    ) -> object:
        """Plot a normalized field using matplotlib.

        Args:
            name: Field name (e.g. 'smoke', 'speed', 'vorticity', 'pressure').
            cmap: Matplotlib colormap name.
            title: Plot title. Defaults to field name.
            figsize: Figure size in inches (width, height).
            colorbar: Whether to draw a colorbar.
            ax: Existing matplotlib Axes to draw on. If None, creates a new figure.

        Returns:
            matplotlib Axes object.
        """
        import matplotlib.pyplot as plt

        field = self.get_field(name)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title or name.capitalize())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax

    def plot_velocity(
        self,
        *,
        figsize: tuple[int, int] = (7, 5),
        colorbar: bool = True,
        ax: object | None = None,
    ) -> object:
        """Plot velocity magnitude with quiver overlay.

        Returns:
            matplotlib Axes object.
        """
        import matplotlib.pyplot as plt

        vel = self.get_velocity()
        if vel.ndim == 3 and vel.shape[2] >= 2:
            u, v = vel[:, :, 0], vel[:, :, 1]
        else:
            return self.plot_field("speed", figsize=figsize, colorbar=colorbar, ax=ax)

        speed = np.sqrt(u**2 + v**2)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(speed, cmap="coolwarm", origin="lower")
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Speed")

        skip = max(1, min(u.shape[0], u.shape[1]) // 16)
        y_grid, x_grid = np.mgrid[: u.shape[0] : skip, : u.shape[1] : skip]
        ax.quiver(
            x_grid,
            y_grid,
            u[::skip, ::skip],
            v[::skip, ::skip],
            color="white",
            alpha=0.7,
            scale=None,
        )
        ax.set_title("Velocity")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax

    def plot_pressure(
        self,
        *,
        figsize: tuple[int, int] = (6, 5),
        colorbar: bool = True,
        ax: object | None = None,
    ) -> object:
        """Plot the gauge pressure field, p' = c_s^2 (rho - rho0).

        Symmetric about zero, since gauge pressure is: the reference state plots
        as mid-colour and a symmetric axis makes the high/low lobes comparable.
        The docstring used to claim `rho - 1` while the body plotted `rho / 3`,
        which is a different quantity by a factor of 3 plus an offset — harmless
        in shape, but it is what made the definition ambiguous everywhere else.

        Returns:
            matplotlib Axes object.
        """
        import matplotlib.pyplot as plt

        p = self.get_pressure_gauge()
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        lim = max(float(np.percentile(np.abs(p), 98)), 1e-6)
        im = ax.imshow(p, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p' (lattice)")
        ax.set_title("Gauge pressure  p' = (rho - 1)/3")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax

    def plot_smoke(
        self,
        *,
        figsize: tuple[int, int] = (6, 5),
        colorbar: bool = True,
        ax: object | None = None,
    ) -> object:
        """Plot the smoke (passive scalar) field.

        Returns:
            matplotlib Axes object.
        """
        return self.plot_field(
            "smoke",
            cmap="gray",
            title="Smoke",
            figsize=figsize,
            colorbar=colorbar,
            ax=ax,
        )

    def plot_vorticity(
        self,
        *,
        figsize: tuple[int, int] = (6, 5),
        colorbar: bool = True,
        ax: object | None = None,
    ) -> object:
        """Plot the vorticity field.

        Returns:
            matplotlib Axes object.
        """
        return self.plot_field(
            "vorticity",
            cmap="RdBu_r",
            title="Vorticity",
            figsize=figsize,
            colorbar=colorbar,
            ax=ax,
        )

    def _repr_png_(self) -> bytes | None:
        """Return PNG bytes for inline Jupyter display."""
        try:
            import io

            import matplotlib.pyplot as plt

            field = self.get_field("smoke")
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.imshow(field, cmap="gray", origin="lower")
            ax.set_title("S-Stream Simulation")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception:
            return None

    def _repr_html_(self) -> str | None:
        """Return HTML img tag for inline Jupyter display."""
        png = self._repr_png_()
        if png is None:
            return None
        import base64

        b64 = base64.b64encode(png).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" />'
