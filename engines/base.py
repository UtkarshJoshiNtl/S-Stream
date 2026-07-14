from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engines.particle_tracer import ParticleTracer


class SimEngine(ABC):
    """
    Abstract interface that every simulation backend must implement.
    The UI layer talks exclusively to this interface — no engine subclass
    is ever referenced directly from ui/.
    """

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
        """Return the pressure field (rho - 1.0)."""
        ...

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

        speed = np.sqrt(u ** 2 + v ** 2)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(speed, cmap="coolwarm", origin="lower")
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Speed")

        skip = max(1, min(u.shape[0], u.shape[1]) // 16)
        y_grid, x_grid = np.mgrid[: u.shape[0] : skip, : u.shape[1] : skip]
        ax.quiver(x_grid, y_grid, u[::skip, ::skip], v[::skip, ::skip],
                  color="white", alpha=0.7, scale=None)
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
        """Plot the pressure field (rho - 1).

        Returns:
            matplotlib Axes object.
        """
        import matplotlib.pyplot as plt

        p = self.get_pressure()
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        lim = max(float(np.percentile(np.abs(p), 98)), 0.001)
        im = ax.imshow(p, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pressure")
        ax.set_title("Pressure")
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
            "smoke", cmap="gray", title="Smoke", figsize=figsize,
            colorbar=colorbar, ax=ax,
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
            "vorticity", cmap="RdBu_r", title="Vorticity", figsize=figsize,
            colorbar=colorbar, ax=ax,
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
