from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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

    @property
    @abstractmethod
    def omega(self) -> float:
        """Relaxation parameter derived from viscosity (BGK)."""
        ...

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

    @abstractmethod
    def get_smoke(self) -> np.ndarray:
        """Return a copy of the smoke (passive scalar) field as a NumPy array."""
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
