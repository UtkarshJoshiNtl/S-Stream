"""Fluid–structure interaction (rigid body + IBM) — Phase D Experimental.

Couples RigidIBM markers to a 2D rigid-body ODE (translation only) using
the net IBM force. Soft toys for fluttering / falling discs — not a
full structural solver.
"""

from __future__ import annotations

import numpy as np

from engines.ibm import RigidIBM


class RigidBody2D:
    def __init__(self, mass: float = 10.0) -> None:
        self.mass = mass
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def integrate(self, fx: float, fy: float, dt: float = 1.0) -> None:
        ax = fx / self.mass
        ay = fy / self.mass
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt


class FSIRigidCircle:
    """Falling/forced circle: update IBM markers from rigid-body state."""

    def __init__(
        self,
        ibm: RigidIBM,
        radius: float,
        mass: float = 20.0,
        n_markers: int | None = None,
    ) -> None:
        self.ibm = ibm
        self.radius = radius
        self.body = RigidBody2D(mass=mass)
        self.n_markers = n_markers

    def reset(self, x: float, y: float) -> None:
        self.body.x = x
        self.body.y = y
        self.body.vx = 0.0
        self.body.vy = 0.0
        self.ibm.set_circle(x, y, self.radius, self.n_markers)

    def step(self, gravity: float = 0.0) -> None:
        # Net force ≈ −sum of Eulerian IBM force (action-reaction)
        fx = -float(np.sum(self.ibm.fx))
        fy = -float(np.sum(self.ibm.fy)) + gravity * self.body.mass
        self.body.integrate(fx, fy, dt=1.0)
        self.ibm.set_circle(
            self.body.x,
            self.body.y,
            self.radius,
            self.n_markers,
            u_body=self.body.vx,
            v_body=self.body.vy,
        )
