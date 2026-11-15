"""Dye advection tests.

The dye is carried by the solver's own velocity field and never feeds back.
Two properties matter: it stays out of solids, and a stripe released at the
inlet really does travel downstream.
"""

import numpy as np

from sstream.dye import DyeField
from sstream.lbm import LbmConfig, LbmSolver, pick_device

DEVICE = pick_device()


def test_stripes_are_emitted_at_the_inlet_and_stay_out_of_the_solid():
    shape = (40, 16, 16)
    solid = np.zeros(shape, dtype=bool)
    solid[10:14, 6:10, 6:10] = True
    solver = LbmSolver(
        LbmConfig(shape=shape, tau=0.6, periodic=(True, True, True), device=DEVICE),
        solid,
    )
    solver.initialize(ux=0.05)
    dye = DyeField(shape, solver.device, stripe=4)
    dye.step(solver.u, solver.solid)

    field = dye.numpy()
    assert np.allclose(field[0], dye.pattern)
    assert np.allclose(field[solid], 0.0)
    assert field[1:].max() == 0.0


def test_a_stripe_travels_downstream_with_the_stream():
    shape = (40, 16, 16)
    solver = LbmSolver(
        LbmConfig(shape=shape, tau=0.6, periodic=(True, True, True), device=DEVICE)
    )
    solver.initialize(ux=0.05)
    dye = DyeField(shape, solver.device, stripe=4)
    for _ in range(120):
        solver.step()
        dye.step(solver.u, solver.solid)

    field = dye.numpy()
    assert field.max() == 1.0                       # the inlet is still emitting
    assert field[2:9].max() > 0.5                   # the front has moved ~6 cells
    assert field[15:].max() < 0.05                  # but not halfway down the tunnel
    assert np.isfinite(field).all()
