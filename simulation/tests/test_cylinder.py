"""Canonical cylinder wake at Re = 100.

The cylinder is generated analytically so the validation has no meshing or
import step hiding in it.  The wake is periodic in y and z and open along x.
At Re = 100 the well-known benchmarks are ``Cd ~ 1.3-1.4`` and ``St ~ 0.165``
for an unconfined cylinder; this short domain has about 13% blockage, which
raises both numbers.  The bands below are deliberately wider than the measured
values so the test detects a broken solver rather than a slightly different
convection scheme.

Measured on this implementation (80 x 60 x 4, D = 8): Cd 1.59, St 0.184,
lift amplitude 0.32 in the same normalization.  The test takes about 20 s on
the RTX 2050 and is the slowest case in the suite.
"""

import numpy as np
import pytest

from sstream.lbm import LbmConfig, LbmSolver, pick_device

DEVICE = pick_device()


def test_cylinder_wake_has_correct_drag_and_shedding():
    nx, ny, nz, diameter = 80, 60, 4, 8.0
    speed, reynolds = 0.05, 100.0
    centre_x, centre_y = nx * 0.28, ny * 0.5

    xs = np.arange(nx)[:, None, None]
    ys = np.arange(ny)[None, :, None]
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, :, :] = (xs - centre_x) ** 2 + (ys - centre_y) ** 2 <= (diameter * 0.5) ** 2

    tau = 0.5 + 3.0 * speed * diameter / reynolds
    solver = LbmSolver(
        LbmConfig(
            shape=(nx, ny, nz),
            tau=tau,
            periodic=(False, True, True),
            inlet_speed=speed,
            device=DEVICE,
        ),
        solid,
    )
    solver.initialize(ux=speed)

    drag, lift = [], []
    for step in range(1, 30001):
        solver.step()
        if step >= 10000 and step % 4 == 0:
            force = solver.total_wall_force()
            drag.append(force[0])
            lift.append(force[1])

    drag = np.asarray(drag)
    lift = np.asarray(lift)
    dynamic = 0.5 * speed * speed * diameter * nz
    cd = drag.mean() / dynamic
    cl_amplitude = 0.5 * (lift.max() - lift.min()) / dynamic

    # Vortex shedding frequency from the lift signal.
    spectrum = np.abs(np.fft.rfft(lift - lift.mean()))
    frequencies = np.fft.rfftfreq(len(lift), d=4.0)
    peak = frequencies[1:][np.argmax(spectrum[1:])]
    strouhal = peak * diameter / speed

    assert 1.35 < cd < 1.85, f"Cd {cd:.3f} outside the confined-cylinder band"
    assert 0.16 < strouhal < 0.21, f"St {strouhal:.4f} outside the Re=100 band"
    assert 0.15 < cl_amplitude < 0.60, f"lift amplitude {cl_amplitude:.3f} is not shedding"
    assert solver.max_speed() < 0.2
    assert np.isfinite(solver.density()).all()

    # The wake must be asymmetric in y; a perfectly symmetric solution means
    # the instability never started and the Strouhal number is meaningless.
    lateral = solver.velocity()[..., 1]
    assert np.abs(lateral).max() > 0.01
