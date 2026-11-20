"""Permanent validation regression tests with stored baselines.

These tests catch regressions in the LBM solver by comparing against
measured baseline values with tight tolerances.  They complement the
unit tests (which check correctness properties) by checking quantitative
accuracy.

Baselines were measured on RTX 2050 with the current implementation.
Tolerances are ±5% for Cd, ±10% for Strouhal, ±10% for Cl amplitude.

Reference: Schafer & Turek (1996) "Benchmark Computations of Laminar Flow
Around a Cylinder", Notes on Numerical Fluid Mechanics 52, 547-566.
Their 2D-2 case (Re=100, parabolic inflow) gives Cd≈3.2, St≈0.30 in
their normalization.  Our uniform-inlet confined cylinder (80x60x4,
D=8, ~13% blockage) gives Cd≈1.59, St≈0.184 — different geometry and
normalization, so we use measured baselines rather than literature values.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from sstream.lbm import LbmConfig, LbmSolver, pick_device

DEVICE = pick_device()

# Baseline values measured on this implementation (RTX 2050, FP32)
POISEUILLE_BASELINE = {
    "max_error": 0.0038,  # 0.38% max relative error vs exact solution
    "tolerance": 0.015,   # 1.5% threshold
}

CYLINDER_BASELINE = {
    "cd": 1.583,
    "cd_tolerance": 0.08,      # ±5%
    "strouhal": 0.1840,
    "strouhal_tolerance": 0.018,  # ±10%
    "cl_amplitude": 0.324,
    "cl_tolerance": 0.032,     # ±10%
}


def test_poiseuille_regression():
    """Poiseuille flow must match exact solution within baseline tolerance."""
    from sstream.lattice import CS2

    nx, ny, nz = 4, 20, 4
    tau = 0.65
    target_speed = 0.04

    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    nu = CS2 * (tau - 0.5)
    height = ny - 2
    force = 8.0 * nu * target_speed / (height * height)

    solver = LbmSolver(
        LbmConfig(
            shape=(nx, ny, nz),
            tau=tau,
            smagorinsky=0.0,
            body_force=(force, 0.0, 0.0),
            periodic=(True, False, True),
            device=DEVICE,
        ),
        solid,
    )
    solver.run(12000)

    u = solver.velocity()[0, :, 0, 0]
    y = np.arange(ny)
    fluid = (y >= 1) & (y <= ny - 2)
    exact = force / (2.0 * nu) * (y - 0.5) * (ny - 1.5 - y)

    error = np.abs(u[fluid] - exact[fluid]).max() / exact[fluid].max()

    assert error < POISEUILLE_BASELINE["tolerance"], (
        f"Poiseuille error {error:.4f} exceeds tolerance "
        f"{POISEUILLE_BASELINE['tolerance']}"
    )
    # Check we're not significantly worse than baseline
    assert error < POISEUILLE_BASELINE["max_error"] * 2, (
        f"Poiseuille error {error:.4f} is >2x baseline "
        f"{POISEUILLE_BASELINE['max_error']}"
    )


def test_cylinder_regression(tmp_path):
    """Cylinder Re=100 must match baseline Cd, St, Cl within tolerance."""
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
            smagorinsky=0.0,
            inlet_speed=speed,
            outlet_mode="zero_gradient",
            periodic=(False, True, True),
            device=DEVICE,
        ),
        solid,
    )
    solver.initialize(ux=speed)

    drag_samples = []
    lift_samples = []
    step_numbers = []

    for step in range(1, 30001):
        solver.step()
        if step >= 10000 and step % 4 == 0:
            force = solver.total_wall_force()
            drag_samples.append(force[0])
            lift_samples.append(force[1])
            step_numbers.append(step)

    drag = np.asarray(drag_samples)
    lift = np.asarray(lift_samples)

    frontal_area = float(diameter * nz)
    dynamic_pressure = 0.5 * speed * speed * frontal_area

    cd = float(drag.mean() / dynamic_pressure)
    cl_amplitude = float(0.5 * (lift.max() - lift.min()) / dynamic_pressure)

    spectrum = np.abs(np.fft.rfft(lift - lift.mean()))
    frequencies = np.fft.rfftfreq(len(lift), d=4.0)
    peak = frequencies[1:][np.argmax(spectrum[1:])]
    strouhal = float(peak * diameter / speed)

    # Export force history to CSV for analysis
    csv_path = tmp_path / "force_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "drag_force", "lift_force", "cd", "cl"])
        for i, step in enumerate(step_numbers):
            writer.writerow([
                step,
                f"{drag[i]:.6f}",
                f"{lift[i]:.6f}",
                f"{drag[i] / dynamic_pressure:.6f}",
                f"{lift[i] / dynamic_pressure:.6f}",
            ])

    # Assert against baselines with tolerances
    baseline = CYLINDER_BASELINE

    assert abs(cd - baseline["cd"]) < baseline["cd_tolerance"], (
        f"Cd {cd:.3f} differs from baseline {baseline['cd']:.3f} "
        f"by {abs(cd - baseline['cd']):.3f} (tolerance {baseline['cd_tolerance']})"
    )

    assert abs(strouhal - baseline["strouhal"]) < baseline["strouhal_tolerance"], (
        f"St {strouhal:.4f} differs from baseline {baseline['strouhal']:.4f} "
        f"by {abs(strouhal - baseline['strouhal']):.4f} "
        f"(tolerance {baseline['strouhal_tolerance']})"
    )

    assert abs(cl_amplitude - baseline["cl_amplitude"]) < baseline["cl_tolerance"], (
        f"Cl amplitude {cl_amplitude:.3f} differs from baseline "
        f"{baseline['cl_amplitude']:.3f} by {abs(cl_amplitude - baseline['cl_amplitude']):.3f} "
        f"(tolerance {baseline['cl_tolerance']})"
    )

    # Sanity checks
    assert solver.max_speed() < 0.2
    assert np.isfinite(solver.density()).all()

    # Wake must be asymmetric (vortex shedding active)
    lateral = solver.velocity()[..., 1]
    assert np.abs(lateral).max() > 0.01


def test_validation_baselines_documented():
    """Ensure baselines are documented and reasonable."""
    # Poiseuille should have very low error
    assert POISEUILLE_BASELINE["max_error"] < 0.01
    assert POISEUILLE_BASELINE["tolerance"] < 0.05

    # Cylinder baselines should be in physical ranges
    assert 1.0 < CYLINDER_BASELINE["cd"] < 2.0
    assert 0.1 < CYLINDER_BASELINE["strouhal"] < 0.3
    assert 0.1 < CYLINDER_BASELINE["cl_amplitude"] < 1.0

    # Tolerances should be reasonable (not too tight, not too loose)
    assert 0.01 < CYLINDER_BASELINE["cd_tolerance"] < 0.2
    assert 0.005 < CYLINDER_BASELINE["strouhal_tolerance"] < 0.05
    assert 0.01 < CYLINDER_BASELINE["cl_tolerance"] < 0.1
