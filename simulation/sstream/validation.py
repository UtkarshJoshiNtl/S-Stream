"""Validation cases for the LBM solver.

Implements standard validation cases with known solutions:
- Plane Poiseuille flow (exact solution)
- Flow past a cylinder at Re=100 (literature Cd ≈ 1.0-1.2)

Records baseline performance and accuracy metrics for comparison
after optimizations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lbm import LbmConfig, LbmSolver, pick_device
from .lattice import CS2


@dataclass
class ValidationResult:
    """Results from a validation case."""
    name: str
    device: str
    grid_shape: tuple[int, int, int]
    cells: int
    steps: int
    elapsed_s: float
    mlups: float
    cd: float | None = None
    cl: float | None = None
    error_metric: float | None = None
    error_threshold: float | None = None
    passed: bool = False
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def run_poiseuille_validation(
    ny: int = 20,
    tau: float = 0.65,
    target_speed: float = 0.04,
    steps: int = 12000,
    device: str | None = None,
) -> ValidationResult:
    """Validate against exact plane Poiseuille solution.

    The exact solution for plane Poiseuille flow driven by a body force is:
        u(y) = (F / (2*nu)) * (y - y_wall) * (y_wall_top - y)

    where F is the body force, nu is kinematic viscosity, and walls are at
    y=0.5 and y=ny-1.5 (half-way bounce-back).
    """
    if device is None:
        device = pick_device()

    nx, nz = 4, 4
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    nu = CS2 * (tau - 0.5)
    height = ny - 2
    force = 8.0 * nu * target_speed / (height * height)

    config = LbmConfig(
        shape=(nx, ny, nz),
        tau=tau,
        smagorinsky=0.0,
        body_force=(force, 0.0, 0.0),
        periodic=(True, False, True),
        device=device,
    )

    solver = LbmSolver(config, solid)

    start = time.perf_counter()
    solver.run(steps)
    elapsed = time.perf_counter() - start

    # Compute error against exact solution
    u = solver.velocity()[0, :, 0, 0]
    y = np.arange(ny)
    fluid = (y >= 1) & (y <= ny - 2)
    exact = force / (2.0 * nu) * (y - 0.5) * (ny - 1.5 - y)

    error = np.abs(u[fluid] - exact[fluid]).max() / exact[fluid].max()
    threshold = 0.015  # 1.5% error tolerance

    cells = nx * ny * nz
    mlups = cells * steps / elapsed / 1e6

    return ValidationResult(
        name="Poiseuille Flow",
        device=device,
        grid_shape=(nx, ny, nz),
        cells=cells,
        steps=steps,
        elapsed_s=elapsed,
        mlups=mlups,
        error_metric=float(error),
        error_threshold=threshold,
        passed=error < threshold,
        details={
            "tau": tau,
            "nu": nu,
            "force": force,
            "target_speed": target_speed,
            "max_velocity": float(u[fluid].max()),
            "exact_max_velocity": float(exact[fluid].max()),
        },
    )


def run_cylinder_validation(
    diameter: float = 8.0,
    length: int = 80,
    height: int = 60,
    depth: int = 4,
    speed: float = 0.05,
    reynolds: float = 100.0,
    steps: int = 30000,
    device: str | None = None,
    output_dir: Path | None = None,
) -> ValidationResult:
    """Validate flow past a cylinder at Re=100.

    Matches the canonical test in ``tests/test_cylinder.py``: a 2D cylinder
    extruded along z, periodic in y/z, inlet/outlet along x.  The measured
    values on this implementation are Cd ≈ 1.59, St ≈ 0.184 (confined by
    ~13% blockage, so above the unconfined literature Cd ≈ 1.3).

    Force is sampled every 4 steps after step 10000 to allow the wake to
    develop and reach periodic vortex shedding.
    """
    if device is None:
        device = pick_device()

    nx, ny, nz = length, height, depth
    centre_x, centre_y = nx * 0.28, ny * 0.5

    xs = np.arange(nx)[:, None, None]
    ys = np.arange(ny)[None, :, None]
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, :, :] = (xs - centre_x) ** 2 + (ys - centre_y) ** 2 <= (diameter * 0.5) ** 2

    # tau chosen so that Re = u * D / nu = 100 with nu = cs2 * (tau - 0.5)
    tau = 0.5 + 3.0 * speed * diameter / reynolds

    config = LbmConfig(
        shape=(nx, ny, nz),
        tau=tau,
        smagorinsky=0.0,
        inlet_speed=speed,
        outlet_mode="zero_gradient",
        periodic=(False, True, True),
        device=device,
    )

    solver = LbmSolver(config, solid)
    solver.initialize(ux=speed)

    start = time.perf_counter()
    drag_samples = []
    lift_samples = []
    step_numbers = []
    for step in range(1, steps + 1):
        solver.step()
        if step >= 10000 and step % 4 == 0:
            force = solver.total_wall_force()
            drag_samples.append(force[0])
            lift_samples.append(force[1])
            step_numbers.append(step)
    elapsed = time.perf_counter() - start

    drag = np.asarray(drag_samples)
    lift = np.asarray(lift_samples)

    # Frontal area = diameter * depth (projected area perpendicular to flow)
    frontal_area = float(diameter * nz)
    dynamic_pressure = 0.5 * speed * speed * frontal_area

    cd = float(drag.mean() / dynamic_pressure)
    cl_amplitude = float(0.5 * (lift.max() - lift.min()) / dynamic_pressure)

    # Export force history to CSV if output_dir specified
    if output_dir is not None:
        import csv
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "cylinder_force_history.csv"
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

    # Strouhal number from lift signal FFT
    spectrum = np.abs(np.fft.rfft(lift - lift.mean()))
    frequencies = np.fft.rfftfreq(len(lift), d=4.0)
    peak = frequencies[1:][np.argmax(spectrum[1:])]
    strouhal = float(peak * diameter / speed)

    # Confined-cylinder bands from test_cylinder.py
    cd_min, cd_max = 1.35, 1.85
    st_min, st_max = 0.16, 0.21
    cl_min, cl_max = 0.15, 0.60

    passed = (cd_min <= cd <= cd_max and st_min <= strouhal <= st_max
              and cl_min <= cl_amplitude <= cl_max)

    cells = nx * ny * nz
    mlups = cells * steps / elapsed / 1e6

    return ValidationResult(
        name="Cylinder Flow (Re=100)",
        device=device,
        grid_shape=(nx, ny, nz),
        cells=cells,
        steps=steps,
        elapsed_s=elapsed,
        mlups=mlups,
        cd=cd,
        cl=cl_amplitude,
        error_metric=abs(cd - 1.59),  # Distance from measured baseline
        error_threshold=0.3,
        passed=passed,
        details={
            "diameter": diameter,
            "speed": speed,
            "tau": tau,
            "reynolds": reynolds,
            "strouhal": strouhal,
            "frontal_area": frontal_area,
            "dynamic_pressure": dynamic_pressure,
            "cd_range": [cd_min, cd_max],
            "st_range": [st_min, st_max],
            "cl_range": [cl_min, cl_max],
            "note": "Matches tests/test_cylinder.py setup; confined cylinder",
        },
    )


def run_all_validations(device: str | None = None, output_dir: Path | None = None) -> list[ValidationResult]:
    """Run all validation cases and return results."""
    results = []

    # Poiseuille (fast, exact solution)
    results.append(run_poiseuille_validation(device=device))

    # Cylinder (slower, literature comparison)
    results.append(run_cylinder_validation(device=device, output_dir=output_dir))

    return results


def save_validation_results(results: list[ValidationResult], output_dir: Path):
    """Save validation results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    data = []
    for r in results:
        data.append({
            "name": r.name,
            "device": r.device,
            "grid_shape": r.grid_shape,
            "cells": r.cells,
            "steps": r.steps,
            "elapsed_s": r.elapsed_s,
            "mlups": r.mlups,
            "cd": r.cd,
            "cl": r.cl,
            "error_metric": r.error_metric,
            "error_threshold": r.error_threshold,
            "passed": bool(r.passed),
            "details": r.details,
        })

    output_file = output_dir / "validation_results.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    return output_file


def print_validation_summary(results: list[ValidationResult]):
    """Print a summary of validation results."""
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n{r.name}: {status}")
        print(f"  Grid: {r.grid_shape} ({r.cells:,} cells)")
        print(f"  Steps: {r.steps:,}")
        print(f"  Time: {r.elapsed_s:.2f}s")
        print(f"  Performance: {r.mlups:.1f} MLUPs")

        if r.cd is not None:
            print(f"  Cd: {r.cd:.3f}")
        if r.cl is not None:
            print(f"  Cl amplitude: {r.cl:.3f}")
        if r.details and "strouhal" in r.details:
            print(f"  Strouhal: {r.details['strouhal']:.4f}")

        if r.error_metric is not None:
            print(f"  Error: {r.error_metric:.4f} (threshold: {r.error_threshold})")

        if r.details:
            print("  Details:")
            for key, value in r.details.items():
                print(f"    {key}: {value}")

    print("\n" + "=" * 80)
