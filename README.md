# S-Stream

A headless GPU wind-tunnel solver. Import a car, a bull, or a person; drop
it into the tunnel; solve the flow; read the numbers and a report. No GUI:
the solver prints progress and writes files. The target platform is Windows
(the solver is developed on Linux and must run on Windows).

Physics: D3Q19 lattice Boltzmann, half-way bounce-back, Guo forcing,
Smagorinsky-Lilly LES. Python + [Warp](https://nvidia.github.io/warp/)
kernels on CUDA. Validated on channel flow, cylinder wake, and sphere drag
(see `simulation/tests/`).

## Install

```sh
python -m pip install -e ./simulation
```

Requires: Python 3.10+, CUDA-capable GPU, ~4 GB VRAM for preview preset.

## Quickstart

### Validate the solver

```sh
python -m sstream.cli --validate --out validation-results
```

Runs Poiseuille flow (exact solution) and cylinder wake (literature comparison).

### Run a simulation

```sh
python -m sstream.cli path/to/model.stl --preset preview --out results
```

Outputs under `results/`:
- `results.json` — Cd, Cl, Reynolds, timings, convergence metrics
- `report.txt` — plain-text summary with honest limits
- `snapshots/` — velocity fields as `.npz` (if `--snapshots` specified)

### View a snapshot

```sh
python scripts/slice_view.py results/snapshots wake.png --field vorticity
```

Generates a 2D midplane slice PNG showing vorticity or speed magnitude.

## Validation

| Case | Metric | Measured | Reference | Status |
|------|--------|----------|-----------|--------|
| Poiseuille flow | Max velocity error | 0.38% | Exact solution | ✅ Pass |
| Cylinder wake (Re=100) | Cd | 1.587 | 1.3–1.6 (confined) | ✅ Pass |
| Cylinder wake (Re=100) | St | 0.184 | 0.16–0.21 (confined) | ✅ Pass |
| Sphere drag (Re=100) | Cd | 1.20 | 1.087 (Johnson & Patel 1999) | ⚠️ +10% |
| Sphere drag (Re=200) | Cd | 0.89 | ~0.78 (literature) | ⚠️ +14% |
| Sphere drag (Re=300) | Cd | 0.77 | ~0.65 (literature) | ⚠️ +18% |

**Notes:**
- Cylinder values are for confined geometry (13% blockage); unconfined literature Cd ≈ 1.3, St ≈ 0.166.
- Sphere Cd is systematically high due to tunnel wall effects and grid resolution; see `docs/DIAGNOSIS_SPHERE.md` for analysis.
- All validation uses low-Re flows (Re ≤ 300); high-Re Ahmed/DrivAer cases are not yet validated.

## Snapshots

Velocity snapshots are saved during the solve for later analysis:

```sh
sstream path/to/car.stl --preset preview \
  --snapshots results/snapshots --snapshot-count 6
```

Each snapshot is a `.npz` file containing:
- `velocity` — float16 array `(nx, ny, nz, 3)` in lattice units
- `density` — float16 array `(nx, ny, nz)` (optional)
- `step` — integer step number

`meta.json` contains grid spacing, tunnel dimensions, and placement info.
`solid.npy` contains the boolean solid mask.

**Dtype:** `--snapshot-dtype float16` (default, ~50% smaller) or `float32` (higher precision).

## Tests

```sh
python -m pytest simulation/tests
```

101 tests covering: lattice units, equilibrium, boundary conditions, force calculation, dye advection, voxelization, mesh loading, pipeline integration, and validation cases.

Slow GPU tests (marked `@pytest.mark.slow`):
- `test_sphere_re100.py` — sphere drag at Re=100
- `test_outlet_reflection.py` — outlet boundary sensitivity

## Diagnostics

Development-time diagnostic scripts are in `simulation/scripts/diagnostics/`. These were used during solver development to investigate specific issues (geometry audit, force balance, tunnel drift, resolution sweeps). See `simulation/scripts/diagnostics/README.md` for details.

## Limitations

- **Low-Re validation only:** Ahmed body and DrivAer cases are not yet validated against literature.
- **Grid resolution:** Sphere drag is ~10-18% high due to coarse grid and wall effects.
- **No visual mode calibration:** The solver does not claim quantitative accuracy for vehicle-scale flows.
- **Windows target:** Developed on Linux, must run on Windows (PyInstaller bundle in `simulation/packaging/`).

## Lessons Learned

See `docs/LEARNINGS.md` for a development journal covering:
- Bug fixes (tau calculation, outlet boundary conditions, force sampling)
- Performance tuning (occupancy, block dimensions — null results)
- Validation methodology (blockage correction, reference selection)

## Windows build

See `simulation/packaging/README.md` (PyInstaller bundle, one command).

## Credits

The solver's design (D3Q19 + SRT, GPU residency, fixed color scales,
Q-criterion thinking, live-loop lessons) was informed by
[ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D).
No FluidX3D source was copied; see `credit.md` for the full record and the
license compliance rules S-Stream follows.
