# S-Stream

A headless GPU wind-tunnel solver. Import a car, a bull, or a person; drop
it into the tunnel; solve the flow; read the numbers and a report. No GUI:
the solver prints progress and writes files. The target platform is Windows
(the solver is developed on Linux and must run on Windows).

Physics: D3Q19 lattice Boltzmann, half-way bounce-back, Guo forcing,
Smagorinsky-Lilly LES. Python + [Warp](https://nvidia.github.io/warp/)
kernels on CUDA. Validated on channel flow and cylinder wake
(see `simulation/tests/`).

## Install

```sh
python -m pip install -e ./simulation
```

## Run

```sh
python -m sstream.cli path\to\model.stl --preset preview --out results
sstream --help        # after install, the console script
```

Outputs under `results/`: `results.json` (Cd, Cl, Reynolds, timings) and
`report.txt` (plain-text summary with honest limits). `--save-fields`
additionally writes `fields.npz`.

## Snapshots

Velocity snapshots can be saved during the solve for later analysis:

```sh
sstream path/to/car.stl --preset preview \
  --snapshots results/snapshots --snapshot-count 6
```

Note: `--snapshot-dtype float16` (default) uses less disk space, while
`--snapshot-dtype float32` provides higher precision for analysis.

## Tests

```sh
python -m pytest simulation/tests
```

## Windows build

See `simulation/packaging/README.md` (PyInstaller bundle, one command).

## Credits

The solver's design (D3Q19 + SRT, GPU residency, fixed color scales,
Q-criterion thinking, live-loop lessons) was informed by
[ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D).
No FluidX3D source was copied; see `credit.md` for the full record and the
license compliance rules S-Stream follows.
