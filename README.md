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

## Render

Snapshots come from the solver; the renderer is a separate tool and the
solver never imports it:

```sh
sstream path/to/car.stl --preset preview --visual \
  --snapshots results/snapshots --snapshot-count 6
sstream-render results/snapshots --out drive.mp4 \
  --streamlines --smoke --vortices --model
```

Note: `--snapshot-dtype float16` (default) can inject noise into Q-criterion due to
reduced precision. Use `--snapshot-dtype float32` for quality runs where accurate
vortex visualization is required.

`--streamlines` (speed-coloured tubes, last snapshot), `--smoke`
(RK2 particles with fading trails), `--vortices` (Q-criterion isosurface,
`--q-fraction`, 40% opacity) and `--model` (the placed model) are
independent toggles; at least one is required. `--still FRAME` writes a PNG
of one frame, `--preview` renders 960x540 at every 2nd snapshot, and the
default is 1920x1080 30 fps MP4. `--tube-radius`, `--q-fraction` and
`--orbit-deg-per-sec` tune the look. `PYVISTA_OFF_SCREEN=true` is required
on headless machines (see the WSL section below).

## Tests

```sh
python -m pytest simulation/tests
```

## Debian WSL (rendering offscreen)

`sstream` itself never opens a display. The two commands that do —
`sstream-place` and `sstream-render` — use pyvista (VTK) and need OpenGL
system libraries. On a Debian WSL shell:

```sh
sudo apt update && sudo apt install -y \
  xvfb libgl1 libegl1 libgl1-mesa-dri libosmesa6 libxrender1 libxkbcommon0
```

Then set the offscreen flag and run:

```sh
export PYVISTA_OFF_SCREEN=true
sstream-render results/snapshots --out drive.mp4 --smoke --model
```

If you still get errors about a display or GL context, run under a virtual
framebuffer instead — `xvfb-run -a sstream-render …`, or start one manually
(`Xvfb :99 -screen 0 1920x1080x24 &` then `export DISPLAY=:99`). This is how
the stills and movies in this README are produced: WSL has no physical
display, so rendering is offscreen on purpose.

In CI/headless pipelines where the GL packages cannot be installed, skip the
two rendering commands and exercise the solver only; rendering will crash at
plotter creation without a GL context.

## Windows build

See `simulation/packaging/README.md` (PyInstaller bundle, one command).

## Credits

The solver's design (D3Q19 + SRT, GPU residency, fixed color scales,
Q-criterion thinking, live-loop lessons) was informed by
[ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D).
No FluidX3D source was copied; see `credit.md` for the full record and the
license compliance rules S-Stream follows.
