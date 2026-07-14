# S-Stream — Fluid Workbench

[![CI](https://github.com/UtkarshJoshiNtl/S-Stream/actions/workflows/ci.yml/badge.svg)](https://github.com/UtkarshJoshiNtl/S-Stream/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/s-stream)](https://pypi.org/project/s-stream/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

2D/3D Lattice Boltzmann fluid simulation workbench with PySide6 UI.
Designed for engineers, students, and educators who need a fluid dynamics answer in under two minutes.

## Quick Start

### Install from PyPI

```bash
pip install s-stream
s-stream
```

### Install from source

```bash
git clone https://github.com/UtkarshJoshiNtl/S-Stream.git
cd S-Stream
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -e ".[dev]"
python main.py
```

### CLI flags

| Flag | What it does |
|------|-------------|
| `--liquid` | Shan-Chen liquid/vapor multiphase engine |
| `--multicomponent` | Two immiscible fluids (oil-water separation) |
| `--gpu` | CuPy GPU acceleration |
| `--headless` | No GUI (for benchmarks/scripts) |
| `--width N --height N` | Custom grid size |
| `--steps N` | Run N steps then exit (with --headless) |
| `--serve --port 8080` | REST API server mode |

## Features

- **Live simulation** — Real-time 2D/3D LBM at 30+ fps on a modern laptop
- **6 engines** — D2Q9 CPU, D3Q19 3D CPU, CuPy GPU, PyTorch/Lettuce GPU, Shan-Chen liquid, multi-component
- **5 collision operators** — BGK, TRT, MRT, Smagorinsky SGS, WALE turbulence model
- **3 non-Newtonian models** — Power-law, Carreau, Bingham
- **Thermal LBM** — Boussinesq buoyancy for natural convection
- **Scene system** — Save/load your setup as JSON (File → Save / Open)
- **Guided Setup Wizard** — 10 templates to get started in one click (Ctrl+W)
- **Interactive geometry** — Click+drag circles, rectangles, ellipses, polygons, airfoils, channels, porous lattices
- **STL import** — Load 3D mesh files as obstacle masks
- **Image-to-obstacle** — PNG/BMP → obstacle mask
- **Preset library** — 10 ready-made experiments
- **Probes** — Click to place velocity/pressure probes with live time-series plots
- **Analysis panel** — Reynolds number, Strouhal number, drag coefficient, field statistics — all updating live
- **Flow regime detection** — Automatic classification (laminar, transitional, turbulent, vortex shedding)
- **Sanity checks** — Warnings for low viscosity, fast inflow, missing obstacles
- **Design scorecard** — Drag, wake strength, pressure drop with textual summary
- **Colormaps** — 10 fields (speed, smoke, vorticity, pressure, density, phase, temperature, component1, component2, color)
- **Overlays** — Velocity quiver, streamlines, pressure contours, force arrows, particle trails
- **Parameter sweep** — Run parameter studies with inline pyqtgraph plots
- **Particle tracer** — Lagrangian advection with RK2 integration and trail rendering
- **Export** — High-res PNG with annotations, Markdown reports, MP4/GIF recording, probe CSV, field snapshots (.npz)
- **Recipes** — Guided flow story workflows
- **Jupyter** — `plot_velocity()`, `plot_pressure()`, inline display in notebooks
- **REST API** — 11 endpoints via `python main.py --serve`
- **Multi-platform** — Linux, Windows, macOS (PyInstaller binaries in GitHub Releases)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| S | Step once |
| R | Reset simulation |
| Escape | Quit |
| Ctrl+S | Save scene |
| Ctrl+O | Open scene |
| Ctrl+E | Export dialog |
| Ctrl+W | Setup Wizard |
| 1-5 | Toggle overlays |

## System Requirements

- Python 3.10+
- OpenGL 3.3+ (for viewport rendering)
- A display server (X11/Wayland on Linux, native on Windows/macOS)
- Optional: CUDA-capable GPU for `--gpu` mode
- Optional: `pip install s-stream[stl]` for STL mesh import

## Project Structure

```
main.py                     Entry point
engines/                    Simulation backends (plug in via SimEngine ABC)
├── base.py                 SimEngine abstract base class
├── lbm_common.py           Shared lattice definitions (D2Q9, D3Q19)
├── lbm2d.py                D2Q9 CPU engine
├── lbm3d.py                D3Q19 3D CPU engine with thermal buoyancy
├── lbm2d_gpu.py            D2Q9 GPU engine (CuPy)
├── lbm2d_lettuce.py        D2Q9 PyTorch GPU engine (Lettuce)
├── lbm2d_liquid.py         D2Q9 Shan-Chen multiphase liquid engine
├── lbm2d_multicomponent.py D2Q9 two-component Shan-Chen engine
├── collision.py            BGK, TRT, MRT, Smagorinsky, WALE operators
├── non_newtonian.py        Power-law, Carreau, Bingham rheology
├── thermal_mixin.py        Boussinesq buoyancy mixin
├── smoke_mixin.py          Smoke advection/diffusion/decay mixin
├── particle_tracer.py      Lagrangian particle tracer (RK2)
├── boundary_conditions.py  Boundary condition module
└── api.py                  FastAPI REST server
workbench/                  UI layer (engine-agnostic, talks only to SimEngine)
├── app.py                  MainWindow + docking + menus + toolbar
├── viewport.py             QOpenGLWidget rendering (smoke/velocity/vorticity)
├── panels/
│   ├── scene_panel.py      Scene tree + properties + engine-specific params
│   ├── analysis_panel.py   Probes, plots, physics readouts
│   └── outcome_panel.py    Flow regime, warnings, scorecard
└── dialogs/
    ├── wizard_dialog.py    Guided Setup Wizard (10 templates)
    ├── export_dialog.py    Image/video/data export
    ├── presets_dialog.py   Preset gallery with metadata
    ├── recipes_dialog.py   Guided flow story workflows
    └── sweep_dialog.py     Parameter sweep config + inline plots
scene/                      Scene system (serializable)
├── scene.py                Scene dataclass + 7 geometry types
├── serializer.py           JSON save/load with backward compatibility
└── probe.py                Measurement probe with rolling history
analysis/                   Physics analysis
├── physics.py              Re, drag coefficient, Strouhal, pressure drop
├── regimes.py              Flow regime classification (7 regimes)
├── sanity.py               Simulation sanity checks
├── scorecard.py            Design scorecard computation
├── ai_context.py           AI prompt context builder
└── sweep.py                Parameter sweep runner
export/                     Export
├── image.py                High-res PNG with annotations + colorbar
├── video.py                MP4 / GIF recording via imageio
├── data.py                 Probe CSV, field snapshots (.npz)
└── report.py               Markdown report generation
sstream/                    Package entry point (for `import sstream`)
presets/                    Preset library
├── loader.py               Discover + load presets
└── scenes/                 JSON scene files + thumbnails (10 presets)
resources/                  UI resources (colormaps, theme, icon)
tests/                      273+ tests (pytest)
```

## LBM Method

| Dimension | Lattice | Velocities | Collision | Engines |
|-----------|---------|------------|-----------|---------|
| 2D | D2Q9 | 9 | BGK, TRT, MRT, Smagorinsky, WALE | CPU, GPU (CuPy), Lettuce (PyTorch) |
| 3D | D3Q19 | 19 | BGK | CPU |

## Physics Models

### Liquid Multiphase (Shan-Chen)

The `--liquid` flag enables the Shan-Chen pseudopotential model:
- Inter-particle force `F = −g·ψ·Σwᵢ·ψ(ρ(x+eᵢ))·eᵢ` with `g < 0` for cohesion
- Spontaneous liquid/vapor phase separation with surface tension
- `g_adhesion < 0` = wetting, `g_adhesion > 0` = non-wetting at walls
- Closed domain (bounce-back on all 4 sides); no inflow/outflow

### Multi-Component (Oil-Water)

The `--multicomponent` flag enables two immiscible fluid species:
- Intra-component cohesion (`g11`, `g22`) and inter-component repulsion (`g12`)
- Color gradient perturbation (`sigma`) for sharp interface tracking
- Fields: component1, component2, color, phase
- Useful for studying phase separation, emulsions, and miscibility

### Non-Newtonian Rheology

- **Power-law**: `ν = ν₀ · γ̇^(n−1)` — shear-thinning (n<1) or thickening (n>1)
- **Carreau**: Realistic polymer behavior with zero-shear and infinite-shear plateaus
- **Bingham**: Yield-stress fluid — solid below threshold, flows above

## Testing

```bash
pytest                  # 273+ tests
pytest -v               # verbose
pytest -m "not slow"    # skip slow tests
pytest tests/test_lbm.py::TestBoundaries   # focused
```

## Lint & Format

```bash
ruff check .
black --check .
```

## Building

### Linux

```bash
pyinstaller --onefile \
    --name "S-Stream" --windowed \
    --add-data "presets/scenes:presets/scenes" \
    --add-data "resources:resources" \
    --hidden-import PySide6.QtOpenGL \
    --hidden-import PySide6.QtOpenGLWidgets \
    --hidden-import OpenGL --hidden-import numba \
    --exclude cupy --exclude torch \
    main.py
# Output: dist/S-Stream
```

### Windows

```bat
pyinstaller --onefile ^
    --name "S-Stream" --windowed ^
    --add-data "presets/scenes;presets/scenes" ^
    --add-data "resources;resources" ^
    --hidden-import PySide6.QtOpenGL ^
    --hidden-import PySide6.QtOpenGLWidgets ^
    --hidden-import OpenGL --hidden-import numba ^
    --exclude cupy --exclude torch ^
    --icon resources\icon.ico ^
    main.py
:: Output: dist\S-Stream.exe
```

### macOS

```bash
pyinstaller --onefile \
    --name "S-Stream" --windowed \
    --add-data "presets/scenes:presets/scenes" \
    --add-data "resources:resources" \
    --hidden-import PySide6.QtOpenGL \
    --hidden-import PySide6.QtOpenGLWidgets \
    --hidden-import OpenGL --hidden-import numba \
    --exclude cupy --exclude torch \
    main.py
# Output: dist/S-Stream
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies (`pip install -e ".[dev]"`)
4. Make changes and ensure tests pass (`pytest`) and lint passes (`ruff check . && black --check .`)
5. Commit and push (`git push origin feature/my-feature`)
6. Open a Pull Request

## Future Features

| Feature | Complexity | Status |
|---------|-----------|--------|
| Free surface flow (VOF/level-set) | High | Planned |
| Phase change (latent heat + thermal) | High | Planned |
| Immersed boundary method | High | Planned |
| Adaptive mesh refinement | High | Planned |
| Fluid-structure interaction | Very high | Planned |
| Adjoint LBM shape optimization | Very high | Planned |
| 3D volume rendering viewport | High | Planned |
| TRT/MRT 3D support | Medium | Planned |
| Zou-He boundary conditions | Medium | Planned |
| D3Q27 lattice | Low | Planned |

## License

MIT — see [LICENSE](LICENSE) for details.
