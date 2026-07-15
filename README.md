# S-Stream — Fluid Workbench

[![CI](https://github.com/UtkarshJoshiNtl/S-Stream/actions/workflows/ci.yml/badge.svg)](https://github.com/UtkarshJoshiNtl/S-Stream/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/s-stream)](https://pypi.org/project/s-stream/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

2D/3D Lattice Boltzmann fluid simulation workbench with PySide6 UI.
Designed for engineers, students, and educators who need a **trustworthy** fluid
dynamics answer in under two minutes.

**Roadmap priority:** accuracy → performance → UX → features.
See [TRUST.md](TRUST.md) for capability labels
(**Verified** / **Experimental** / **Hidden**).

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
| `--liquid` | Shan-Chen liquid/vapor multiphase (**Experimental**) |
| `--multicomponent` | Two immiscible fluids (**Experimental**) |
| `--gpu` | CuPy GPU acceleration (**Experimental** feature parity) |
| `--headless` | No GUI (for benchmarks/scripts) |
| `--physics-only` | Skip smoke/particles in headless steps (honest MLUPs) |
| `--width N --height N` | Custom grid size |
| `--steps N` | Run N steps then exit (with --headless) |
| `--serve --port 8080` | REST API server mode |

## Features (honest)

### Verified / core workbench

- **Live 2D simulation** — D2Q9 BGK with bounce-back obstacles, inflow/outflow
- **Validation suite** — Poiseuille, lid-driven cavity, cylinder Cd (see TRUST.md)
- **Scene system** — JSON save/load
- **Start gallery** — Verified presets and templates
- **Interactive geometry** — Circles, rectangles, polygons, ellipses, airfoils, channels, lattices
- **Probes + analysis** — Re, St, Cd, field stats (plain-language Outcome dock)
- **Sanity checks** — Low-Mach / ω warnings
- **Export** — PNG, Markdown report, MP4/GIF, CSV, `.npz`
- **Jupyter** — `plot_velocity()`, `plot_pressure()`, inline display
- **REST API** — `python main.py --serve`

### Experimental (use with care)

- TRT / Smagorinsky / WALE / MRT collision operators (2D)
- Shan-Chen liquid and multi-component engines
- Non-Newtonian models (Power-law, Carreau, Bingham)
- CuPy and Lettuce GPU backends
- 3D D3Q19 CPU engine
- Particle tracer, parameter sweep, STL / image obstacles

### Hidden until fixed

- Product-facing thermal buoyancy UI
- Live Gemini AI tutor
- MPI / multi-GPU

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| S | Step once |
| R | Reset simulation |
| Escape | Cancel in-progress drawing |
| Ctrl+S | Save scene |
| Ctrl+O | Open scene |
| Ctrl+E | Export dialog |
| Ctrl+W | Start gallery |
| 1-5 | Toggle overlays (expert mode) |

## System Requirements

- Python 3.10+
- OpenGL 3.3+ (for viewport rendering)
- A display server (X11/Wayland on Linux, native on Windows/macOS)
- Optional: CUDA-capable GPU for `--gpu` mode
- Optional: `pip install s-stream[stl]` for STL mesh import

## Project Structure

```
main.py                     Entry point
engines/                    Simulation backends (SimEngine ABC)
workbench/                  UI (engine-agnostic)
scene/                      Serializable scenes + probes
analysis/                   Physics, regimes, sanity, scorecard, sweep
export/                     Image, video, data, report
sstream/                    Package entry (`import sstream`)
presets/                    Preset library
resources/                  Colormaps, theme, icon
tests/                      Unit + validation suites
TRUST.md                    What we claim vs what CI proves
```

Full module map: [AGENTS.md](AGENTS.md).

## LBM Method

| Dimension | Lattice | Velocities | Collision (default) | Engines |
|-----------|---------|------------|---------------------|---------|
| 2D | D2Q9 | 9 | BGK (**Verified** path) | CPU; GPU Experimental |
| 3D | D3Q19 | 19 | BGK | CPU Experimental |

Pressure is lattice `p = ρ/3` (`c_s² = 1/3`). Hydro presets target `U ≲ 0.05`.

## Testing

```bash
pytest                                    # full suite
pytest -m "not slow"                      # skip slow
pytest tests/validation/ -v               # trust benchmarks
pytest tests/test_lbm.py::TestBoundaries  # focused
python tests/benchmark.py                 # MLUPs (use physics-only)
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
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies (`pip install -e ".[dev]"`)
4. Prefer Accuracy → Performance → UX → Features
5. Ensure `pytest` and `ruff check . && black --check .` pass
6. Open a Pull Request — do not advertise Experimental models as Verified

## License

MIT — see [LICENSE](LICENSE) for details.
