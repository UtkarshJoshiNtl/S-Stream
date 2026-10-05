# S-Stream — See Fluid Flow in Real Time

[![CI](https://github.com/UtkarshJoshiNtl/S-Stream/actions/workflows/ci.yml/badge.svg)](https://github.com/UtkarshJoshiNtl/S-Stream/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/s-stream)](https://pypi.org/project/s-stream/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**S-Stream is an interactive 2D fluid dynamics educational tool that lets you see how fluids behave in real time.** Watch vortices shed behind a cylinder, see pressure drop across a channel, or understand how Reynolds number changes flow patterns—all in seconds, not hours.

## Who This Is For

- **Students** learning fluid mechanics who want to visualize concepts like Reynolds number, drag, and vortex shedding
- **Self-learners** exploring fluid dynamics without needing a supercomputer or commercial CFD license
- **Educators** who need a live, interactive demonstration tool for classroom teaching

**What this is not:** S-Stream is not a replacement for OpenFOAM, ANSYS Fluent, or other professional CFD suites. It's a learning and intuition-building tool, not a production engineering solver. If you need industrial-grade accuracy for aerospace or automotive design, use those tools. If you want to understand *how* fluids behave, use S-Stream.

## Why You Can Trust What You See

S-Stream is built around honesty. We don't hide behind marketing claims—we validate:

- **Verified physics:** The core 2D LBM engine (D2Q9 BGK) passes validation tests against analytical solutions (Poiseuille flow, lid-driven cavity) and experimental benchmarks (cylinder drag coefficient)
- **Sanity checks:** The app warns you when simulation parameters violate assumptions (e.g., Low-Mach number limits, relaxation rate stability)
- **Confidence badges:** Every running simulation shows a confidence score that tells you whether the flow has converged or is still developing
- **Labeled capabilities:** Features are clearly marked as **Verified**, **Experimental**, or **Hidden** in [TRUST.md](TRUST.md)—no guessing what's production-ready

See the validation suite: `pytest tests/validation/ -v`

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
| `--headless` | No GUI (for benchmarks/scripts) |
| `--width N --height N` | Custom grid size |
| `--steps N` | Run N steps then exit (with --headless) |

Advanced flags (`--liquid`, `--multicomponent`, `--gpu`, `--serve`) are available but target experimental features. See `python main.py --help` for details.

## Core Features

- **Live 2D simulation** — Watch fluids flow in real time with D2Q9 Lattice Boltzmann Method
- **Guided scenarios** — 10 built-in presets (vortex shedding, channel flow, lid-driven cavity, etc.) with explanations
- **Interactive geometry** — Draw circles, rectangles, polygons, and obstacles directly on the viewport
- **Select and delete** — Click to select obstacles and probes, press Delete to remove them
- **Physics readouts** — Real-time Reynolds number, drag coefficient, Strouhal number with tooltips explaining what they mean
- **Confidence badges** — Visual indicator showing whether the flow has converged or is still developing
- **Export for reports** — PNG figures, Markdown summaries, and CSV data for lab reports
- **Probes** — Place measurement points to track velocity, pressure, and vorticity over time
- **Timeline scrubbing** — Memory-budgeted history for reviewing past simulation states
- **Keyboard shortcuts** — Unified shortcut registry with status-bar help

## Experimental Features

<details>
<summary>Click to expand experimental features (use with caution)</summary>

The following features are available but not yet validated for production use:

- Alternative collision operators (TRT, Smagorinsky, WALE, MRT)
- Shan-Chen multiphase engines (liquid/vapor, oil-water separation)
- Non-Newtonian models (Power-law, Carreau, Bingham)
- GPU acceleration (CuPy, Lettuce)
- 3D D3Q19 engine
- Particle tracer, parameter sweep, STL/image obstacles

See [TRUST.md](TRUST.md) for detailed capability labels.
</details>

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| R | Reset simulation |
| Escape | Cancel in-progress drawing |
| Ctrl+S | Save scene |
| Ctrl+O | Open scene |
| Ctrl+E | Export dialog |
| Ctrl+W | Start gallery |

## System Requirements

- Python 3.10+
- OpenGL 3.3+ (for viewport rendering)
- A display server (X11/Wayland on Linux, native on Windows/macOS)

## How It Works

S-Stream uses the Lattice Boltzmann Method (LBM), specifically the D2Q9 model for 2D simulations. Instead of solving the Navier-Stokes equations directly, LBM simulates particle distributions on a discrete lattice that stream and collide. This approach is particularly well-suited for complex geometries and parallel computation.

The core physics (D2Q9 BGK collision) is validated against analytical solutions and experimental benchmarks. See [TRUST.md](TRUST.md) for details on what's verified vs experimental.

## Testing & Validation

```bash
pytest                                    # full suite
pytest -m "not slow"                      # skip slow tests
pytest tests/validation/ -v               # validation benchmarks
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Ensure tests pass: `pytest` and `ruff check . && black --check .`
5. Open a Pull Request

**Important:** Do not advertise Experimental features as Verified. See [TRUST.md](TRUST.md) for capability labels.

## License

MIT — see [LICENSE](LICENSE) for details.
