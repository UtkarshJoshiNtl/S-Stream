# S-Stream — Fluid Workbench

2D Lattice Boltzmann (D2Q9 + BGK) fluid simulation workbench with PySide6 UI.
Designed for engineers, students, and educators who need a fluid dynamics answer in under two minutes.

```bash
pip install -r requirements.txt && python main.py
```

## Quick Start

| What | How |
|------|-----|
| Run the app | `python main.py` |
| Liquid multiphase | `python main.py --liquid` |
| Headless benchmark | `python main.py --headless --steps 5000` |
| Custom grid | `python main.py --headless --width 256 --height 256 --steps 5000` |
| Run tests | `pytest` |
| Lint | `ruff check .` |

## Features

- **Live simulation** — Real-time 2D LBM fluid at 30+ fps on a modern laptop
- **Scene system** — Save/load your setup as JSON (File → Save / Open)
- **Interactive geometry** — Click+drag circles, rectangles, or freehand polygons in the viewport
- **Preset library** — Nine ready-made experiments (cylinder wake, channel flow, cavity, backward-facing step, bluff body drag, porous screen, two-cylinder interference, nozzle diffuser, high-Re wake)
- **Probes** — Click to place velocity/pressure probes with live time-series plots
- **Analysis panel** — Reynolds number, Strouhal number, drag coefficient, field statistics — all updating live
- **Flow regime detection** — Automatic classification (laminar, transitional, turbulent, vortex shedding, etc.)
- **Sanity checks** — Warnings for low viscosity, fast inflow, missing obstacles, coarse resolution
- **Design scorecard** — Drag, wake strength, pressure drop, shedding confidence with textual summary
- **Beginner/Expert mode** — Toggle to hide advanced parameters for new users
- **Colormaps** — Cycle through smoke, speed, vorticity, pressure, density, and phase views
- **Parameter sweep** — Run viscosity, inflow, or obstacle parameter sweeps — results plotted inline
- **Export** — High-res PNG with annotations and colorbar, Markdown reports, MP4/GIF recording, probe CSV, field snapshots (.npz)
- **Recipes** — Guided flow story workflows for common patterns
- **Liquid multiphase** — Shan-Chen pseudopotential model for liquid/vapor phase separation with surface tension (`--liquid`)

## Controls

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| R | Reset simulation |
| Escape | Quit |
| Ctrl+S | Save scene |
| Ctrl+O | Open scene |
| Ctrl+E | Export dialog |
| Ctrl+Shift+S | Parameter sweep |

## Project Structure

```
main.py                     Entry point
engines/                    Simulation backends (plug in via SimEngine ABC)
├── base.py                 SimEngine abstract base class
├── lbm_common.py           Shared lattice definitions (D2Q9)
├── lbm2d.py                D2Q9 CPU engine (BGK collision)
├── lbm2d_liquid.py         D2Q9 Shan-Chen multiphase liquid engine
└── lbm2d_gpu.py            D2Q9 GPU engine (CuPy)
workbench/                  UI layer (engine-agnostic, talks only to SimEngine)
├── app.py                  MainWindow + docking + menus + toolbar
├── viewport.py             QOpenGLWidget rendering (smoke/velocity/vorticity)
├── panels/
│   ├── scene_panel.py      Scene tree + properties + beginner/expert mode
│   ├── analysis_panel.py   Probes, plots, physics readouts
│   └── outcome_panel.py    Flow regime, warnings, scorecard, AI preview
└── dialogs/
    ├── export_dialog.py    Image/video/data export
    ├── presets_dialog.py   Preset gallery with metadata
    ├── recipes_dialog.py   Guided flow story workflows
    └── sweep_dialog.py     Parameter sweep config + inline plots
scene/                      Scene system (serializable)
├── scene.py                Scene dataclass + geometry types + product metadata
├── serializer.py           JSON save/load with backward compatibility
└── probe.py                Measurement probe with rolling history
analysis/                   Physics analysis
├── physics.py              Re, drag coefficient, Strouhal, pressure drop
├── regimes.py              Flow regime classification (7 regimes)
├── sanity.py               Simulation sanity checks (6 checks)
├── scorecard.py            Design scorecard computation
├── ai_context.py           AI prompt context builder (Gemini integration stub)
└── sweep.py                Parameter sweep runner (QThread-based)
export/                     Export
├── image.py                High-res PNG with annotations + colorbar
├── video.py                MP4 / GIF recording via imageio
├── data.py                 Probe CSV, field snapshots (.npz)
└── report.py               Markdown report generation
presets/                    Preset library
├── loader.py               Discover + load presets with metadata
└── scenes/                 JSON scene files + thumbnails (9 presets)
tests/                      101 tests (pytest)
├── test_basic.py           Basic simulation tests
├── test_lbm.py             Class-based LBM2D tests
├── test_liquid.py          Shan-Chen liquid engine tests
├── test_analysis.py        Analysis module tests
├── test_physics.py         Physics computation tests
├── test_scene.py           Scene system tests
└── benchmark.py            Standalone benchmark (not pytest-collected)
docs/
└── naming_pass.md          Product naming shortlist
```

## LBM Method

| Dimension | Lattice | Velocities | Collision | Engine |
|-----------|---------|------------|-----------|--------|
| 2D | D2Q9 | 9 | BGK | CPU / GPU / Liquid (Shan-Chen) |

Step order: streaming → obstacles (bounce-back) → inflow → outflow → walls → collision → emitters → advect smoke → diffuse smoke → clear obstacle smoke → decay

## Liquid Multiphase (Shan-Chen)

The `--liquid` flag enables the Shan-Chen pseudopotential model:
- Inter-particle force `F = −g·ψ·Σwᵢ·ψ(ρ(x+eᵢ))·eᵢ` with `g < 0` for cohesion
- Spontaneous liquid/vapor phase separation with surface tension
- `g_adhesion < 0` = wetting, `g_adhesion > 0` = non-wetting at walls
- Closed domain (bounce-back on all 4 sides); no inflow/outflow

## Testing

```bash
pytest                  # 101 tests
pytest -v               # verbose
pytest tests/test_lbm.py::TestBoundaries   # focused
pytest tests/test_liquid.py -v             # liquid engine tests
```

## Lint & Format

```bash
ruff check .
black --check .
flake8
```

All enforce 88-char line length.

## Building

```bash
# Build using PyInstaller spec file (recommended)
pyinstaller sstream.spec              # Build as directory
pyinstaller sstream.spec --onefile    # Build as single executable

# Or use the convenience script
python scripts/build.py                # Build for current platform
python scripts/build.py --onefile      # Single executable
```

## License

MIT
