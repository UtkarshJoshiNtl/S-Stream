# S-Stream — Fluid Workbench

2D Lattice Boltzmann (D2Q9 + BGK) fluid simulation workbench with PySide6 UI.
Designed for engineers and students who need a fluid dynamics answer in under two minutes.

```bash
pip install -r requirements.txt && python main.py
```

## Quick Start

| What | How |
|------|-----|
| Run the app | `python main.py` |
| Headless benchmark | `python main.py --headless --steps 5000` |
| Custom grid | `python main.py --headless --width 256 --height 256 --steps 5000` |
| Run tests | `pytest` |
| Lint | `ruff check .` |

## Features

- **Live simulation** — Real-time 2D LBM fluid at 30+ fps on a modern laptop
- **Scene system** — Save/load your setup as JSON (File → Save / Open)
- **Interactive geometry** — Click+drag circles, rectangles, or freehand polygons in the viewport
- **Preset library** — Five ready-made experiments with product metadata (cylinder wake, channel flow, cavity, backward-facing step, high-Re wake) — open from File → Open Preset
- **Probes** — Click to place velocity/pressure probes with live time-series plots
- **Analysis panel** — Reynolds number, Strouhal number, drag coefficient, field statistics — all updating live
- **Flow regime detection** — Automatic classification (laminar, transitional, turbulent, steady, vortex shedding)
- **Sanity checks** — Warnings for low viscosity, fast inflow, missing obstacles, coarse resolution
- **Design scorecard** — Drag, wake strength, pressure drop, shedding confidence with textual summary
- **Beginner/Expert mode** — Toggle to hide advanced parameters for new users
- **Colormaps** — Toggle between smoke, speed, vorticity, and pressure views
- **Parameter sweep** — Run viscosity, inflow, or obstacle parameter sweeps — results plotted inline (File → Sweep)
- **Export** — High-res PNG with annotations and colorbar, Markdown reports, MP4/GIF recording, probe CSV, field snapshots (.npz)
- **Recipes** — Guided flow story workflows for common patterns
- **AI assistant stub** — Prompt context builder for future Gemini AI integration
- **3D engines** — D3Q19 CPU/GPU engines included (frozen post-v1, revisit planned)

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
engines/                    Simulation backends (SimEngine ABC)
├── base.py                 Abstract base class
├── lbm_common.py           Lattice constants (D2Q9)
├── lbm2d.py                D2Q9 CPU engine
├── lbm3d_cpu.py            D3Q19 CPU engine (frozen, post-v1)
└── lbm3d_gpu.py            D3Q19 GPU engine (frozen, post-v1)
workbench/                  UI layer (PySide6)
├── app.py                  MainWindow + docking + menus
├── viewport.py             QOpenGLWidget rendering
├── panels/
│   ├── scene_panel.py      Scene tree + properties + beginner/expert mode
│   ├── analysis_panel.py   Probes, plots, physics readouts
│   └── outcome_panel.py    Flow regime, warnings, scorecard, AI preview
└── dialogs/
    ├── export_dialog.py    Image/video/data export
    ├── presets_dialog.py   Preset gallery with metadata
    ├── sweep_dialog.py     Parameter sweep config
    └── recipes_dialog.py   Guided flow story workflows
scene/                      Scene system
├── scene.py                Scene dataclass + geometry types + product metadata
├── serializer.py           JSON save/load with backward compatibility
└── probe.py                Measurement probe with history
analysis/                   Physics analysis
├── physics.py              Re, drag, Strouhal
├── regimes.py              Flow regime detection
├── sanity.py               Simulation sanity checks
├── scorecard.py            Design scorecard computation
├── ai_context.py           AI prompt context builder
└── sweep.py                Parameter sweep runner
export/                     Export
├── image.py                High-res PNG with annotations + colorbar
├── report.py               Markdown report generation
├── video.py                MP4 / GIF recording
└── data.py                 Probe CSV, field snapshots
presets/                    Preset library
├── loader.py               Discover + load presets with metadata
└── scenes/                 JSON scene files + thumbnails + product metadata
tests/                      115 tests
docs/                       Documentation
└── naming_pass.md          Product naming shortlist
```

## LBM Method

| Dimension | Lattice | Velocities | Collision |
|-----------|---------|------------|-----------|
| 2D | D2Q9 | 9 | BGK |
| 3D | D3Q19 | 19 | BGK |

Step order: streaming → obstacles (bounce-back) → inflow → outflow → walls → collision → emitters → advect smoke → diffuse smoke → clear obstacle smoke → decay

## Testing

```bash
pytest                  # 115 tests
pytest -v               # verbose
pytest tests/test_lbm.py::TestBoundaries   # focused
pytest tests/test_analysis.py              # analysis modules
```

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
