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
- **Preset library** — Five ready-made experiments (cylinder wake, channel flow, cavity, backward-facing step, high-Re wake) — open from File → Open Preset
- **Probes** — Click to place velocity/pressure probes with live time-series plots
- **Analysis panel** — Reynolds number, Strouhal number, drag coefficient, field statistics — all updating live
- **Colormaps** — Toggle between smoke, speed, vorticity, and pressure views
- **Parameter sweep** — Run viscosity, inflow, or obstacle parameter sweeps — results plotted inline (File → Sweep)
- **Export** — High-res PNG with colorbar, MP4/GIF recording, probe CSV, field snapshots (.npz)
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
│   ├── scene_panel.py      Scene tree + properties
│   └── analysis_panel.py   Probes, plots, physics readouts
└── dialogs/
    ├── export_dialog.py    Image/video/data export
    ├── presets_dialog.py   Preset selection
    └── sweep_dialog.py     Parameter sweep config
scene/                      Scene system
├── scene.py                Scene dataclass + geometry types
├── serializer.py           JSON save/load
└── probe.py                Measurement probe with history
analysis/                   Physics analysis
├── physics.py              Re, drag, Strouhal
└── sweep.py                Parameter sweep runner
export/                     Export
├── image.py                High-res PNG with colorbar
├── video.py                MP4 / GIF recording
└── data.py                 Probe CSV, field snapshots
presets/                    Preset library
├── loader.py               Discover + load presets
└── scenes/                 JSON scene files + thumbnails
tests/                      100+ tests
```

## LBM Method

| Dimension | Lattice | Velocities | Collision |
|-----------|---------|------------|-----------|
| 2D | D2Q9 | 9 | BGK |
| 3D | D3Q19 | 19 | BGK |

Step order: streaming → obstacles (bounce-back) → inflow → outflow → walls → collision → emitters → advect smoke → diffuse smoke → clear obstacle smoke → decay

## Testing

```bash
pytest                  # 100 tests
pytest -v               # verbose
pytest tests/test_lbm.py::TestBoundaries   # focused
```

## License

MIT
