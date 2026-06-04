# CuFloda

2D/3D Lattice Boltzmann (D2Q9 / D3Q19 + BGK) fluid simulation with modular UI.

## Entry points

- `main.py` — CLI entry point (`python main.py` or `cufloda` after `pip install -e .`)
- `engines/lbm2d.py:LBM2D` — core 2D LBM engine (D2Q9)
- `engines/lbm3d_cpu.py:LBM3DCPU` — core 3D LBM engine (D3Q19, CPU)
- `engines/lbm3d_gpu.py:LBM3DGPU` — GPU-accelerated 3D LBM engine (CuPy, same interface)
- `engines/base.py:SimEngine` — abstract base class for all simulation backends
- `engines/lbm_common.py` — shared lattice definitions (LATTICE_2D, LATTICE_3D)
- `visualizer.py:FluidVisualizer` — 2D PyGame rendering (legacy)
- `visualizer3d.py:FluidVisualizer3D` — 3D OpenGL volume rendering (legacy)
- `ui/renderer.py` — OpenGL smoke rendering (2D fullscreen quad + 3D volume)
- `ui/widgets.py` — ImGui control panels
- `ui/app.py` — GLFW + pyimgui main loop

## Quick commands

| what | how |
|------|-----|
| install deps | `pip install -r requirements.txt` |
| run 2D visual | `python main.py` |
| run 3D visual | `python main.py --3d` |
| run 3D w/ GPU | `python main.py --3d --gpu` |
| run 3D headless | `python main.py --3d --headless --steps 5000` |
| run 3D w/ custom grid | `python main.py --3d --width 64 --height 64 --depth 64` |
| run headless | `python main.py --headless` |
| run headless w/ custom grid | `python main.py --headless --width 256 --height 256 --steps 5000` |
| all tests | `pytest` (configured in `pyproject.toml`, testpaths = ["tests"]) |
| focused test | `pytest tests/test_lbm.py::TestBoundaries` |
| 3D test suite | `pytest tests/test_lbm3d.py -v` |
| benchmark | `python tests/benchmark.py` |

## Lint & format

```
ruff check .
black --check .
flake8
```

All enforce 88-char line length. Ruff `select = ["E", "F", "W"]`, flake8 ignores E203/W503.

## Architecture notes

### Module structure
```
engines/           # Simulation backends (plug in via SimEngine ABC)
├── base.py        # SimEngine abstract base class
├── lbm_common.py  # Shared lattice definitions (LATTICE_2D, LATTICE_3D)
├── lbm2d.py       # D2Q9 CPU engine
├── lbm3d_cpu.py   # D3Q19 CPU engine
└── lbm3d_gpu.py   # D3Q19 GPU engine (CuPy)
ui/                # UI layer (engine-agnostic, talks only to SimEngine)
├── renderer.py    # OpenGL smoke rendering
├── widgets.py     # ImGui control panels
└── app.py         # GLFW + pyimgui main loop
```

### 2D (D2Q9)
- `LBM2D.step()` calls: streaming → obstacles → inflow → outflow → walls → collision → emitters → advect smoke → diffuse smoke → decay smoke (order matters)
- Obstacles use bounce-back BC; inflow sets equilibrium at left column; outflow copies last column; walls bounce-back top/bottom
- Smoke is a passive scalar advected via bilinear interpolation, with velocity zeroed inside obstacles to prevent drift-through
- Diffusion (`smoke_diffusion=0.05`) and decay (`smoke_decay=0.999`)
- `C:clear_emitters` calls `sim.clear_emitters()` (list-based, not grid-based)

### 3D (D3Q19)
- `LBM3DCPU` / `LBM3DGPU` share the same SimEngine interface — swap via `--gpu` flag
- Default depth: 64 for CPU, 128 for GPU (auto-selected)
- Lattice: D3Q19 (19 velocity directions), weights `1/3`, `1/18`, `1/36`
- Smoke advection uses trilinear interpolation; diffusion uses boundary-safe 6-neighbor Laplacian
- Obstacle smoke cleared after advection (not before) to prevent drift-through
- `add_obstacle(x, y, z, radius)` for spherical obstacles (unified interface with 2D)

### 3D Visualizer (legacy, being replaced)
- OpenGL 3.3 core profile volume renderer (ray-marching fragment shader)
- Two view modes: volume rendering (V key to toggle) and 2D slice view (W/S to scroll)
- Camera: spherical orbit (mouse drag), zoom (scroll wheel)
- HUD overlay rendered via separate orthographic shader pass

## Testing quirks

- `test_lbm.py` uses class-based pytest organization (`TestInit`, `TestCollision`, ...)
- `test_lbm3d.py` mirrors the same structure for 3D (42 tests)
- `test_basic.py` has plain test functions (older tests)
- `benchmark.py` is a standalone script, not collected by pytest
- All tests are CPU-only, no GPU required

## Controls (legacy visualizers)

### 2D mode
Space=pause, O=obstacle mode, E=emitter mode, R=reset, C=clear emitters, ESC=quit

### 3D mode
Space=pause, O=obstacle mode, E=emitter mode, R=reset, C=clear emitters, V=toggle view,
W=slice up, S=slice down, drag=rotate, scroll=zoom, ESC=quit
