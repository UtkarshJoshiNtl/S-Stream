# CuFloda

2D/3D Lattice Boltzmann (D2Q9 / D3Q19 + BGK) fluid simulation with PyGame visualization.

## Entry points

- `main.py` — CLI entry point (`python main.py` or `cufloda` after `pip install -e .`)
- `cpu_lbm.py:CPULBM2D` — core 2D LBM engine (D2Q9)
- `visualizer.py:FludVisualizer` — 2D PyGame rendering
- `cpu_lbm3d.py:CPULBM3D` — core 3D LBM engine (D3Q19)
- `gpu_lbm3d.py:GPULBM3D` — GPU-accelerated 3D LBM engine (CuPy, same interface)
- `visualizer3d.py:FluidVisualizer3D` — 3D OpenGL volume rendering

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

### 2D (D2Q9)
- `CPULBM2D.step()` calls: streaming → obstacles → inflow → outflow → walls → collision → emitters → advect smoke → diffuse smoke → decay smoke (order matters)
- Obstacles use bounce-back BC; inflow sets equilibrium at left column; outflow copies last column; walls bounce-back top/bottom
- Smoke is a passive scalar advected via bilinear interpolation, with diffusion (`smoke_diffusion=0.05`) and decay (`smoke_decay=0.999`)
- `C:clear_emitters` in visualizer calls `sim.clear_emitters()` (list-based, not grid-based)

### 3D (D3Q19)
- `CPULBM3D` / `GPULBM3D` share the same interface — swap via `--gpu` flag
- Default depth: 64 for CPU, 128 for GPU (auto-selected)
- Lattice: D3Q19 (19 velocity directions), weights `1/3`, `1/18`, `1/36`
- Smoke advection uses trilinear interpolation; diffusion uses boundary-safe 6-neighbor Laplacian
- Obstacle smoke cleared after advection (not before) to prevent drift-through
- `add_obstacle_sphere(x, y, z, radius)` for spherical obstacles

### 3D Visualizer
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

## Controls

### 2D mode
Space=pause, O=obstacle mode, E=emitter mode, R=reset, C=clear emitters, ESC=quit

### 3D mode
Space=pause, O=obstacle mode, E=emitter mode, R=reset, C=clear emitters, V=toggle view,
W=slice up, S=slice down, drag=rotate, scroll=zoom, ESC=quit
