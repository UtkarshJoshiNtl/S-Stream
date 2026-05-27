# CuFloda — Fluid Dynamics Simulation

2D/3D Lattice Boltzmann fluid simulation (D2Q9 / D3Q19 + BGK) with
PyGame and OpenGL visualization, plus optional GPU acceleration.

```bash
pip install -r requirements.txt && python main.py                # 2D
python main.py --3d                                              # 3D
python main.py --3d --gpu                                        # 3D + GPU
python main.py --headless --steps 5000 --width 256 --height 256  # benchmark
```

## Why

LBM is a mesoscopic CFD method well-suited for parallel computation.
This project exists to provide an interactive, visual introduction to LBM
in 2D and 3D, with a clean Python codebase that separates physics engines,
visualization, and CLI entry point. The GPU backend (CuPy) demonstrates how
the same NumPy-style code maps to CUDA with minimal changes.

## Quick Start

```bash
git clone https://github.com/UtkarshJoshiNtl/CuFloda.git
cd CuFloda
pip install -r requirements.txt
python main.py              # 2D default (128×128)
python main.py --3d         # 3D volume rendering (64³ CPU / 128³ GPU)
```

## Installation

### Prerequisites
- Python 3.10+
- OpenGL 3.3+ (GPU, for 3D visualizer)
- NVIDIA GPU + CUDA (optional, for `--gpu`)

### Dependencies
```bash
pip install -r requirements.txt
```

For GPU acceleration:
```bash
pip install cupy-cuda12x    # match your CUDA version
```

Or install the project in editable mode:
```bash
pip install -e .
```

## Usage

### 2D Simulation (D2Q9)
```bash
# Default 128×128 with PyGame visualization
python main.py

# Custom grid, headless benchmark
python main.py --width 256 --height 256 --headless --steps 5000

# Custom viscosity
python main.py --viscosity 0.01
```

### 3D Simulation (D3Q19)
```bash
# CPU — 64³ default
python main.py --3d

# GPU — 128³ default (auto-selected)
python main.py --3d --gpu

# Custom grid
python main.py --3d --width 48 --height 48 --depth 48

# 3D headless
python main.py --3d --headless --steps 5000
```

### All CLI Options
| Flag | Default | Description |
|------|---------|-------------|
| `--3d` | off | Run in 3D mode (D3Q19 lattice) |
| `--gpu` | off | Use GPU acceleration (CuPy) |
| `--width` | 128 | Grid width |
| `--height` | 128 | Grid height |
| `--depth` | auto | Grid depth (64 CPU / 128 GPU) |
| `--viscosity` | 0.02 | Fluid viscosity |
| `--headless` | off | Run without visualization |
| `--steps` | 1000 | Simulation steps (headless only) |

## Controls

### 2D Mode
| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| O | Obstacle drawing mode |
| E | Emitter placement mode |
| R | Reset simulation |
| C | Clear all emitters |
| Mouse drag | Draw obstacles |
| ESC | Quit |

### 3D Mode
| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| O | Obstacle drawing mode |
| E | Emitter placement mode |
| R | Reset simulation |
| C | Clear all emitters |
| V | Toggle volume / slice view |
| W | Scroll slice up |
| S | Scroll slice down |
| Click+drag | Orbit camera |
| Scroll | Zoom in / out |
| ESC | Quit |

## Architecture

```
main.py                     CLI entry point (routes to 2D/3D/GPU/headless)
├── cpu_lbm.py:CPULBM2D     D2Q9 engine (CPU, NumPy)
├── visualizer.py           2D PyGame renderer
├── cpu_lbm3d.py:CPULBM3D   D3Q19 engine (CPU, NumPy)
├── gpu_lbm3d.py:GPULBM3D   D3Q19 engine (GPU, CuPy, same interface)
└── visualizer3d.py         3D OpenGL volume renderer
tests/
├── test_lbm.py             38 tests (2D)
├── test_lbm3d.py           42 tests (3D)
├── test_basic.py           5 tests (legacy)
└── benchmark.py            Standalone performance benchmark
```

All three physics engines (`CPULBM2D`, `CPULBM3D`, `GPULBM3D`) share the same
interface — swap with `--gpu` to go from CPU to GPU.

### Lattice & Method

| Dimension | Lattice | Velocities | Weights | Collision |
|-----------|---------|------------|---------|-----------|
| 2D | D2Q9 | 9 | 4/9, 1/9, 1/36 | BGK |
| 3D | D3Q19 | 19 | 1/3, 1/18, 1/36 | BGK |

### Step Order (3D)
`streaming → obstacles (bounce-back) → inflow → outflow → walls → collision → emitters → advect smoke → diffuse smoke → clear obstacle smoke → decay`

This order matters: obstacles must bounce back before boundary conditions,
and smoke is cleared from obstacles **after** advection to prevent drift-through.

## API Reference

Each engine exposes:
- `step()` — advance one timestep
- `run(steps)` — advance multiple timesteps
- `initialize(rho, u, v, w)` — reset to equilibrium
- `get_density()` → `np.ndarray`
- `get_velocity()` → `np.ndarray`
- `get_smoke()` → `np.ndarray`
- `add_emitter(x, y, z, strength)` — inject smoke
- `clear_emitters()` — remove all emitters
- `add_obstacle_sphere(x, y, z, radius)` — add spherical obstacle (3D)
- `add_obstacle(x, y, radius)` — add circular obstacle (2D)
- `clear_obstacles()` — remove all obstacles

## Testing

```bash
pytest                  # all 80 tests
pytest -v               # verbose
pytest tests/test_lbm3d.py -v   # 3D only
pytest tests/test_lbm.py::TestBoundaries   # focused
```

Tests are CPU-only (no GPU required). 42 tests cover the 3D engine:
initialization, equilibrium momenta, collision invariants, streaming,
boundary conditions, obstacles, smoke advection/diffusion/decay.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Run tests (`pytest`)
4. Run lint (`ruff check .`)
5. Submit a pull request

Code style: 88-char line length, ruff selects E/F/W, flake8 ignores E203/W503.

## License

MIT
