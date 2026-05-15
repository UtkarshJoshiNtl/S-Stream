# CuFloda - CUDA-Accelerated Fluid Dynamics for Blender

GPU-accelerated fluid simulation plugin for Blender using Lattice Boltzmann Method (LBM).

## Goals
- 10-50× speedup over Blender's CPU-based fluid simulation
- Real-time preview at ≥30 FPS for 128³ resolution grids
- Interactive simulation with obstacle interaction
- Blender plugin interface for parameter control

## Technical Approach
- **Method**: Lattice Boltzmann Method (LBM)
- **GPU**: CUDA with custom Python bindings (pybind11)
- **Lattice**: D2Q9 for 2D, D3Q19 for 3D
- **Target**: sm_70+ (RTX 20 series and newer)
- **Blender**: 4.0+

## Project Structure
```
CuFloda/
├── cuda/           # CUDA kernels
├── python/          # Python bindings
├── blender/         # Blender addon
├── tests/           # Tests and benchmarks
└── docs/            # Documentation
```

## Installation
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Usage
```python
import bpy
import cufloda

# Initialize simulation
sim = cufloda.LBM2D(width=256, height=256)
sim.set_obstacle(obstacle_mask)
sim.run(steps=1000)
```

## License
MIT
