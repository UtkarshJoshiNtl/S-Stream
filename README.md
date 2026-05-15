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

## Prerequisites

### Required
- NVIDIA GPU with compute capability 7.0+ (RTX 20 series or newer)
- CUDA Toolkit 11.0 or higher
- Python 3.10+
- Blender 4.0+

### Install CUDA Toolkit
**Linux:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-2
```

**Windows:**
Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/UtkarshJoshiNtl/CuFloda.git
cd CuFloda
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the CUDA extension
```bash
python setup.py build_ext --inplace
```

### 4. Install Blender addon
```bash
# Copy the blender directory to Blender's addons folder
cp -r blender ~/.config/blender/4.0/scripts/addons/cufloda

# Or install from within Blender:
# Edit > Preferences > Add-ons > Install > Select blender/__init__.py
```

## Usage

### Python API
```python
import cufloda
import numpy as np

# Initialize simulation
sim = cufloda.LBM2D(width=256, height=256, viscosity=0.02)
sim.initialize(rho=1.0, u=0.1, v=0.0)

# Set obstacles
obstacles = np.zeros((256, 256), dtype=bool)
obstacles[100:150, 100:150] = True
sim.set_obstacles(obstacles)

# Run simulation
sim.run(steps=1000)

# Get results
density = sim.get_density()
velocity = sim.get_velocity()
```

### Blender Addon
1. Open Blender
2. Go to View3D > Sidebar > CuFloda
3. Set simulation parameters (width, height, viscosity)
4. Click "Initialize Simulation"
5. (Optional) Select an obstacle object and click "Set Obstacles"
6. Click "Real-time Preview" for live visualization
7. Click "Export to Particles" or "Export to Mesh" to export results

## Benchmarking

Run the benchmark script to compare GPU vs CPU performance:
```bash
python tests/benchmark.py
```

Expected results (RTX 3080):
- 64x64: ~5000 FPS GPU vs ~50 FPS CPU (100x speedup)
- 128x128: ~2000 FPS GPU vs ~10 FPS CPU (200x speedup)
- 256x256: ~500 FPS GPU vs ~2 FPS CPU (250x speedup)

## Testing

Run basic tests:
```bash
python tests/test_basic.py
```

## Troubleshooting

### CUDA not found
Ensure CUDA Toolkit is installed and in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build errors
- Ensure you have a compatible NVIDIA GPU
- Check CUDA version compatibility
- Verify compute capability of your GPU with `nvidia-smi`

### Blender addon not showing
- Check Blender console for errors
- Ensure the addon is enabled in Preferences > Add-ons
- Verify Python dependencies are installed in Blender's Python environment

## License
MIT
