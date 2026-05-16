# CuFloda - Fluid Dynamics Simulation

Fluid dynamics simulation using Lattice Boltzmann Method (LBM) with CPU prototype and future CUDA acceleration.

## Development Phases

**Current Phase: CPU Prototype with PyGame Visualization**
- D2Q9 Lattice Boltzmann Method implementation
- Real-time visualization with PyGame
- Complex boundary conditions (inflow, outflow, walls, obstacles)
- Target: 128x128 resolution at 30+ FPS

**Future Phases:**
- Advanced visualization (volume rendering, lighting, fire coloring)
- CUDA port with SOA (Structure of Arrays) memory layout
- Performance optimization and benchmarking

## Project Structure
```
CuFloda/
├── cpu_lbm.py           # CPU D2Q9 implementation
├── visualizer.py        # PyGame visualization
├── main.py              # Entry point
├── tests/
│   └── benchmark.py     # CPU benchmark
└── README.md
```

## Prerequisites

### Required
- Python 3.10+
- NumPy
- PyGame

## Installation

```bash
# Clone the repository
git clone https://github.com/UtkarshJoshiNtl/CuFloda.git
cd CuFloda

# Install dependencies
pip install numpy pygame
```

## Usage

Run the simulation with visualization:
```bash
python main.py
```

Controls:
- Space: Pause/Resume
- R: Reset simulation
- ESC: Quit
- Mouse: Draw obstacles

## Technical Approach

**Method:** Lattice Boltzmann Method (LBM)
**Lattice:** D2Q9 for 2D simulations
**Collision:** BGK (Bhatnagar-Gross-Krook)
**Boundaries:** Complex inflow/outflow, bounce-back for obstacles

## License
MIT
