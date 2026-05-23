# CuFloda - Fluid Dynamics Simulation

Fluid dynamics simulation using Lattice Boltzmann Method (LBM) with CPU prototype and future CUDA acceleration.

## Development Phases

**Current Phase: CPU Prototype with PyGame Visualization**
- D2Q9 Lattice Boltzmann Method implementation
- Real-time visualization with PyGame (fire/smoke coloring)
- Complex boundary conditions (inflow, outflow, walls, obstacles)
- Headless mode for benchmarking

## Project Structure
```
CuFloda/
├── cpu_lbm.py           # CPU D2Q9 LBM implementation
├── visualizer.py        # PyGame visualization
├── main.py              # Entry point (visual + headless)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
├── tests/
│   ├── __init__.py
│   ├── test_basic.py    # Unit tests
│   └── benchmark.py     # Performance benchmark
└── README.md
```

## Prerequisites

### Required
- Python 3.10+

## Installation

```bash
# Clone the repository
git clone https://github.com/UtkarshJoshiNtl/CuFloda.git
cd CuFloda

# Install dependencies
pip install -r requirements.txt
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
