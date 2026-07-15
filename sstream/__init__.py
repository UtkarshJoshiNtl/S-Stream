"""S-Stream: Lattice Boltzmann fluid simulation platform.

Usage::

    import sstream
    sim = sstream.LBM2D(width=256, height=256)
    sim.add_obstacle(128, 128, radius=20)
    sim.run(1000)
    sim.plot_velocity()   # inline matplotlib figure in Jupyter

    # Or headless:
    sim = sstream.LBM2D(width=128, height=128)
    sim.run(500)
    print(sim.get_pressure().mean())

Available engines:
    - LBM2D: D2Q9 CPU engine (default)
    - LBM3D: D3Q19 CPU engine
    - LBM2DLiquid: Shan-Chen multiphase liquid engine
"""

from __future__ import annotations

from engines.base import SimEngine
from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from engines.lbm3d import LBM3D
from engines.particle_tracer import ParticleTracer

try:
    from engines.lbm2d_lettuce import LBM2DLettuce
except ImportError:
    LBM2DLettuce = None  # type: ignore[assignment,misc]

try:
    from engines.lbm2d_gpu import LBM2DGPU
except ImportError:
    LBM2DGPU = None  # type: ignore[assignment,misc]

try:
    from engines.lbm3d_gpu import LBM3DGPU
except ImportError:
    LBM3DGPU = None  # type: ignore[assignment,misc]

__all__ = [
    "SimEngine",
    "LBM2D",
    "LBM3D",
    "LBM2DLiquid",
    "LBM2DMultiComponent",
    "LBM2DLettuce",
    "LBM2DGPU",
    "LBM3DGPU",
    "ParticleTracer",
]

__version__ = "0.3.5"
