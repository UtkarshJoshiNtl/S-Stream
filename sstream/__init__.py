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

import pathlib

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


def _resolve_version() -> str:
    """Resolve the package version from its single source of truth.

    ``pyproject.toml`` wins because that is what a developer edits and what
    build backends read. Installed distribution metadata is the fallback for
    layouts where the source tree is unavailable.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        pass
    else:
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        else:
            found = data.get("project", {}).get("version")
            if isinstance(found, str):
                return found
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover - Python < 3.8
        return "0.0.0+unknown"
    try:
        return version("s-stream")
    except Exception:
        return "0.0.0+unknown"


__version__ = _resolve_version()
