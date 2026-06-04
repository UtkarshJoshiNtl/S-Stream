from __future__ import annotations

from engines.base import SimEngine
from engines.lbm2d import LBM2D
from engines.lbm3d_cpu import LBM3DCPU

try:
    from engines.lbm3d_gpu import LBM3DGPU
except ImportError:
    LBM3DGPU = None  # type: ignore[assignment]

__all__ = ["SimEngine", "LBM2D", "LBM3DCPU", "LBM3DGPU"]
