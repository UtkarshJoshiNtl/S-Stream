from __future__ import annotations

from engines.base import SimEngine
from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent

try:
    from engines.lbm2d_gpu import LBM2DGPU
except ImportError:
    LBM2DGPU = None  # type: ignore[assignment]

__all__ = ["SimEngine", "LBM2D", "LBM2DGPU", "LBM2DLiquid", "LBM2DMultiComponent"]
