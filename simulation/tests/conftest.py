import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


@pytest.fixture(scope="session")
def tiny_result():
    """One small, complete solve shared by the pipeline and visual tests."""
    from sstream.pipeline import SolvePlan, run_case
    from sstream.preview import RunPreset
    from sstream.schema import FluidSpec, TunnelSpec
    from sstream.voxelize import VoxelResult

    tunnel = TunnelSpec(cells_streamwise=48)
    shape = tunnel.grid_shape
    x, y, z = np.indices(shape)
    solid = (
        (x - shape[0] * 0.35) ** 2
        + (y - shape[1] * 0.5) ** 2
        + (z - shape[2] * 0.45) ** 2
    ) < (min(shape) * 0.18) ** 2
    mask = VoxelResult(
        solid=solid,
        boundary=solid,
        origin=tunnel.grid_origin,
        spacing=tunnel.grid_spacing,
        dimensions=shape,
    )
    plan = SolvePlan(
        mask=mask,
        tunnel=tunnel,
        fluid=FluidSpec(),
        preset=RunPreset("Test", 48, 640, "pipeline test"),
    )
    return run_case(plan)
