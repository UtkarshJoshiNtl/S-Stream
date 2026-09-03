"""Scene reset must not impose the scene's IC on engines that own their own.

Regression cover for PLAN.md C10: ``apply_to_sim`` used to call
``initialize(rho=1.0, u=scene.u_inflow, v=0.0)`` on every engine, including
the closed-domain Shan–Chen engines whose whole point is a low-density vapor
background and a quiescent droplet.
"""

from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from scene.scene import Scene, apply_to_sim


def _scene(u_inflow: float = 0.15) -> Scene:
    return Scene(u_inflow=u_inflow, viscosity=0.02)


class TestChannelEngineKeepsSceneDrive:
    def test_inflow_seeded_engine_still_uses_scene_velocity(self) -> None:
        sim = LBM2D(width=32, height=32)
        apply_to_sim(_scene(u_inflow=0.12), sim)
        assert sim.rho[16, 16] == pytest.approx(1.0, abs=1e-5)
        assert np.mean(sim.u) == pytest.approx(0.12, abs=1e-3)


class TestLiquidEngineOwnsItsInitialCondition:
    def test_vapor_background_is_low_density(self) -> None:
        sim = LBM2DLiquid(width=48, height=48)
        apply_to_sim(_scene(), sim)
        assert np.median(sim.rho) == pytest.approx(0.01, abs=1e-6)

    def test_starts_quiescent(self) -> None:
        sim = LBM2DLiquid(width=48, height=48)
        apply_to_sim(_scene(u_inflow=0.15), sim)
        assert np.allclose(sim.u, 0.0, atol=1e-6)
        assert np.allclose(sim.v, 0.0, atol=1e-6)

    def test_droplet_is_still_seeded(self) -> None:
        sim = LBM2DLiquid(width=48, height=48)
        apply_to_sim(_scene(), sim)
        centre = 24, 24
        assert sim.rho[centre] == pytest.approx(2.0, abs=1e-6)
        assert sim.rho[0, 0] == pytest.approx(0.01, abs=1e-6)

    def test_scene_u_inflow_does_not_seed_velocity(self) -> None:
        """The scene's inlet velocity is inert here; the engine must not use it."""
        slow = LBM2DLiquid(width=48, height=48)
        fast = LBM2DLiquid(width=48, height=48)
        apply_to_sim(_scene(u_inflow=0.0), slow)
        apply_to_sim(_scene(u_inflow=0.15), fast)
        slow.run(20)
        fast.run(20)
        np.testing.assert_allclose(fast.rho, slow.rho, atol=1e-6)

    def test_repeated_apply_is_idempotent(self) -> None:
        sim = LBM2DLiquid(width=48, height=48)
        apply_to_sim(_scene(), sim)
        first = sim.rho.copy()
        apply_to_sim(_scene(), sim)
        np.testing.assert_allclose(sim.rho, first, atol=1e-9)


class TestMultiComponentEngineOwnsItsInitialCondition:
    def test_starts_quiescent(self) -> None:
        sim = LBM2DMultiComponent(width=40, height=40)
        apply_to_sim(_scene(u_inflow=0.15), sim)
        assert np.allclose(sim.u, 0.0, atol=1e-6)
        assert np.allclose(sim.v, 0.0, atol=1e-6)

    def test_phase_structure_is_seeded(self) -> None:
        sim = LBM2DMultiComponent(width=40, height=40)
        apply_to_sim(_scene(), sim)
        assert sim.rho1[20, 20] == pytest.approx(2.0, abs=1e-6)
        assert sim.rho2[0, 0] == pytest.approx(2.0, abs=1e-6)
        assert sim.rho1[0, 0] == pytest.approx(0.01, abs=1e-6)
        assert sim.rho2[20, 20] == pytest.approx(0.01, abs=1e-6)


class TestResetForSceneIsTheSingleEntryPoint:
    def test_default_implementation_matches_old_behaviour(self) -> None:
        sim = LBM2D(width=32, height=32)
        sim.reset_for_scene(_scene(u_inflow=0.09))
        assert np.mean(sim.u) == pytest.approx(0.09, abs=1e-3)

    def test_every_engine_exposes_reset_for_scene(self) -> None:
        for cls in (LBM2D, LBM2DLiquid, LBM2DMultiComponent):
            assert callable(cls.reset_for_scene)
