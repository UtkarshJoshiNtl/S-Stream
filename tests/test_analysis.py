from __future__ import annotations

import numpy as np

from analysis.ai_context import build_ai_context, local_ai_response
from analysis.physics import (
    characteristic_length,
    drag_coefficient,
    reynolds_number,
    strouhal_number,
)
from analysis.regimes import detect_flow_regime
from analysis.sanity import check_sanity
from analysis.scorecard import compute_scorecard
from analysis.sweep import SweepResult, run_sweep
from engines.lbm2d import LBM2D
from scene.scene import CircleObstacle, Scene


def test_reynolds_number() -> None:
    sim = LBM2D(64, 64, viscosity=0.02)
    sim.u_inflow = 0.15
    Re = reynolds_number(sim)
    expected = 0.15 * 64 / 0.02
    assert abs(Re - expected) < 0.01


def test_reynolds_number_with_obstacle() -> None:
    sim = LBM2D(64, 64, viscosity=0.01)
    sim.u_inflow = 0.1
    Re = reynolds_number(sim, obstacle_diameter=10.0)
    expected = 0.1 * 10.0 / 0.01
    assert abs(Re - expected) < 0.01


def test_reynolds_number_zero_inflow() -> None:
    sim = LBM2D(32, 32)
    sim.u_inflow = 0.0
    assert reynolds_number(sim) == 0.0


def test_drag_coefficient_no_obstacle() -> None:
    sim = LBM2D(32, 32)
    assert drag_coefficient(sim) == 0.0


def test_drag_coefficient_cylinder() -> None:
    sim = LBM2D(64, 64, viscosity=0.02)
    sim.u_inflow = 0.1
    sim.initialize(rho=1.0, u=0.1, v=0.0)
    sim.add_obstacle(32, 32, radius=5)
    for _ in range(2000):
        sim.step()
    Cd = drag_coefficient(sim)
    assert Cd > 0.0
    # cylinder at Re ≈ 32 should have Cd > 1.0
    assert Cd > 0.5


def test_strouhal_number_regular_sine() -> None:
    n = 256
    dt = 0.1
    freq_bin = 5
    freq = freq_bin / (n * dt)  # exactly aligns with FFT bin
    t = np.arange(n) * dt
    v = np.sin(2 * np.pi * freq * t)
    St = strouhal_number(list(v), dt=dt, diameter=1.0, velocity=1.0)
    assert St is not None
    assert abs(St - freq) < 0.01


def test_strouhal_number_noise() -> None:
    rng = np.random.default_rng(42)
    noise = list(rng.normal(0, 0.01, 200))
    St = strouhal_number(noise, dt=0.01)
    assert St is None


def test_strouhal_number_short_history() -> None:
    assert strouhal_number([1.0, 2.0, 3.0], dt=1.0) is None


def test_characteristic_length_circle() -> None:
    scene = Scene(obstacles=[CircleObstacle(name="Cyl", radius=10)])
    assert characteristic_length(scene) == 20.0


def test_characteristic_length_no_obstacle() -> None:
    scene = Scene(width=128, height=64)
    assert characteristic_length(scene) == 128.0


def test_detect_flow_regime_no_flow() -> None:
    sim = LBM2D(32, 32)
    sim.u_inflow = 0.0
    scene = Scene()
    regime = detect_flow_regime(sim, scene, [], 0)
    assert regime.label == "No driven flow"
    assert regime.confidence == 0.95


def test_detect_flow_regime_developing() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.1
    scene = Scene(obstacles=[CircleObstacle(name="Cyl", x=16, y=16, radius=5)])
    regime = detect_flow_regime(sim, scene, [], 100)
    assert regime.label == "Developing flow"
    assert regime.confidence == 0.7


def test_detect_flow_regime_no_obstacle() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.1
    scene = Scene()
    regime = detect_flow_regime(sim, scene, [], 500)
    assert regime.label == "Open channel flow"


def test_check_sanity_low_viscosity() -> None:
    sim = LBM2D(32, 32, viscosity=0.001)
    sim.u_inflow = 0.1
    scene = Scene()
    warnings = check_sanity(sim, scene, [], 0)
    assert any(w.title == "Low viscosity" for w in warnings)


def test_check_sanity_fast_inflow() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.3
    scene = Scene()
    warnings = check_sanity(sim, scene, [], 0)
    assert any("inlet" in w.title.lower() for w in warnings)


def test_check_sanity_no_obstacle() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.1
    scene = Scene()
    warnings = check_sanity(sim, scene, [], 0)
    assert any(w.title == "No obstacle" for w in warnings)


def test_check_sanity_coarse_obstacle() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.1
    scene = Scene(obstacles=[CircleObstacle(name="Cyl", radius=3)])
    warnings = check_sanity(sim, scene, [], 0)
    assert any(w.title == "Coarse obstacle" for w in warnings)


def test_compute_scorecard_no_obstacle() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.1
    scene = Scene()
    score = compute_scorecard(sim, scene, [], 0)
    assert "Open flow" in score.summary


def test_scorecard_to_dict() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.1
    scene = Scene()
    score = compute_scorecard(sim, scene, [], 0)
    d = score.to_dict()
    assert "drag_coefficient" in d
    assert "reynolds_number" in d
    assert "summary" in d


def test_build_ai_context() -> None:
    sim = LBM2D(32, 32, viscosity=0.02)
    sim.u_inflow = 0.1
    scene = Scene(name="Test Scene", description="A test")
    context = build_ai_context(scene, sim, None, [], 0)
    assert "Test Scene" in context
    assert "A test" in context
    assert "SStream" in context


def test_local_ai_response() -> None:
    response = local_ai_response("test context", has_api_key=False)
    assert "local preview" in response.lower()
    response_with_key = local_ai_response("test context", has_api_key=True)
    assert "Gemini" in response_with_key


def test_sweep_result_serialization() -> None:
    result = SweepResult(
        parameter="viscosity",
        values=[0.01, 0.02],
        measurements=["reynolds_number"],
        data={"reynolds_number": [100.0, 50.0]},
        elapsed=1.5,
    )
    d = result.to_dict()
    assert d["parameter"] == "viscosity"
    assert d["elapsed"] == 1.5
    restored = SweepResult.from_dict(d)
    assert restored.parameter == "viscosity"
    assert restored.elapsed == 1.5


def test_sweep_small() -> None:
    scene = Scene(
        width=32,
        height=32,
        viscosity=0.02,
        u_inflow=0.1,
        obstacles=[CircleObstacle(name="Cyl", x=16, y=16, radius=4)],
    )
    result = run_sweep(
        scene,
        parameter="viscosity",
        values=[0.01, 0.02],
        measurements=["reynolds_number"],
        steps_per_run=100,
    )
    assert result.parameter == "viscosity"
    assert len(result.values) == 2
    assert "reynolds_number" in result.data
    assert len(result.data["reynolds_number"]) == 2
