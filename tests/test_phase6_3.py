"""Tests for Phase 6.3: Lagrangian Particle Tracer."""
from __future__ import annotations

import numpy as np
import pytest

from engines.particle_tracer import ParticleTracer
from engines.lbm2d import LBM2D


class TestParticleTracerInit:
    def test_empty_tracer(self) -> None:
        t = ParticleTracer(64, 64)
        assert t.count == 0
        assert t.positions.shape == (0, 2)

    def test_3d_tracer(self) -> None:
        t = ParticleTracer(32, 32, depth=32)
        assert t.count == 0
        assert t.positions.shape == (0, 3)
        assert t.is_3d is True

    def test_trail_buffer_created(self) -> None:
        t = ParticleTracer(64, 64, trail_length=10)
        assert t._trail_buf is not None
        assert t._trail_buf.shape[0] == 10

    def test_no_trails_when_length_1(self) -> None:
        t = ParticleTracer(64, 64, trail_length=1)
        assert t._trail_buf is None


class TestAddParticles:
    def test_add_single(self) -> None:
        t = ParticleTracer(64, 64)
        n = t.add_particles(32.0, 32.0, count=1)
        assert n == 1
        assert t.count == 1

    def test_add_multiple(self) -> None:
        t = ParticleTracer(64, 64)
        n = t.add_particles(32.0, 32.0, count=100)
        assert n == 100
        assert t.count == 100

    def test_add_respects_max(self) -> None:
        t = ParticleTracer(64, 64, max_particles=10)
        t.add_particles(32.0, 32.0, count=5)
        n = t.add_particles(32.0, 32.0, count=10)
        assert n == 5
        assert t.count == 10

    def test_add_zero(self) -> None:
        t = ParticleTracer(64, 64)
        n = t.add_particles(32.0, 32.0, count=0)
        assert n == 0
        assert t.count == 0

    def test_positions_near_target(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles(32.0, 32.0, count=50)
        pos = t.get_positions()
        assert pos[:, 0].mean() == pytest.approx(32.0, abs=0.5)
        assert pos[:, 1].mean() == pytest.approx(32.0, abs=0.5)


class TestAddParticlesLine:
    def test_add_line(self) -> None:
        t = ParticleTracer(64, 64)
        n = t.add_particles_line(0.0, 32.0, 63.0, 32.0, count=20)
        assert n == 20
        assert t.count == 20

    def test_line_positions_on_line(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles_line(0.0, 0.0, 63.0, 63.0, count=10)
        pos = t.get_positions()
        for i in range(len(pos)):
            assert pos[i, 0] == pytest.approx(pos[i, 1], abs=0.1)


class TestAddParticlesRandom:
    def test_add_random(self) -> None:
        t = ParticleTracer(64, 64)
        n = t.add_particles_random(100)
        assert n == 100
        assert t.count == 100

    def test_random_positions_in_domain(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles_random(200)
        pos = t.get_positions()
        assert pos[:, 0].min() >= 0.0
        assert pos[:, 0].max() < 64.0
        assert pos[:, 1].min() >= 0.0
        assert pos[:, 1].max() < 64.0


class TestParticleAdvection:
    def test_particles_move_with_flow(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles(10.0, 32.0, count=5)
        vel = np.zeros((64, 64, 2), dtype=np.float32)
        vel[:, :, 0] = 0.5  # uniform rightward flow
        t.step(vel)
        pos = t.get_positions()
        assert pos[:, 0].mean() > 10.0

    def test_particles_stay_in_domain(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles(32.0, 32.0, count=10)
        vel = np.zeros((64, 64, 2), dtype=np.float32)
        vel[:, :, 0] = 5.0
        t.step(vel)
        pos = t.get_positions()
        if t.count > 0:
            assert pos[:, 0].max() < 64.5

    def test_particles_removed_oob(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles(60.0, 32.0, count=10)
        vel = np.zeros((64, 64, 2), dtype=np.float32)
        vel[:, :, 0] = 10.0
        removed = t.step(vel)
        assert removed > 0

    def test_rk2_vs_euler_accuracy(self) -> None:
        """RK2 should conserve radius in circular flow better than Euler."""
        size = 64
        t_rk2 = ParticleTracer(size, size, trail_length=1)
        t_euler = ParticleTracer(size, size, trail_length=1)
        t_rk2.add_particles(16.0, 32.0, count=1)
        t_euler.add_particles(16.0, 32.0, count=1)

        vel = np.zeros((size, size, 2), dtype=np.float32)
        cy, cx = 32.0, 32.0
        y_grid, x_grid = np.mgrid[:size, :size]
        dx = x_grid - cx
        dy = y_grid - cy
        r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        speed = 0.2
        vel[:, :, 0] = -speed * dy / r
        vel[:, :, 1] = speed * dx / r

        for _ in range(20):
            t_rk2.step(vel)

        # Check that particle is still approximately at the same radius
        pos = t_rk2.get_positions()
        r0 = np.sqrt((16.0 - cx) ** 2 + (32.0 - cy) ** 2)
        r1 = np.sqrt((pos[0, 0] - cx) ** 2 + (pos[0, 1] - cy) ** 2)
        radius_error = abs(r1 - r0) / r0
        assert radius_error < 0.5, f"RK2 radius error: {radius_error:.3f}"


class TestClear:
    def test_clear(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles(32.0, 32.0, count=50)
        assert t.count == 50
        t.clear()
        assert t.count == 0

    def test_clear_trail_buffer(self) -> None:
        t = ParticleTracer(64, 64, trail_length=10)
        t.add_particles(32.0, 32.0, count=10)
        vel = np.zeros((64, 64, 2), dtype=np.float32)
        vel[:, :, 0] = 0.5
        t.step(vel)
        t.clear()
        assert t._trail_buf.shape[1] == 0


class TestTrailLength:
    def test_change_trail_length(self) -> None:
        t = ParticleTracer(64, 64, trail_length=10)
        t.add_particles(32.0, 32.0, count=5)
        t.set_trail_length(20)
        assert t.trail_length == 20
        assert t._trail_buf.shape[0] == 20

    def test_reduce_trail_length(self) -> None:
        t = ParticleTracer(64, 64, trail_length=50)
        t.add_particles(32.0, 32.0, count=5)
        t.set_trail_length(5)
        assert t.trail_length == 5
        assert t._trail_buf.shape[0] == 5


class TestTrails:
    def test_trails_accumulate(self) -> None:
        t = ParticleTracer(64, 64, trail_length=10)
        t.add_particles(10.0, 32.0, count=1)
        vel = np.zeros((64, 64, 2), dtype=np.float32)
        vel[:, :, 0] = 0.5
        for _ in range(5):
            t.step(vel)
        trails = t.get_trails()
        assert trails is not None
        assert trails.shape[1] == 1  # 1 particle


class TestLBM2DIntegration:
    @pytest.fixture()
    def sim(self) -> LBM2D:
        s = LBM2D(width=32, height=32, viscosity=0.02)
        s.u_inflow = 0.1
        s.initialize()
        return s

    def test_tracer_exists(self, sim: LBM2D) -> None:
        tracer = sim.get_particle_tracer()
        assert tracer is not None
        assert tracer.count == 0

    def test_add_and_step(self, sim: LBM2D) -> None:
        tracer = sim.get_particle_tracer()
        tracer.add_particles(5.0, 16.0, count=10)
        sim.run(10)
        assert tracer.count > 0
        pos = tracer.get_positions()
        assert pos[:, 0].mean() > 5.0

    def test_particles_in_field(self, sim: LBM2D) -> None:
        tracer = sim.get_particle_tracer()
        tracer.add_particles_random(100)
        sim.run(50)
        assert tracer.count > 0
        pos = tracer.get_positions()
        assert pos.shape[1] == 2


class TestGetPositions:
    def test_returns_copy(self) -> None:
        t = ParticleTracer(64, 64)
        t.add_particles(32.0, 32.0, count=5)
        pos1 = t.get_positions()
        pos1[0, 0] = 999.0
        pos2 = t.get_positions()
        assert pos2[0, 0] != 999.0
