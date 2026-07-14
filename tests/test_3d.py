"""Tests for 3D LBM engine (D3Q19)."""

import numpy as np
import pytest

from engines.collision import BGKCollision, MRTCollision, TRTCollision
from engines.lbm2d import LBM2D
from engines.lbm3d import LBM3D
from engines.lbm_common import LATTICE_3D_Q19, LATTICE_3D_Q27


class TestLattice3D:
    """Tests for 3D lattice constants."""

    def test_d3q19_has_19_velocities(self) -> None:
        assert LATTICE_3D_Q19.n_velocities == 19

    def test_d3q27_has_27_velocities(self) -> None:
        assert LATTICE_3D_Q27.n_velocities == 27

    def test_d3q19_weights_sum_to_one(self) -> None:
        assert abs(np.sum(LATTICE_3D_Q19.w) - 1.0) < 1e-5

    def test_d3q27_weights_sum_to_one(self) -> None:
        assert abs(np.sum(LATTICE_3D_Q27.w) - 1.0) < 1e-5

    def test_d3q19_opp_symmetric(self) -> None:
        for i in range(19):
            j = LATTICE_3D_Q19.opp[i]
            assert LATTICE_3D_Q19.opp[j] == i

    def test_d3q27_opp_symmetric(self) -> None:
        for i in range(27):
            j = LATTICE_3D_Q27.opp[i]
            assert LATTICE_3D_Q27.opp[j] == i

    def test_d3q19_rest_weight(self) -> None:
        assert abs(LATTICE_3D_Q19.w[0] - 1 / 3) < 1e-5

    def test_d3q19_face_center_weight(self) -> None:
        for i in range(1, 7):
            assert abs(LATTICE_3D_Q19.w[i] - 1 / 18) < 1e-5

    def test_d3q19_edge_center_weight(self) -> None:
        for i in range(7, 19):
            assert abs(LATTICE_3D_Q19.w[i] - 1 / 36) < 1e-5

    def test_d3q27_corner_weight(self) -> None:
        corner_indices = [0, 2, 6, 8, 18, 20, 24, 26]
        for i in corner_indices:
            assert abs(LATTICE_3D_Q27.w[i] - 1 / 216) < 1e-5


class TestInit3D:
    """Tests for 3D engine initialization."""

    def test_default_grid(self) -> None:
        sim = LBM3D()
        assert sim.depth == 64
        assert sim.height == 64
        assert sim.width == 64

    def test_custom_grid(self) -> None:
        sim = LBM3D(width=32, height=32, depth=32)
        assert sim.depth == 32
        assert sim.height == 32
        assert sim.width == 32

    def test_initial_fields_not_nan(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        assert not np.any(np.isnan(sim.rho))
        assert not np.any(np.isnan(sim.u))
        assert not np.any(np.isnan(sim.v))
        assert not np.any(np.isnan(sim.w_vel))

    def test_initial_density_uniform(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        sim.initialize(rho=2.0)
        np.testing.assert_allclose(sim.rho, 2.0, atol=1e-5)

    def test_initial_velocity_uniform(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        sim.initialize(u=0.5, v=0.3, w=0.1)
        np.testing.assert_allclose(sim.u, 0.5, atol=1e-5)
        np.testing.assert_allclose(sim.v, 0.3, atol=1e-5)
        np.testing.assert_allclose(sim.w_vel, 0.1, atol=1e-5)


class TestStep3D:
    """Tests for 3D stepping."""

    def test_step_updates_state(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        f_before = sim.f.copy()
        sim.step()
        assert not np.array_equal(sim.f, f_before)

    def test_multiple_steps(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        sim.run(10)
        assert not np.any(np.isnan(sim.rho))


class TestGetters3D:
    """Tests for 3D getter methods."""

    def test_get_density_shape(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        rho = sim.get_density()
        assert rho.shape == (16, 16, 16)

    def test_get_velocity_shape(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        vel = sim.get_velocity()
        assert vel.shape == (16, 16, 16, 3)

    def test_get_pressure(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        sim.initialize(rho=1.5)
        p = sim.get_pressure()
        np.testing.assert_allclose(p, 0.5, atol=1e-5)


class TestObstacles3D:
    """Tests for 3D obstacles."""

    def test_add_obstacle(self) -> None:
        sim = LBM3D(width=32, height=32, depth=32)
        sim.add_obstacle(16, 16, 16, radius=5)
        assert np.any(sim.obstacles)

    def test_clear_obstacles(self) -> None:
        sim = LBM3D(width=32, height=32, depth=32)
        sim.add_obstacle(16, 16, 16, radius=5)
        sim.clear_obstacles()
        assert not np.any(sim.obstacles)


class TestCollision3D:
    """Tests for 3D collision operators."""

    def test_bgk_collision(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16, collision=BGKCollision())
        sim.step()
        assert not np.any(np.isnan(sim.rho))

    def test_trt_collision_3d_not_implemented(self) -> None:
        trt = TRTCollision()
        sim = LBM3D.__new__(LBM3D)
        sim.width = sim.height = sim.depth = 16
        with pytest.raises(NotImplementedError):
            trt.collide(
                np.zeros((19, 16, 16, 16), dtype=np.float32),
                np.zeros((16, 16, 16), dtype=np.float32),
                np.zeros((16, 16, 16), dtype=np.float32),
                np.zeros((16, 16, 16), dtype=np.float32),
                LATTICE_3D_Q19, 0.02,
                w_vel=np.zeros((16, 16, 16), dtype=np.float32),
            )

    def test_mrt_collision_3d_not_implemented(self) -> None:
        mrt = MRTCollision()
        with pytest.raises(NotImplementedError):
            mrt.collide(
                np.zeros((19, 16, 16, 16), dtype=np.float32),
                np.zeros((16, 16, 16), dtype=np.float32),
                np.zeros((16, 16, 16), dtype=np.float32),
                np.zeros((16, 16, 16), dtype=np.float32),
                LATTICE_3D_Q19, 0.02,
                w_vel=np.zeros((16, 16, 16), dtype=np.float32),
            )

    def test_mass_conserved_bgk(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16, collision=BGKCollision())
        total_mass_before = np.sum(sim.rho)
        sim.step()
        total_mass_after = np.sum(sim.rho)
        assert abs(total_mass_after - total_mass_before) / total_mass_before < 1e-5


class TestEmitters3D:
    """Tests for 3D emitters."""

    def test_add_emitter(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        sim.add_emitter(8, 8, 8)
        assert sim.get_emitter_count() == 1

    def test_clear_emitters(self) -> None:
        sim = LBM3D(width=16, height=16, depth=16)
        sim.add_emitter(8, 8, 8)
        sim.clear_emitters()
        assert sim.get_emitter_count() == 0


class TestTRTCollision2D:
    """Tests for TRT collision operator in 2D."""

    def test_trt_produces_valid_output(self) -> None:
        trt = TRTCollision()
        sim = LBM2D(width=32, height=32, collision=trt)
        sim.step()
        assert not np.any(np.isnan(sim.rho))
        assert not np.any(np.isnan(sim.u))
        assert not np.any(np.isnan(sim.v))

    def test_trt_custom_s_minus(self) -> None:
        trt = TRTCollision(s_minus=0.5)
        sim = LBM2D(width=32, height=32, collision=trt)
        sim.step()
        assert not np.any(np.isnan(sim.rho))


class TestMRTCollision2D:
    """Tests for MRT collision operator in 2D."""

    def test_mrt_produces_valid_output(self) -> None:
        mrt = MRTCollision()
        sim = LBM2D(width=32, height=32, collision=mrt)
        sim.step()
        assert not np.any(np.isnan(sim.rho))
        assert not np.any(np.isnan(sim.u))
        assert not np.any(np.isnan(sim.v))
