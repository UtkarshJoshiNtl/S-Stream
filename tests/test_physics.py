from __future__ import annotations

import numpy as np
import pytest

from engines.lbm2d import LBM2D


def _parabolic_profile(y: np.ndarray, a: float, H: int) -> np.ndarray:
    return a * y * (H - 1 - y)


def _fit_parabola(u: np.ndarray, H: int) -> float:
    y = np.arange(H, dtype=np.float64)
    return np.sum(u * y * (H - 1 - y)) / np.sum((y * (H - 1 - y)) ** 2)


@pytest.mark.slow
class TestPoiseuilleFlow:
    """2D channel flow should develop a parabolic (Poiseuille) profile."""

    H = 32
    W = 128
    U0 = 0.05
    NU = 0.05
    STEPS = 8000
    TOL = 0.15

    @pytest.fixture(scope="class")
    def sim(self) -> LBM2D:
        """The settled channel, run once.

        All four assertions below read `sim.u` and none of them write to it, so
        one 8000-step settle answers all four. Running it per test meant 32000
        steps to ask the same question four times.
        """
        s = LBM2D(width=self.W, height=self.H, viscosity=self.NU)
        s.u_inflow = self.U0
        s.initialize(rho=1.0, u=self.U0, v=0.0)
        s.run(self.STEPS)
        return s

    def test_profile_symmetric(self, sim: LBM2D) -> None:
        _, _, H = sim.width, sim.height, self.H
        u = sim.u[:, sim.width // 2]
        for j in range(H // 2):
            assert (
                abs(u[j] - u[H - 1 - j]) < self.TOL * u[H // 2]
            ), f"asymmetry at y={j}: u[{j}]={u[j]:.6f}, u[{H-1-j}]={u[H-1-j]:.6f}"

    def test_profile_max_at_center(self, sim: LBM2D) -> None:
        H = self.H
        u = sim.u[:, sim.width // 2]
        center = H // 2
        maxima = np.argmax(u)
        diff = abs(int(maxima) - center)
        assert diff <= 1, f"profile max at y={maxima}, expected near center y={center}"

    def test_profile_parabolic_shape(self, sim: LBM2D) -> None:
        H = self.H
        u = sim.u[:, sim.width // 2]
        a_fit = _fit_parabola(u, H)
        u_fit = _parabolic_profile(np.arange(H, dtype=np.float64), a_fit, H)
        residuals = u - u_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((u - np.mean(u)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.98, f"parabolic fit R²={r2:.4f}, expected > 0.98"

    def test_profile_developed_downstream(self, sim: LBM2D) -> None:
        W = sim.width
        u_left = sim.u[:, W // 4]
        u_mid = sim.u[:, W // 2]
        u_right = sim.u[:, 3 * W // 4]
        diff_mid_left = np.max(np.abs(u_mid - u_left))
        diff_right_mid = np.max(np.abs(u_right - u_mid))
        assert max(diff_mid_left, diff_right_mid) < self.U0 * 0.05, (
            f"profile not converged: mid-vs-left diff={diff_mid_left:.6f}, "
            f"right-vs-mid diff={diff_right_mid:.6f}"
        )


@pytest.mark.slow
class TestVortexShedding:
    """Flow past a cylinder should produce a Kármán vortex street at Re~200."""

    H = 48
    W = 192
    U0 = 0.10
    NU = 0.01
    SETTLE_STEPS = 12000
    OBSERVATION_STEPS = 4000

    @pytest.fixture(scope="class")
    def shedding(self) -> tuple[LBM2D, np.ndarray]:
        """One trajectory, settle then observe, shared by both assertions.

        The oscillation test needs the `v` series recorded *during* the
        observation window and the wake test needs the field at the end of it;
        both describe the same run. Running them separately meant settling to the
        same 12000-step state twice (32000 steps) to learn two things about one
        simulation.
        """
        s = LBM2D(width=self.W, height=self.H, viscosity=self.NU)
        s.u_inflow = self.U0
        s.initialize(rho=1.0, u=self.U0, v=0.0)
        s.add_obstacle(x=self.W // 4, y=self.H // 2, radius=5)
        s.run(self.SETTLE_STEPS)
        probe_x = s.width // 2 + s.width // 4
        probe_y = s.height // 2 + s.height // 4
        v_samples = []
        for _ in range(self.OBSERVATION_STEPS):
            s.step()
            v_samples.append(s.v[probe_y, probe_x])
        return s, np.array(v_samples)

    def test_v_velocity_oscillates(self, shedding) -> None:
        _, v_arr = shedding
        std_v = np.std(v_arr)
        assert (
            std_v > 1e-4
        ), f"v-velocity std={std_v:.6f} too low — no vortex shedding detected"
        fft = np.fft.rfft(v_arr - np.mean(v_arr))
        peak_amp = np.max(np.abs(fft[1:]))
        assert peak_amp > 10 * np.std(np.abs(fft[1:])), (
            "no clear spectral peak in v-velocity — " "no periodic shedding detected"
        )

    def test_wake_velocity_deficit(self, shedding) -> None:
        sim, _ = shedding
        obstacle_x = sim.width // 4
        u_wake = sim.u[:, obstacle_x + 10]
        u_far = sim.u[:, sim.width - 10]
        wake_center = np.mean(u_wake[sim.height // 3 : 2 * sim.height // 3])
        far_center = np.mean(u_far[sim.height // 3 : 2 * sim.height // 3])
        assert wake_center < far_center * 0.95, (
            "wake velocity deficit not observed: "
            f"wake={wake_center:.6f}, far={far_center:.6f}"
        )
