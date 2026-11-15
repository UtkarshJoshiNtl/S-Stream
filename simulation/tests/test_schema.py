import pytest

from sstream.schema import (
    FluidSpec,
    MeshTransform,
    TunnelSpec,
    snap_value,
)


def test_default_tunnel_derives_a_cubic_cell_grid():
    tunnel = TunnelSpec()
    assert tunnel.grid_shape == (256, 102, 77)
    assert tunnel.cell_count == 256 * 102 * 77
    assert tunnel.cell_size == pytest.approx(6.0 / 256.0)
    assert tunnel.grid_spacing == (tunnel.cell_size,) * 3
    assert tunnel.bounds == (0.0, 6.0, -1.2, 1.2, 0.0, 1.8)

    size = tunnel.cell_size
    assert tunnel.grid_origin == pytest.approx((size / 2, -1.2 + size / 2, size / 2))


def test_with_resolution_keeps_the_same_tunnel():
    tunnel = TunnelSpec().with_resolution(128)
    assert tunnel.cells_streamwise == 128
    assert tunnel.length == 6.0
    assert tunnel.grid_shape[0] == 128
    assert tunnel.cell_size == pytest.approx(6.0 / 128.0)


def test_fluid_reference_numbers():
    fluid = FluidSpec()
    assert fluid.reynolds == pytest.approx(30.0 * 1.2 / 1.5e-5)
    assert fluid.dynamic_pressure == pytest.approx(0.5 * 1.225 * 30.0**2)


def test_snap_value():
    assert snap_value(17.0, 15.0) == 15.0
    assert snap_value(23.0, 15.0) == 30.0
    assert snap_value(-23.0, 15.0) == -30.0
    assert snap_value(3.04, 0.1) == 3.0
    assert snap_value(3.06, 0.1) == 3.1
    assert snap_value(17.0, 0.0) == 17.0


def test_transform_snapping_matches_the_gui_contract():
    transform = MeshTransform(
        pitch=17.0,
        roll=-23.0,
        yaw=7.0,
        tunnel_yaw=38.0,
        position_x=3.04,
        position_y=-0.26,
    )
    snapped = transform.snapped()
    assert snapped.pitch == 15.0
    assert snapped.roll == -30.0
    assert snapped.yaw == 0.0
    assert snapped.tunnel_yaw == 45.0
    assert snapped.position_x == pytest.approx(3.0)
    assert snapped.position_y == pytest.approx(-0.3)

    fine = transform.snapped(angle_step=1.0, position_step=0.01)
    assert fine.pitch == 17.0
    assert fine.tunnel_yaw == 38.0
    assert fine.position_x == pytest.approx(3.04)


def test_transform_reset_only_clears_rotation():
    transform = MeshTransform(pitch=10, roll=20, yaw=30, tunnel_yaw=40, position_x=2.5)
    reset = transform.reset()
    assert (reset.pitch, reset.roll, reset.yaw, reset.tunnel_yaw) == (0, 0, 0, 0)
    assert reset.position_x == 2.5
