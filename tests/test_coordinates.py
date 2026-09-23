import numpy as np
import pytest

from cap_flight_sim.coordinates import (
    body_to_world,
    euler_deg_to_quat,
    kmh_to_m_s,
    quat_derivative,
    quat_from_axis_angle,
    quat_from_face_normal,
    quat_multiply,
    quat_normalize,
    quat_to_euler_deg,
    quat_to_matrix,
    rad_s_to_rpm,
    rpm_to_rad_s,
    safe_unit,
    world_to_body,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

TOL = 1e-12


def test_safe_unit_normalizes_and_handles_zero():
    assert np.linalg.norm(safe_unit(np.array([3.0, 4.0, 0.0]))) == pytest.approx(1.0, abs=TOL)
    assert not safe_unit(np.zeros(3)).any()


def test_world_frame_is_right_handed():
    x, y, z = np.eye(3)
    assert np.allclose(np.cross(x, y), z)


def test_body_world_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = quat_normalize(rng.normal(size=4))
        v = rng.normal(size=3)
        assert np.allclose(world_to_body(q, body_to_world(q, v)), v, atol=1e-12)
        assert np.linalg.det(quat_to_matrix(q)) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_normalization():
    q = quat_normalize(np.array([2.0, 0.0, 0.0, 0.0]))
    assert np.allclose(q, [1, 0, 0, 0])
    with pytest.raises(ValueError):
        quat_normalize(np.zeros(4))


def test_rotation_90deg_about_z_maps_x_to_y():
    q = quat_from_axis_angle(np.array([0, 0, 1.0]), np.pi / 2)
    assert np.allclose(body_to_world(q, np.array([1.0, 0, 0])), [0, 1, 0], atol=1e-12)


def test_positive_pitch_points_nose_down_in_flu():
    q = euler_deg_to_quat([0.0, 10.0, 0.0])
    nose = body_to_world(q, np.array([1.0, 0, 0]))
    assert nose[2] < 0


def test_euler_round_trip_away_from_gimbal_lock():
    angles = np.array([12.0, -30.0, 75.0])
    assert np.allclose(quat_to_euler_deg(euler_deg_to_quat(angles)), angles, atol=1e-9)


def test_face_normal_orientation():
    q = quat_from_face_normal(np.array([0.0, 1.0, 0.0]))
    assert np.allclose(body_to_world(q, np.array([0, 0, 1.0])), [0, 1, 0], atol=1e-12)
    assert np.allclose(body_to_world(q, np.array([1.0, 0, 0])), [1, 0, 0], atol=1e-12)


def test_unit_conversions():
    assert rpm_to_rad_s(60.0) == pytest.approx(2 * np.pi)
    assert rad_s_to_rpm(rpm_to_rad_s(1234.0)) == pytest.approx(1234.0)
    assert kmh_to_m_s(36.0) == pytest.approx(10.0)


def test_quaternion_order_conversion_round_trip_and_scipy_agreement():
    from scipy.spatial.transform import Rotation

    q = quat_normalize(np.array([0.3, -0.2, 0.9, 0.1]))
    assert np.allclose(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)
    assert np.allclose(Rotation.from_quat(wxyz_to_xyzw(q)).as_matrix(), quat_to_matrix(q), atol=1e-12)


def test_quaternion_derivative_matches_finite_rotation():
    q0 = quat_normalize(np.array([0.9, 0.1, -0.3, 0.2]))
    omega = np.array([3.0, -1.0, 2.0])
    dt = 1e-6
    q1 = quat_multiply(q0, quat_from_axis_angle(omega, np.linalg.norm(omega) * dt))
    assert np.allclose((q1 - q0) / dt, quat_derivative(q0, omega), atol=1e-5)
