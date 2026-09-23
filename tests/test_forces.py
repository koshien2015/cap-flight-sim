import numpy as np
import pytest

from cap_flight_sim.aerodynamics.factory import build_aerodynamic_model
from cap_flight_sim.coordinates import quat_from_face_normal, world_to_body
from cap_flight_sim.config import config_from_dict
from cap_flight_sim.models import RigidBodyState


def _state(velocity, normal=(0.0, 0.0, 1.0), omega_world=(0.0, 0.0, 0.0)):
    q = quat_from_face_normal(np.array(normal, dtype=float))
    return RigidBodyState(
        0.0, np.array([0.0, 0.0, 1.0]), np.array(velocity, dtype=float), q,
        world_to_body(q, np.array(omega_world, dtype=float)),
    )


@pytest.fixture
def setup(base_raw):
    cfg = config_from_dict(base_raw)
    return cfg, build_aerodynamic_model(cfg.aerodynamics, cfg.cap)


def test_drag_opposes_velocity(setup):
    cfg, model = setup
    v = np.array([15.0, 2.0, -1.0])
    res = model.evaluate(_state(v, normal=(0.3, 0.2, 0.9)), cfg.cap, cfg.environment)
    assert np.dot(res.drag_world_n, v) < 0
    assert np.allclose(np.cross(res.drag_world_n, v), 0, atol=1e-12)


def test_lift_perpendicular_to_relative_wind(setup):
    cfg, model = setup
    v = np.array([15.0, 0.0, -1.5])  # 面水平・やや下降 → α > 0
    res = model.evaluate(_state(v), cfg.cap, cfg.environment)
    assert np.dot(res.lift_world_n, v) == pytest.approx(0.0, abs=1e-12)
    assert res.lift_world_n[2] > 0
    assert res.diagnostics["angle_of_attack_deg"] > 0


def test_lift_invariant_to_flipping_symmetric_cap(setup):
    cfg, model = setup
    v = np.array([15.0, 0.0, -1.5])
    up = model.evaluate(_state(v, normal=(0, 0, 1)), cfg.cap, cfg.environment)
    down = model.evaluate(_state(v, normal=(0, 0, -1)), cfg.cap, cfg.environment)
    assert np.allclose(up.lift_world_n, down.lift_world_n, atol=1e-12)


def test_magnus_direction_follows_right_hand_rule(setup):
    cfg, model = setup
    # ω = +Z（上から見て反時計回り）, v = +X → ω × v = +Y（左）
    res = model.evaluate(_state([16.0, 0, 0], omega_world=(0, 0, 150.0)), cfg.cap, cfg.environment)
    assert res.magnus_world_n[1] > 0
    assert abs(res.magnus_world_n[0]) < 1e-12 and abs(res.magnus_world_n[2]) < 1e-12


def test_backspin_gives_upward_magnus(setup):
    cfg, model = setup
    # FLU では ω = -Y がバックスピン（v=+X）
    res = model.evaluate(_state([16.0, 0, 0], normal=(0, 1, 0), omega_world=(0, -150.0, 0)), cfg.cap, cfg.environment)
    assert res.magnus_world_n[2] > 0


def test_reversing_spin_reverses_magnus(setup):
    cfg, model = setup
    a = model.evaluate(_state([16.0, 1.0, 0.5], omega_world=(10.0, 20.0, 150.0)), cfg.cap, cfg.environment)
    b = model.evaluate(_state([16.0, 1.0, 0.5], omega_world=(-10.0, -20.0, -150.0)), cfg.cap, cfg.environment)
    assert np.allclose(a.magnus_world_n, -b.magnus_world_n, atol=1e-12)
    assert np.linalg.norm(a.magnus_world_n) > 0


def test_spin_parallel_to_velocity_gives_no_magnus(setup):
    cfg, model = setup
    res = model.evaluate(_state([16.0, 0, 0], omega_world=(300.0, 0, 0)), cfg.cap, cfg.environment)
    assert np.linalg.norm(res.magnus_world_n) < 1e-12
    assert res.diagnostics["spin_parameter"] == pytest.approx(0.0, abs=1e-12)


def test_zero_velocity_produces_no_nan(setup):
    cfg, model = setup
    res = model.evaluate(_state([0.0, 0, 0], omega_world=(0, 0, 100.0)), cfg.cap, cfg.environment)
    for arr in (res.force_world_n, res.moment_body_nm, res.drag_world_n, res.lift_world_n, res.magnus_world_n):
        assert np.all(np.isfinite(arr))


def test_face_on_lift_degeneracy_is_safe(setup):
    cfg, model = setup
    res = model.evaluate(_state([16.0, 0, 0], normal=(1, 0, 0)), cfg.cap, cfg.environment)
    assert np.all(np.isfinite(res.force_world_n))
    assert np.linalg.norm(res.lift_world_n) == pytest.approx(0.0, abs=1e-12)
    assert res.diagnostics["lift_degenerate_face_on"] is True


def test_toggles_disable_forces(base_raw):
    base_raw["aerodynamics"]["enable"] = {"drag": False, "lift": False, "magnus": False, "moments": False}
    cfg = config_from_dict(base_raw)
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    res = model.evaluate(_state([15.0, 0, -1.0], omega_world=(0, 0, 100.0)), cfg.cap, cfg.environment)
    assert not res.force_world_n.any() and not res.moment_body_nm.any()


def test_lift_coefficient_stall_modes(base_raw):
    cfg = config_from_dict(base_raw)
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    stall = np.deg2rad(40.0)
    assert model.lift_coefficient(np.deg2rad(20)) == pytest.approx(1.4 * np.deg2rad(20))
    assert model.lift_coefficient(-np.deg2rad(20)) == pytest.approx(-1.4 * np.deg2rad(20))
    assert model.lift_coefficient(np.pi / 2) == pytest.approx(0.0, abs=1e-12)
    assert model.lift_coefficient(stall) == pytest.approx(1.4 * stall)


def test_pitch_moment_sign_increases_alpha(base_raw):
    base_raw["aerodynamics"]["moments"] = {"cm_pitch0": 0.1}
    cfg = config_from_dict(base_raw)
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    state = _state([15.0, 0, 0])  # 面水平・α=0
    res = model.evaluate(state, cfg.cap, cfg.environment)
    # 正のピッチモーメントの下で微小回転させると α が増えること
    from cap_flight_sim.coordinates import quat_from_axis_angle, quat_multiply

    dq = quat_from_axis_angle(res.moment_body_nm, 1e-3)
    rotated = RigidBodyState(0.0, state.position_world_m, state.velocity_world_m_s,
                             quat_multiply(state.orientation_body_to_world, dq), np.zeros(3))
    res2 = model.evaluate(rotated, cfg.cap, cfg.environment)
    assert res2.diagnostics["angle_of_attack_deg"] > 0
