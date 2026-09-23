import numpy as np
import pytest

from cap_flight_sim.config import config_from_dict
from cap_flight_sim.simulation import run_simulation

ALL_OFF = {"drag": False, "lift": False, "magnus": False, "moments": False}


def _positions(result):
    return np.array([f.state.time_sec for f in result.frames]), np.array([f.state.position_world_m for f in result.frames])


def test_no_aero_matches_projectile(base_raw):
    base_raw["aerodynamics"]["enable"] = ALL_OFF
    base_raw["simulation"]["termination"] = {"stop_at_catcher_plane": False}
    result = run_simulation(config_from_dict(base_raw))
    t, pos = _positions(result)
    s0 = result.config.initial_state
    expected = s0.position_world_m + np.outer(t, s0.velocity_world_m_s) - np.outer(0.5 * 9.80665 * t**2, [0, 0, 1])
    assert np.allclose(pos, expected, atol=1e-6)
    assert result.termination_reason == "ground"


def test_no_gravity_no_aero_is_uniform_motion(base_raw):
    base_raw["aerodynamics"]["enable"] = ALL_OFF
    base_raw["environment"] = {"gravity_m_s2": 0.0}
    base_raw["simulation"] = {"duration_sec": 0.3}
    result = run_simulation(config_from_dict(base_raw))
    t, pos = _positions(result)
    s0 = result.config.initial_state
    assert np.allclose(pos, s0.position_world_m + np.outer(t, s0.velocity_world_m_s), atol=1e-8)


def test_drag_reduces_speed(base_raw):
    base_raw["aerodynamics"]["enable"] = {"drag": True, "lift": False, "magnus": False, "moments": False}
    base_raw["environment"] = {"gravity_m_s2": 0.0}
    base_raw["simulation"] = {"duration_sec": 0.3}
    result = run_simulation(config_from_dict(base_raw))
    speeds = [np.linalg.norm(f.state.velocity_world_m_s) for f in result.frames]
    assert np.all(np.diff(speeds) < 0)


def test_torque_free_angular_momentum_conserved_and_quaternion_norm(base_raw):
    base_raw["aerodynamics"]["enable"] = ALL_OFF
    base_raw["initial_state"]["angular_velocity"] = {"omega_body_rad_s": [30.0, -10.0, 200.0]}
    base_raw["simulation"] = {"duration_sec": 0.4}
    cfg = config_from_dict(base_raw)
    result = run_simulation(cfg)
    inertia = cfg.cap.inertia_tensor_body_kg_m2
    from cap_flight_sim.coordinates import quat_to_matrix

    momenta = [quat_to_matrix(f.state.orientation_body_to_world) @ (inertia @ f.state.angular_velocity_body_rad_s)
               for f in result.frames]
    h0 = momenta[0]
    assert max(np.linalg.norm(h - h0) for h in momenta) / np.linalg.norm(h0) < 1e-5
    energies = [0.5 * f.state.angular_velocity_body_rad_s @ inertia @ f.state.angular_velocity_body_rad_s for f in result.frames]
    assert max(abs(e - energies[0]) for e in energies) / energies[0] < 1e-5


def test_quaternion_norm_maintained_in_raw_integration(base_raw):
    """出力は正規化されるため、積分器内部の生の q ノルムを検査する。"""
    from cap_flight_sim.aerodynamics.factory import build_aerodynamic_model
    from cap_flight_sim.integrators import adaptive_segments
    from cap_flight_sim.rigid_body import make_state_derivative

    base_raw["initial_state"]["angular_velocity"] = {"omega_body_rad_s": [50.0, 20.0, 300.0]}
    cfg = config_from_dict(base_raw)
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    fun = make_state_derivative(cfg.cap, cfg.environment, lambda s: model.evaluate(s, cfg.cap, cfg.environment), 10.0)
    norms = [np.linalg.norm(seg.interp(seg.t1)[6:10])
             for seg in adaptive_segments(fun, 0.0, cfg.initial_state.to_vector(), 0.5, "DOP853", 1e-7, 1e-9, 1e-3)]
    assert max(abs(n - 1.0) for n in norms) < 1e-6


def test_rk4_agrees_with_adaptive(base_raw):
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 1500, "axis_body": [0, 0, 1]}
    base_raw["simulation"] = {"duration_sec": 0.3, "termination": {"stop_at_ground": False}}
    a = run_simulation(config_from_dict(base_raw))
    base_raw["simulation"].update({"solver": "rk4", "fixed_step_sec": 2e-4})
    b = run_simulation(config_from_dict(base_raw))
    assert np.allclose(a.frames[-1].state.position_world_m, b.frames[-1].state.position_world_m, atol=1e-5)


def test_reversed_spin_mirrors_lateral_break(base_raw):
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 1500, "axis_world": [0, 0, 1]}
    a = run_simulation(config_from_dict(base_raw))
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 1500, "axis_world": [0, 0, -1]}
    b = run_simulation(config_from_dict(base_raw))
    ya, yb = a.frames[-1].state.position_world_m[1], b.frames[-1].state.position_world_m[1]
    assert ya > 0.01
    assert ya == pytest.approx(-yb, rel=1e-6, abs=1e-9)
