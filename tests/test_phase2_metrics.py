"""scripts/phase2_sensitivity.py の計測関数と、2条件だけの実行確認。全スイープは重いので pytest では回さない。"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def p2():
    spec = importlib.util.spec_from_file_location("phase2_sensitivity", ROOT / "scripts/phase2_sensitivity.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass が自モジュールを引けるように登録する
    spec.loader.exec_module(module)
    return module


def test_normal_angles(p2):
    n = np.array([[1.0, 0.0, 0.0], [np.cos(np.radians(30)), 0.0, np.sin(np.radians(30))], [0.0, 1.0, 0.0]])
    elev, azim = p2.normal_angles_deg(n)
    assert elev == pytest.approx([0.0, 30.0, 0.0])
    assert azim == pytest.approx([0.0, 0.0, 90.0])


def test_value_at_interpolates_and_refuses_extrapolation(p2):
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 10.0, 30.0])
    assert p2.value_at(xs, ys, 1.5) == pytest.approx(20.0)
    assert np.isnan(p2.value_at(xs, ys, 2.5))
    assert np.isnan(p2.value_at(xs, ys, -0.1))


def test_extremum_period_of_sine_and_noise(p2):
    t = np.linspace(0.0, 1.0, 2001)
    assert p2.extremum_period_sec(t, 5.0 * np.sin(2 * np.pi * t / 0.25)) == pytest.approx(0.25, rel=1e-3)
    assert np.isnan(p2.extremum_period_sec(t, t))
    # 数値誤差程度のふらつきは揺れとして数えない
    assert np.isnan(p2.extremum_period_sec(t, 30.0 + 1e-12 * np.sin(2 * np.pi * t / 0.01)))


def test_pitch_rotation_detects_tumbling(p2):
    a = np.radians(np.linspace(30.0, 30.0 + 400.0, 500))
    normals = np.column_stack([np.cos(a), np.zeros_like(a), np.sin(a)])
    assert p2.pitch_rotation_deg(normals) == pytest.approx(400.0, rel=1e-6)


def test_max_angle_change(p2):
    n = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [np.sqrt(0.5), np.sqrt(0.5), 0.0]])
    assert p2.max_angle_change_deg(n) == pytest.approx(90.0)


def test_build_raw_knuckle_attitude(p2):
    from cap_flight_sim.config import config_from_dict
    from cap_flight_sim.coordinates import quat_to_matrix

    cfg = config_from_dict(p2.build_raw(p2.Case(name="k", tilt_deg=30.0, yaw_deg=10.0, cp_dz_mm=-2.5)))
    normal = quat_to_matrix(cfg.initial_state.orientation_body_to_world)[:, 2]
    elev, azim = p2.normal_angles_deg(normal[None, :])
    assert elev[0] == pytest.approx(30.0)
    assert azim[0] == pytest.approx(10.0)
    assert cfg.settings.solver == "rk4"
    cp = np.asarray(cfg.aerodynamics["moments"]["center_of_pressure_body_m"], dtype=float)
    assert (cp - cfg.cap.center_of_mass_body_m)[2] == pytest.approx(-2.5e-3)


def test_zero_rpm_yaw_perturbations_mirror_exactly(p2):
    """無回転では左右が鏡像になる（+ε と −ε の横変位は符号だけ逆）。"""
    base = p2.Case(name="b", cp_dz_mm=-2.5)
    plus = p2.run_case(replace(base, name="p", yaw_deg=1.0))
    minus = p2.run_case(replace(base, name="m", yaw_deg=-1.0))
    assert np.array_equal(np.array(plus["series"]["y"]), -np.array(minus["series"]["y"]))
    assert np.array_equal(np.array(plus["series"]["x"]), np.array(minus["series"]["x"]))


def _face_on_lift_slope(p2):
    """失速後（decay）の揚力係数の、正対（α=90°）での傾き [1/rad]。"""
    from cap_flight_sim.config import config_from_dict

    lift = config_from_dict(p2.build_raw(p2.Case(name="s"))).aerodynamics["lift"]
    cl_a = float(lift["cl_alpha_per_rad"]["value"] if isinstance(lift["cl_alpha_per_rad"], dict) else lift["cl_alpha_per_rad"])
    stall = np.radians(float(lift["stall_angle_deg"]["value"] if isinstance(lift["stall_angle_deg"], dict) else lift["stall_angle_deg"]))
    return cl_a * stall / (np.pi / 2 - stall)


def test_pitch_stiffness_sign_and_magnitude_near_face_on(p2):
    """正対付近の剛性 k ≈ −dz·qA·(Cd − dCl/dθ)。天面側（dz>0）で復元（k>0）、開口側で不安定。"""
    from cap_flight_sim.config import config_from_dict

    fwd = p2.pitch_stiffness(p2.Case(name="s", cp_dz_mm=2.5), 0.02)
    aft = p2.pitch_stiffness(p2.Case(name="s", cp_dz_mm=-2.5), 0.02)
    cfg = config_from_dict(p2.build_raw(p2.Case(name="s")))
    speed = np.linalg.norm(cfg.initial_state.velocity_world_m_s)
    qa = 0.5 * cfg.environment.air_density_kg_m3 * speed**2 * cfg.cap.reference_area_m2
    expected = 2.5e-3 * qa * (_face_on_lift_slope(p2) - 1.0)  # cd_projected = 1、正対の投影面積 = A_ref
    assert fwd["stiffness_nm_per_rad"] > 0 > aft["stiffness_nm_per_rad"]
    assert fwd["stiffness_nm_per_rad"] == pytest.approx(expected, rel=0.05)


def test_cm_pitch_alpha_is_discontinuous_at_face_on(p2):
    case = p2.Case(name="s", cm_pitch_alpha_per_rad=0.05)
    coarse, fine = p2.pitch_stiffness(case, 0.2), p2.pitch_stiffness(case, 0.02)
    assert fine["moment_plus_nm"] == pytest.approx(coarse["moment_plus_nm"], rel=0.01)
    assert fine["moment_plus_nm"] == pytest.approx(-fine["moment_minus_nm"])


def test_straight_rejects_knuckle_only_fields(p2):
    with pytest.raises(ValueError):
        p2.build_raw(p2.Case(name="s", pitch="straight", yaw_deg=1.0))
