"""後流の非定常横力・モーメント（Issue #7 の実験的モデル。既定 OFF）。"""

import numpy as np
import pytest

from cap_flight_sim.aerodynamics.factory import build_aerodynamic_model
from cap_flight_sim.aerodynamics.wake import (
    WakeParams,
    band_signal,
    mulberry32,
    phase_offsets,
    spin_attenuation,
    wake_basis,
)
from cap_flight_sim.config import apply_overrides, config_from_dict
from cap_flight_sim.coordinates import quat_from_face_normal
from cap_flight_sim.models import RigidBodyState
from cap_flight_sim.outputs.serializers import build_viewer_document, timeseries_dataframe
from cap_flight_sim.simulation import run_simulation

# 面を概ね上向き（前縁を 10° 上げる）・無回転・40 km/h。ナックル的な低速・低回転の検証条件
FACE_UP = {
    "initial_state": {
        "position_world_m": [0.0, 0.0, 1.45],
        "speed_kmh": 40.0,
        "launch_elevation_deg": 5.0,
        "face_normal_world": [-np.sin(np.radians(10.0)), 0.0, np.cos(np.radians(10.0))],
        "angular_velocity": {"rpm": 0.0, "axis_body": [0.0, 0.0, 1.0]},
    },
    "simulation.solver": "rk4",
    "simulation.fixed_step_sec": 1e-3,
}


def _raw(base_raw, **overrides):
    return apply_overrides(apply_overrides(base_raw, FACE_UP), overrides)


def _wake(force=0.15, moment=0.0, seed=1, **extra):
    return {"aerodynamics.wake": {"force_coefficient": force, "moment_coefficient": moment, "seed": seed, **extra}}


def _run(raw):
    result = run_simulation(config_from_dict(raw))
    pos = np.array([f.state.position_world_m for f in result.frames])
    normals = np.array([f.normal_world for f in result.frames])
    return result, pos, normals


def _path_length(result) -> float:
    pos = np.array([f.state.position_world_m for f in result.frames])
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def _evaluate(raw, velocity, normal=(0.0, 0.0, 1.0), omega_body=(0.0, 0.0, 0.0), phase=37.0):
    cfg = config_from_dict(raw)
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    state = RigidBodyState(
        0.0, np.zeros(3), np.array(velocity, dtype=float), quat_from_face_normal(np.array(normal, dtype=float)),
        np.array(omega_body, dtype=float), wake_phase_rad=phase,
    )
    return cfg, model.evaluate(state, cfg.cap, cfg.environment)


# ------------------------------------------------------------ 乱数・信号


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (1, [0.6270739405881613, 0.002735721180215478, 0.5274470399599522]),
        (42, [0.6011037519201636, 0.44829055899754167, 0.8524657934904099]),
        (4294967295, [0.8964226141106337, 0.189478256739676, 0.7156526781618595]),
    ],
)
def test_mulberry32_matches_the_javascript_reference(seed, expected):
    rng = mulberry32(seed)
    assert [rng() for _ in range(3)] == expected


def test_phase_offsets_are_deterministic_and_seed_dependent():
    assert phase_offsets(7, 3) == phase_offsets(7, 3)
    assert phase_offsets(7, 3) != phase_offsets(8, 3)
    offsets = phase_offsets(7, 3)
    assert len(offsets) == 4 and all(len(axis) == 3 for axis in offsets)
    assert all(0.0 <= v < 2.0 * np.pi for axis in offsets for v in axis)


def test_band_signal_has_unit_rms():
    params = WakeParams()
    offsets = phase_offsets(3, len(params.reduced_frequencies))[0]
    phases = np.linspace(0.0, 2.0e6, 400_001)
    values = np.array([band_signal(offsets, params.reduced_frequencies, params.weights, p) for p in phases[::40]])
    assert np.sqrt(np.mean(values**2)) == pytest.approx(1.0, abs=0.03)


def test_spin_attenuation_is_continuous_and_monotonic():
    s = np.linspace(0.0, 1.0, 2001)
    g = np.array([spin_attenuation(v, 0.05) for v in s])
    assert g[0] == 1.0
    assert np.all(np.diff(g) < 0.0)
    assert np.max(np.abs(np.diff(g))) < 0.01  # しきい値で飛ばない
    assert spin_attenuation(0.3, None) == 1.0  # None = 回転に依存しない


@pytest.mark.parametrize("u_hat", [[1.0, 0.0, 0.0], [0.6, -0.3, 0.74], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
def test_wake_basis_is_orthonormal_and_perpendicular_to_the_wind(u_hat):
    u = np.array(u_hat) / np.linalg.norm(u_hat)
    e1, e2 = wake_basis(u)
    for e in (e1, e2):
        assert np.linalg.norm(e) == pytest.approx(1.0)
        assert abs(np.dot(e, u)) < 1e-12
    assert abs(np.dot(e1, e2)) < 1e-12


def test_wake_basis_is_lateral_and_up_for_level_flight():
    e1, e2 = wake_basis(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(e1, [0.0, 1.0, 0.0])  # +Y = 投手から見て左
    assert np.allclose(e2, [0.0, 0.0, 1.0])


# ------------------------------------------------------------ 既定 OFF


def test_wake_is_off_by_default(base_raw):
    cfg = config_from_dict(base_raw)
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    assert model.uses_wake_phase is False
    res = model.evaluate(cfg.initial_state, cfg.cap, cfg.environment)
    assert res.wake_world_n is None
    assert "wake" not in res.moment_components_body_nm


@pytest.mark.parametrize("solver", [{"simulation.solver": "rk4", "simulation.fixed_step_sec": 1e-3}, {"simulation.solver": "solve_ivp"}])
def test_zero_coefficients_leave_the_trajectory_bit_identical(base_raw, solver):
    spin = {"initial_state.angular_velocity": {"rpm": 600.0, "axis_body": [0.0, 0.0, 1.0]},
            "aerodynamics.moments": {"center_of_pressure_body_m": [0.0, 0.0, 0.0], "tumble_damping": 0.05}}
    plain = apply_overrides(apply_overrides(base_raw, solver), spin)
    zeroed = apply_overrides(plain, _wake(force=0.0, moment=0.0, seed=5))
    a, b = run_simulation(config_from_dict(plain)), run_simulation(config_from_dict(zeroed))
    assert len(a.frames) == len(b.frames)
    for fa, fb in zip(a.frames, b.frames):
        assert np.array_equal(fa.state.to_vector(), fb.state.to_vector())
        assert np.array_equal(fa.aero.force_world_n, fb.aero.force_world_n)


def test_lookup_table_model_rejects_wake(base_raw, tmp_path):
    raw = apply_overrides(base_raw, {"aerodynamics.model": "lookup_table",
                                     "aerodynamics.lookup_table": {"path": str(tmp_path / "missing.csv")},
                                     **_wake()})
    cfg = config_from_dict(raw)
    with pytest.raises(ValueError, match="wake"):
        build_aerodynamic_model(cfg.aerodynamics, cfg.cap)


@pytest.mark.parametrize(
    "bad",
    [{"reduced_frequencies": [0.01, 0.02], "weights": [1.0]}, {"reduced_frequencies": []}, {"reduced_frequencies": [-0.01, 0.02, 0.03]},
     {"spin_parameter_scale": 0.0}, {"seed": 1.5}, {"force_coefficient": -0.1},
     {"force_coefficient": float("nan")}, {"force_coefficient": float("nan"), "moment_coefficient": 0.001},
     {"moment_coefficient": float("inf")}, {"force_coefficient": True}, {"force_coefficient": "0.1"},
     {"weights": ["1", "0.8", "0.5"]}, {"reduced_frequencies": 0.01}, {"spin_parameter_scale": float("nan")},
     {"force_coeficient": 0.1}],
)
def test_invalid_wake_parameters_are_rejected(base_raw, bad):
    bands = bad.get("reduced_frequencies")
    extra = {"weights": [1.0] * len(bands)} if isinstance(bands, list) and "weights" not in bad else {}
    cfg = config_from_dict(apply_overrides(base_raw, _wake(**{**extra, **bad})))
    with pytest.raises(ValueError):
        build_aerodynamic_model(cfg.aerodynamics, cfg.cap)


def test_wake_section_must_be_a_mapping(base_raw):
    cfg = config_from_dict(apply_overrides(base_raw, {"aerodynamics.wake": True}))
    with pytest.raises(ValueError, match="mapping"):
        build_aerodynamic_model(cfg.aerodynamics, cfg.cap)


# ------------------------------------------------------------ 力・モーメントの性質


def test_wake_force_is_perpendicular_to_the_wind_and_adds_to_the_total(base_raw):
    v = [11.0, 0.7, -0.4]
    _, off = _evaluate(base_raw, v, normal=(-0.2, 0.1, 0.97))
    _, on = _evaluate(apply_overrides(base_raw, _wake()), v, normal=(-0.2, 0.1, 0.97))
    assert np.linalg.norm(on.wake_world_n) > 0.0
    assert abs(np.dot(on.wake_world_n, v)) < 1e-15
    assert np.allclose(on.force_world_n, off.force_world_n + on.wake_world_n, rtol=0, atol=1e-18)
    for key in ("drag_world_n", "lift_world_n", "magnus_world_n"):
        assert np.array_equal(getattr(on, key), getattr(off, key))


def test_wake_force_acts_at_the_centre_of_mass(base_raw):
    """横力だけを足したとき、圧力中心のずれがあってもモーメントは変わらない（姿勢を直接回さない）。"""
    cp = {"aerodynamics.moments.center_of_pressure_body_m": [0.0, 0.0, 0.0]}
    v = [11.0, 0.7, -0.4]
    _, off = _evaluate(apply_overrides(base_raw, cp), v, normal=(-0.2, 0.1, 0.97))
    _, on = _evaluate(apply_overrides(base_raw, {**cp, **_wake()}), v, normal=(-0.2, 0.1, 0.97))
    assert np.array_equal(on.moment_body_nm, off.moment_body_nm)
    assert np.array_equal(on.moment_components_body_nm["wake"], np.zeros(3))


def test_wake_moment_is_a_separate_term_with_no_torque_about_the_wind_axis(base_raw):
    v = np.array([11.0, 0.7, -0.4])
    cfg, off = _evaluate(base_raw, v, normal=(-0.2, 0.1, 0.97))
    _, on = _evaluate(apply_overrides(base_raw, _wake(force=0.0, moment=0.002)), v, normal=(-0.2, 0.1, 0.97))
    assert np.array_equal(on.force_world_n, off.force_world_n)
    wake_moment = on.moment_components_body_nm["wake"]
    assert np.linalg.norm(wake_moment) > 0.0
    assert np.allclose(on.moment_body_nm, off.moment_body_nm + wake_moment, rtol=0, atol=1e-20)
    q = quat_from_face_normal(np.array([-0.2, 0.1, 0.97]))
    from cap_flight_sim.coordinates import quat_to_matrix

    assert abs(np.dot(quat_to_matrix(q) @ wake_moment, v)) < 1e-18


def test_wake_force_scales_with_dynamic_pressure(base_raw):
    raw = apply_overrides(base_raw, _wake())
    _, slow = _evaluate(raw, [8.0, 0.0, 0.0])
    _, fast = _evaluate(raw, [16.0, 0.0, 0.0])
    assert np.allclose(fast.wake_world_n, 4.0 * slow.wake_world_n, rtol=1e-12)


def test_wake_force_follows_the_projected_area(base_raw):
    raw = apply_overrides(base_raw, _wake())
    cfg, edge_on = _evaluate(raw, [11.0, 0.0, 0.0], normal=(0.0, 0.0, 1.0))
    _, face_on = _evaluate(raw, [11.0, 0.0, 0.0], normal=(1.0, 0.0, 0.0))
    ratio = (np.pi * cfg.cap.radius_m**2) / (cfg.cap.outer_diameter_m * cfg.cap.height_m)
    assert np.allclose(face_on.wake_world_n, ratio * edge_on.wake_world_n, rtol=1e-9)


def test_wake_amplitude_is_continuous_in_spin(base_raw):
    raw = apply_overrides(base_raw, _wake())
    rates = np.linspace(0.0, 400.0, 801)  # rad/s（0〜約3800 rpm）
    mags = np.array([np.linalg.norm(_evaluate(raw, [11.0, 0.0, 0.0], omega_body=(0.0, 0.0, w))[1].wake_world_n) for w in rates])
    assert np.all(np.diff(mags) < 0.0)
    assert np.max(np.abs(np.diff(mags))) < 0.02 * mags[0]
    assert mags[-1] < 0.02 * mags[0]


def test_phase_rate_is_one_cycle_per_diameter_travelled(base_raw):
    cfg, on = _evaluate(apply_overrides(base_raw, _wake()), [3.0, 4.0, 0.0])
    assert on.wake_phase_rate_rad_s == pytest.approx(2.0 * np.pi * 5.0 / cfg.cap.outer_diameter_m)


# ------------------------------------------------------------ 軌道


def test_same_seed_reproduces_and_different_seeds_differ(base_raw):
    _, a, _ = _run(_raw(base_raw, **_wake(seed=3)))
    _, b, _ = _run(_raw(base_raw, **_wake(seed=3)))
    _, c, _ = _run(_raw(base_raw, **_wake(seed=4)))
    assert np.array_equal(a, b)
    assert abs(a[-1, 1] - c[-1, 1]) > 1e-4


def test_force_only_wake_sways_sideways_without_turning_the_face(base_raw):
    _, off, _ = _run(_raw(base_raw))
    assert np.all(off[:, 1] == 0.0)  # 現行モデルは左右対称なら横に動かない
    for seed in range(1, 9):
        result, pos, normals = _run(_raw(base_raw, **_wake(seed=seed)))
        assert result.termination_reason in ("ground", "catcher_plane")
        assert np.all(np.isfinite(pos))
        assert 1e-3 < np.max(np.abs(pos[:, 1])) < 0.6
        assert np.allclose(normals, normals[0], atol=1e-12)  # 面は上向きのまま


def test_wake_moment_wobbles_but_does_not_flip_the_face(base_raw):
    damping = {"aerodynamics.moments.tumble_damping": 2.0}
    for seed in range(1, 9):
        result, pos, normals = _run(_raw(base_raw, **damping, **_wake(force=0.15, moment=0.001, seed=seed)))
        assert result.termination_reason in ("ground", "catcher_plane")
        change = np.degrees(np.arccos(np.clip(normals @ normals[0], -1.0, 1.0)))
        assert 0.1 < change.max() < 45.0
        assert normals[:, 2].min() > 0.5


def test_fixed_step_converges_and_matches_the_adaptive_solver(base_raw):
    wake = _wake(force=0.15, moment=0.001, seed=2)
    ends = {}
    for name, sim in {"h": {"simulation.fixed_step_sec": 2e-4}, "h/2": {"simulation.fixed_step_sec": 1e-4},
                      "dop853": {"simulation.solver": "solve_ivp"}}.items():
        result, pos, _ = _run(_raw(base_raw, **wake, **sim))
        ends[name] = (result.frames[-1].state.time_sec, pos[-1])
    assert np.allclose(ends["h"][1], ends["h/2"][1], atol=1e-7)
    assert ends["h"][0] == pytest.approx(ends["h/2"][0], abs=1e-8)
    assert np.allclose(ends["dop853"][1], ends["h/2"][1], atol=1e-4)


def test_phase_state_is_tracked_only_when_enabled(base_raw):
    off, _, _ = _run(_raw(base_raw))
    on, _, _ = _run(_raw(base_raw, **_wake()))
    assert all(f.state.wake_phase_rad == 0.0 for f in off.frames)
    phases = np.array([f.state.wake_phase_rad for f in on.frames])
    assert phases[0] == 0.0 and np.all(np.diff(phases) > 0.0)
    cfg = config_from_dict(_raw(base_raw, **_wake()))
    assert phases[-1] == pytest.approx(2.0 * np.pi * _path_length(on) / cfg.cap.outer_diameter_m, rel=1e-3)


# ------------------------------------------------------------ 出力


def test_outputs_carry_the_wake_force_only_when_enabled(base_raw):
    off, _, _ = _run(_raw(base_raw))
    on, _, _ = _run(_raw(base_raw, **_wake()))
    assert "wake_y" not in timeseries_dataframe(off).columns
    df = timeseries_dataframe(on)
    assert {"wake_x", "wake_y", "wake_z", "wake_phase", "moment_wake_x"} <= set(df.columns)
    assert df["wake_y"].abs().max() > 0.0
    assert "wake" not in build_viewer_document(off)["frames"][0]["forcesN"]
    assert len(build_viewer_document(on)["frames"][0]["forcesN"]["wake"]) == 3
    assert on.model_description["wake"]["status"] == "experimental_unvalidated"
