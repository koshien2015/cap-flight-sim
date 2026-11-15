"""Issue #1 Phase 1: 姿勢安定性の診断値（力学計算は変えない）。"""

from pathlib import Path

import numpy as np
import pytest

from cap_flight_sim.aerodynamics.factory import build_aerodynamic_model
from cap_flight_sim.config import ConfigError, apply_overrides, config_from_dict, load_config
from cap_flight_sim.coordinates import quat_from_face_normal, world_to_body
from cap_flight_sim.models import RigidBodyState
from cap_flight_sim.outputs.serializers import attitude_diagnostics, timeseries_dataframe
from cap_flight_sim.simulation import run_simulation

MOMENTS_ALL = {
    "center_of_pressure_body_m": [0.002, -0.001, 0.003],
    "cm_pitch0": -0.05,
    "cm_pitch_alpha_per_rad": 0.4,
    "spin_damping": 0.02,
    "tumble_damping": 0.05,
}
MOMENTS_NONE = {"center_of_pressure_body_m": None, "cm_pitch0": 0.0, "cm_pitch_alpha_per_rad": 0.0,
                "spin_damping": 0.0, "tumble_damping": 0.0}


def _state(velocity, normal=(0.3, 0.2, 0.9), omega_world=(20.0, -40.0, 150.0)):
    q = quat_from_face_normal(np.array(normal, dtype=float))
    return RigidBodyState(0.0, np.array([0.0, 0.0, 1.0]), np.array(velocity, dtype=float), q,
                          world_to_body(q, np.array(omega_world, dtype=float)))


def _evaluate(raw, state):
    cfg = config_from_dict(raw)
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    return model.evaluate(state, cfg.cap, cfg.environment)


# ------------------------------------------------------------ Reynolds 数


def test_reynolds_number_matches_issue_estimate(base_raw):
    """Issue #1 の概算: D=0.030m, V=50km/h, ρ=1.204, μ=1.81e-5 → Re≈2.8e4。"""
    raw = apply_overrides(base_raw, {"environment": {"air_density_kg_m3": 1.204, "dynamic_viscosity_pa_s": 1.81e-5}})
    res = _evaluate(raw, _state([50.0 / 3.6, 0.0, 0.0]))
    expected = 1.204 * (50.0 / 3.6) * 0.030 / 1.81e-5
    assert res.diagnostics["reynolds_number"] == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(2.77e4, rel=1e-2)


def test_reynolds_number_uses_airspeed_including_wind(base_raw):
    raw = apply_overrides(base_raw, {"environment.wind_velocity_world_m_s": [-2.0, 0.0, 0.0]})
    cfg = config_from_dict(raw)
    res = _evaluate(raw, _state([10.0, 0.0, 0.0]))
    nu = cfg.environment.dynamic_viscosity_pa_s / cfg.environment.air_density_kg_m3
    assert res.diagnostics["reynolds_number"] == pytest.approx(12.0 * 0.03 / nu, rel=1e-12)


def test_default_viscosity_is_air_at_20c(base_raw):
    assert config_from_dict(base_raw).environment.dynamic_viscosity_pa_s == pytest.approx(1.81e-5)


@pytest.mark.parametrize("value", [0.0, -1e-5, float("nan")])
def test_viscosity_must_be_positive(base_raw, value):
    with pytest.raises(ConfigError):
        config_from_dict(apply_overrides(base_raw, {"environment.dynamic_viscosity_pa_s": value}))


def test_stationary_reynolds_is_zero(base_raw):
    res = _evaluate(base_raw, _state([0.0, 0.0, 0.0]))
    assert res.diagnostics["reynolds_number"] == 0.0
    assert set(res.moment_components_body_nm) == {"center_of_pressure", "pitch", "damping"}


# ------------------------------------------------------------ モーメントの内訳


@pytest.mark.parametrize(
    ("component", "only"),
    [
        ("center_of_pressure", {"center_of_pressure_body_m": MOMENTS_ALL["center_of_pressure_body_m"]}),
        ("pitch", {"cm_pitch0": MOMENTS_ALL["cm_pitch0"], "cm_pitch_alpha_per_rad": MOMENTS_ALL["cm_pitch_alpha_per_rad"]}),
        ("damping", {"spin_damping": MOMENTS_ALL["spin_damping"], "tumble_damping": MOMENTS_ALL["tumble_damping"]}),
    ],
)
def test_each_moment_component_equals_single_term_model(base_raw, component, only):
    state = _state([14.0, -1.0, -2.0])
    full = _evaluate(apply_overrides(base_raw, {"aerodynamics.moments": MOMENTS_ALL}), state)
    single = _evaluate(apply_overrides(base_raw, {"aerodynamics.moments": {**MOMENTS_NONE, **only}}), state)
    # 他の項が厳密に 0 なので、単独モデルの合計とビット単位で一致する
    assert np.array_equal(full.moment_components_body_nm[component], single.moment_body_nm)
    assert np.any(single.moment_body_nm != 0.0)


def test_moment_components_sum_to_total(base_raw):
    """合計の足し算の順序（cp → pitch → damping）を変えていないことをビット単位で確かめる。"""
    res = _evaluate(apply_overrides(base_raw, {"aerodynamics.moments": MOMENTS_ALL}), _state([14.0, -1.0, -2.0]))
    c = res.moment_components_body_nm
    assert np.array_equal(((np.zeros(3) + c["center_of_pressure"]) + c["pitch"]) + c["damping"], res.moment_body_nm)


def test_moment_components_zero_when_moments_disabled(base_raw):
    raw = apply_overrides(base_raw, {"aerodynamics.moments": MOMENTS_ALL, "aerodynamics.enable.moments": False})
    res = _evaluate(raw, _state([14.0, -1.0, -2.0]))
    assert all(np.all(v == 0.0) for v in res.moment_components_body_nm.values())


# ------------------------------------------------------------ 時系列


def test_timeseries_has_attitude_diagnostics(base_raw):
    raw = apply_overrides(base_raw, {
        "aerodynamics.moments": MOMENTS_ALL,
        "initial_state.angular_velocity": {"omega_body_rad_s": [30.0, -20.0, 120.0]},
        "simulation.duration_sec": 0.05,
    })
    result = run_simulation(config_from_dict(raw))
    df = timeseries_dataframe(result)
    inertia = result.config.cap.inertia_tensor_body_kg_m2
    row = df.iloc[0]
    assert row["omega_normal"] == pytest.approx(120.0)
    assert row["omega_inplane"] == pytest.approx(np.hypot(30.0, -20.0))
    assert row["spin_axis_tilt_deg"] == pytest.approx(np.degrees(np.arctan2(np.hypot(30.0, -20.0), 120.0)))
    assert row["angular_momentum_normal"] == pytest.approx(inertia[2, 2] * 120.0)
    assert row["angular_momentum_inplane"] == pytest.approx(np.hypot(inertia[0, 0] * 30.0, inertia[1, 1] * -20.0))
    assert row["reynolds_number"] > 0
    frame = result.frames[0]
    for name, vec in frame.aero.moment_components_body_nm.items():
        assert [row[f"moment_{name}_{a}"] for a in "xyz"] == pytest.approx(list(vec))


def test_spin_axis_tilt_is_nan_without_rotation(base_raw):
    raw = apply_overrides(base_raw, {"simulation.duration_sec": 0.01})
    df = timeseries_dataframe(run_simulation(config_from_dict(raw)))
    assert np.isnan(df["spin_axis_tilt_deg"].iloc[0])


def test_lookup_table_reports_reynolds_and_table_moment():
    cfg = load_config(Path(__file__).resolve().parents[1] / "examples/lookup_table_example.yaml")
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    res = model.evaluate(_state([15.0, 0.0, -1.0]), cfg.cap, cfg.environment)
    assert res.diagnostics["reynolds_number"] > 0
    assert set(res.moment_components_body_nm) == {"table"}
    assert np.array_equal(res.moment_components_body_nm["table"], res.moment_body_nm)
    assert res.moment_components_body_nm["table"] is not res.moment_body_nm

    still = model.evaluate(_state([0.0, 0.0, 0.0]), cfg.cap, cfg.environment)
    assert still.diagnostics["reynolds_number"] == 0.0
    assert np.array_equal(still.moment_components_body_nm["table"], np.zeros(3))


def test_attitude_diagnostics_with_off_diagonal_inertia():
    inertia = np.array([[3.0, 0.4, -0.2], [0.4, 2.5, 0.3], [-0.2, 0.3, 4.0]]) * 1e-7
    omega = np.array([12.0, -7.0, 90.0])
    d = attitude_diagnostics(omega, inertia)
    momentum = inertia @ omega
    assert d["angular_momentum_normal"] == momentum[2]
    assert d["angular_momentum_inplane"] == pytest.approx(np.hypot(momentum[0], momentum[1]), rel=1e-15)


def test_spin_axis_tilt_for_reverse_spin_is_180():
    assert attitude_diagnostics(np.array([0.0, 0.0, -5.0]), np.eye(3))["spin_axis_tilt_deg"] == pytest.approx(180.0)
