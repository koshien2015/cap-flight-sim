"""ブラウザ版（TypeScript 移植）の照合用ゴールデンとプリセットを生成する。

Python 実装が物理計算の正本。TS 版はこのファイルが出力する値と一致することを Vitest で検証する。
    uv run python scripts/generate_ts_goldens.py
出力:
    viewer/src/presets/presets.generated.json   … examples/*.yaml を継承展開した raw 設定
    viewer/tests/golden/derivatives.json         … 状態微分・空気力の内訳（手選びの状態）
    viewer/tests/golden/trajectories.json        … solver: rk4 の軌道（決定的なので TS と ~1e-8 m で一致する）
Python 側の物理を変えたら再生成し、TS 側を追従させること（tests/test_ts_goldens.py が古さを検出する）。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cap_flight_sim.aerodynamics.factory import build_aerodynamic_model  # noqa: E402
from cap_flight_sim.config import apply_overrides, config_from_dict, load_raw_config  # noqa: E402
from cap_flight_sim.coordinates import quat_from_axis_angle, quat_normalize  # noqa: E402
from cap_flight_sim.models import RigidBodyState  # noqa: E402
from cap_flight_sim.rigid_body import make_state_derivative  # noqa: E402
from cap_flight_sim.simulation import run_simulation  # noqa: E402

PRESET_OUT = ROOT / "viewer/src/presets/presets.generated.json"
GOLDEN_DIR = ROOT / "viewer/tests/golden"
BROWSER_UNSUPPORTED = ("lookup_table_example",)

BASE: dict[str, Any] = {
    "cap": {"mass_kg": 0.0022, "outer_diameter_m": 0.03, "height_m": 0.015},
    "initial_state": {
        "position_world_m": [0.0, 0.0, 1.45],
        "speed_m_s": 16.0,
        "face_normal_world": [0.0, 0.0, 1.0],
        "angular_velocity": {"rpm": 1200, "axis_body": [0.0, 0.0, 1.0]},
    },
    "aerodynamics": {
        "drag": {"cd_projected": 1.0},
        "lift": {"cl_alpha_per_rad": 1.4, "cl0": 0.05},
        "magnus": {"coefficient": 1.0, "max_coefficient": 0.5},
    },
}

AERO_VARIANTS: dict[str, dict[str, Any]] = {
    "projected_decay": {},
    "constant_hold": {"aerodynamics.drag": {"model": "constant", "cd_base": 0.8},
                      "aerodynamics.lift.post_stall": "hold", "aerodynamics.lift.stall_angle_deg": 20.0},
    "quadratic_constmagnus": {"aerodynamics.drag": {"model": "quadratic_aoa", "cd_base": 0.1, "cd_alpha2_per_rad2": 2.7},
                              "aerodynamics.magnus": {"model": "constant", "coefficient": 0.3, "max_coefficient": 0.2}},
    "moments_all": {"aerodynamics.moments": {"center_of_pressure_body_m": [0.002, -0.001, 0.003], "cm_pitch0": -0.05,
                                             "cm_pitch_alpha_per_rad": 0.4, "spin_damping": 0.02, "tumble_damping": 0.05}},
    "wind_cavity_solid": {"environment": {"wind_velocity_world_m_s": [-2.0, 1.0, 0.5], "air_density_kg_m3": 1.1,
                                          "gravity_m_s2": 9.7, "dynamic_viscosity_pa_s": 1.7e-5},
                          "cap.surface_orientation": "body_z_is_cavity", "cap.inertia_method": "solid_cylinder"},
    "thin_disk_all_off": {"cap.inertia_method": "thin_disk",
                          "aerodynamics.enable": {"drag": False, "lift": False, "magnus": False, "moments": False}},
    "only_magnus": {"aerodynamics.enable": {"drag": False, "lift": False, "magnus": True, "moments": True}},
    # 後流の非定常項（Issue #7 の実験的モデル）。有効なときだけ状態が 14 要素になり、wake の内訳が付く
    "wake_force_moment": {"aerodynamics.wake": {"force_coefficient": 0.15, "moment_coefficient": 0.002, "seed": 7},
                          "aerodynamics.moments": {"center_of_pressure_body_m": [0.0, 0.0, 0.0], "tumble_damping": 0.5}},
    "wake_custom_bands_no_spin_scale": {"aerodynamics.wake": {
        "force_coefficient": 0.3, "seed": 4294967295, "reduced_frequencies": [0.003, 0.05], "weights": [0.4, 1.3],
        "spin_parameter_scale": None}, "aerodynamics.enable": {"drag": True, "lift": True, "magnus": True, "moments": False}},
}
WAKE_PHASE_RAD = 1234.5  # 後流が有効な変種で照合に使う位相（状態ベクトルの 14 番目）


FACE_UP_LOW_SPIN: dict[str, Any] = {
    "initial_state": {
        "position_world_m": [0.0, 0.0, 1.45],
        "speed_kmh": 40.0,
        "launch_elevation_deg": 5.0,
        "orientation_euler_deg": [0.0, -10.0, 0.0],
        "angular_velocity": {"rpm": 30, "axis_body": [0.0, 0.0, 1.0]},
    },
}


def _q(axis, deg):
    return quat_from_axis_angle(np.array(axis, dtype=float), np.deg2rad(deg))


# (名前, 速度, クォータニオン, ω_body) — 分岐を網羅するよう手選び
STATES: list[tuple[str, list[float], np.ndarray, list[float]]] = [
    ("level_alpha0", [16, 0, 0], np.array([1.0, 0, 0, 0]), [0, 0, 150]),
    ("descending_alpha_pos", [15, 0.5, -2.0], _q([0, 1, 0], -3), [0, 0, 150]),
    ("climbing_alpha_neg", [15, 0, 3.0], _q([1, 0, 0], 5), [0, 0, -200]),
    ("beyond_stall", [8, 0, -7], _q([0, 1, 0], 10), [10, -5, 100]),
    ("steep_negative", [5, 1, 9], np.array([1.0, 0, 0, 0]), [0, 0, 0]),
    ("face_on", [16, 0, 0], _q([0, 1, 0], 90), [0, 0, 50]),
    ("edge_on_topspin", [16, 0, 0.3], _q([1, 0, 0], -90), [0, 0, 125]),
    ("tumbling", [14, -1, 1], quat_normalize(np.array([0.7, 0.2, -0.5, 0.4])), [80, -60, 20]),
    ("spin_parallel_velocity", [16, 0, 0], _q([0, 1, 0], 90), [0, 0, 300]),
    ("stationary", [0, 0, 0], np.array([1.0, 0, 0, 0]), [0, 0, 100]),
    ("slow_high_spin_clip", [2, 0, -0.5], _q([1, 0, 0], -90), [0, 0, 600]),
    ("general", [12.3, -2.1, 0.7], quat_normalize(np.array([0.3, -0.6, 0.2, 0.7])), [-40, 90, 250]),
]


def _floats(v) -> list[float]:
    return [float(x) for x in np.asarray(v).ravel()]


def _config_summary(cfg) -> dict[str, Any]:
    s = cfg.settings
    return {
        "centerOfMassBodyM": _floats(cfg.cap.center_of_mass_body_m),
        "inertiaBodyKgM2": _floats(cfg.cap.inertia_tensor_body_kg_m2),
        "referenceAreaM2": cfg.cap.reference_area_m2,
        "initialStateVector": _floats(cfg.initial_state.to_vector()),
        "maxStepSec": s.max_step_sec,
        "fixedStepSec": s.fixed_step_sec,
        "durationSec": s.duration_sec,
    }


def derivative_cases() -> list[dict[str, Any]]:
    cases = []
    for variant, overrides in AERO_VARIANTS.items():
        raw = apply_overrides(copy.deepcopy(BASE), overrides)
        cfg = config_from_dict(raw)
        model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
        fun = make_state_derivative(cfg.cap, cfg.environment, lambda st: model.evaluate(st, cfg.cap, cfg.environment),
                                    cfg.settings.quaternion_drift_gain)
        states = []
        for name, vel, q, omega in STATES:
            q_scaled = np.asarray(q, dtype=float) * 1.001  # ノルム≠1 のときのドリフト項も照合する
            y = np.concatenate([[0.1, -0.2, 1.3], vel, q_scaled, omega])
            phase = 0.0
            if getattr(model, "uses_wake_phase", False):
                phase = WAKE_PHASE_RAD
                y = np.append(y, phase)
            state = RigidBodyState(0.0, y[0:3], y[3:6], quat_normalize(y[6:10]), y[10:13], phase)
            aero = model.evaluate(state, cfg.cap, cfg.environment)
            states.append({
                "name": name,
                "y": _floats(y),
                "dydt": _floats(fun(0.0, y)),
                "force": _floats(aero.force_world_n),
                "moment": _floats(aero.moment_body_nm),
                "drag": _floats(aero.drag_world_n),
                "lift": _floats(aero.lift_world_n),
                "magnus": _floats(aero.magnus_world_n),
                **({} if aero.wake_world_n is None else {"wake": _floats(aero.wake_world_n)}),
                "momentComponents": {k: _floats(v) for k, v in aero.moment_components_body_nm.items()},
                "coefficients": {k: float(v) for k, v in aero.coefficients.items()},
                "diagnostics": {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                                for k, v in aero.diagnostics.items()},
            })
        cases.append({"variant": variant, "raw": raw, "config": _config_summary(cfg), "states": states})
    return cases


def trajectory_cases(presets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rk4 = {"simulation.solver": "rk4", "simulation.output_hz": 60}
    selected = {
        "overhand_drop": presets["overhand_drop"],
        "flick_flat": presets["flick_flat"],
        "spec_example_tumble": presets["spec_example_tumble"],
        "ballistic_no_aero": presets["ballistic_no_aero"],
        "moments_wind": apply_overrides(presets["flick_flat"], {
            "aerodynamics.moments": {"center_of_pressure_body_m": [0.001, 0.0, 0.0], "cm_pitch0": 0.02,
                                     "spin_damping": 0.01, "tumble_damping": 0.02},
            "environment.wind_velocity_world_m_s": [-1.0, 0.5, 0.0]}),
        "catcher_continue_to_ground": apply_overrides(presets["ballistic_no_aero"], {
            "simulation.termination.stop_at_catcher_plane": False}),
        "sidearm_flat": presets["sidearm_flat"],
        # ナックルのリリース姿勢（Issue #1 の検証域）: 閉じた天面を打者（+X）へ正対させ、前縁を上げ（面法線を 30° 上向き）、
        # 回転をかけない。傾き 30° は投手の感覚値（未実測）。プリセット一覧に出さないため inline で定義。
        "knuckle_face_forward_no_spin": apply_overrides(presets["spec_example_tumble"], {
            "initial_state.speed_kmh": 50.0,
            "initial_state.orientation_euler_deg": [0.0, 60.0, 0.0],
            "initial_state.angular_velocity": {"rpm": 0, "axis_body": [0.0, 0.0, 1.0]}}),
        # 後流の非定常項（Issue #7）: 面を概ね上向き（前縁 10° 上げ）・低回転。横力だけ／横力＋モーメントの 2 通り
        "face_up_wake_force": apply_overrides(presets["flick_flat"], {**FACE_UP_LOW_SPIN, "aerodynamics.wake": {
            "force_coefficient": 0.15, "seed": 3}}),
        "face_up_wake_force_moment": apply_overrides(presets["flick_flat"], {**FACE_UP_LOW_SPIN, "aerodynamics.wake": {
            "force_coefficient": 0.15, "moment_coefficient": 0.001, "seed": 5},
            "aerodynamics.moments.tumble_damping": 2.0}),
    }
    cases = []
    for name, raw in selected.items():
        raw = apply_overrides(raw, rk4)
        cfg = config_from_dict(raw, name=name)
        result = run_simulation(cfg)
        cases.append({
            "name": name,
            "raw": raw,
            "config": _config_summary(cfg),
            "terminationReason": result.termination_reason,
            # 着地・捕手面到達の状態も照合する（release / max_time はフレームを持たない）
            "events": [{"type": e["type"], "timeSec": e["timeSec"],
                        **({"position": _floats(e["frame"].state.position_world_m),
                            "normal": _floats(e["frame"].normal_world)} if "frame" in e else {})}
                       for e in result.events],
            "frames": [{
                "t": f.state.time_sec,
                "position": _floats(f.state.position_world_m),
                "velocity": _floats(f.state.velocity_world_m_s),
                "quaternion": _floats(f.state.orientation_body_to_world),
                "omegaBody": _floats(f.state.angular_velocity_body_rad_s),
                "force": _floats(f.aero.force_world_n),
            } for f in result.frames],
        })
    return cases


def load_presets() -> dict[str, dict[str, Any]]:
    presets = {}
    for path in sorted((ROOT / "examples").glob("[!_]*.yaml")):
        if path.stem in BROWSER_UNSUPPORTED:
            continue
        raw = load_raw_config(path)
        presets[path.stem] = {**raw, "name": raw.get("name", path.stem)}
    return presets


def build_all() -> dict[Path, Any]:
    presets = load_presets()
    return {
        PRESET_OUT: presets,
        GOLDEN_DIR / "derivatives.json": {"generator": "scripts/generate_ts_goldens.py", "cases": derivative_cases()},
        GOLDEN_DIR / "trajectories.json": {"generator": "scripts/generate_ts_goldens.py", "cases": trajectory_cases(presets)},
    }


def _finite_or_none(obj: Any) -> Any:
    """JSON は NaN/Inf を表せないため null にする（TS 側は null を NaN として照合）。"""
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite_or_none(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def render(data: Any) -> str:
    return json.dumps(_finite_or_none(data), ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def main() -> None:
    for path, data in build_all().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(data), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
