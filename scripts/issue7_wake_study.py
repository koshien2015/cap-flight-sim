"""Issue #7: 低速・低回転キャップの横揺れの診断と、後流の非定常項（実験的・既定 OFF）の比較。

    uv run python scripts/issue7_wake_study.py            # output/issue7_wake_study/ に CSV・JSON・図を書く
    uv run python scripts/issue7_wake_study.py --quick    # 動作確認用の少数ケース

調べること:
  Phase 1  復原モーメント（圧力中心のずれ）を入れると面が 1 回転する条件と理由。減衰・腕の長さ・ピッチ係数の感度。
           姿勢の乱れだけで「ゆらゆら」した横揺れが出るか。
  Phase 2  後流の非定常な横力 F_w だけ／モーメント M_w だけ／両方を足したときの横揺れと姿勢。
  Phase 3  通常球のプリセットに足したときの影響、回転数への依存。

方針は scripts/phase2_sensitivity.py と同じ: 固定刻み RK4、係数はすべて assumed（実測ではない）。
姿勢は 2 通りを区別して扱う。
  face_up      : 円形面を概ね上向き（前縁を 10° 上げる）。Issue #7 の観察に対応
  face_forward : 閉じた天面を打者へ正対・前縁を 30° 上げる。Issue #1 で使ったナックルのリリース姿勢
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cap_flight_sim.aerodynamics.factory import build_aerodynamic_model  # noqa: E402
from cap_flight_sim.aerodynamics.wake import WakeParams  # noqa: E402
from cap_flight_sim.config import apply_overrides, config_from_dict, load_raw_config  # noqa: E402
from cap_flight_sim.coordinates import euler_deg_to_quat, quat_to_matrix  # noqa: E402
from cap_flight_sim.models import RigidBodyState  # noqa: E402
from cap_flight_sim.simulation import run_simulation  # noqa: E402
from phase2_sensitivity import max_angle_change_deg, pitch_rotation_deg, value_at  # noqa: E402

OUT_DIR = ROOT / "output/issue7_wake_study"
OUTPUT_HZ = 1000
FIXED_STEP_SEC = 1e-4
VY_DEADBAND_M_S = 1e-3  # これより遅い横速度は「止まっている」とみなし、向きの反転に数えない
CP_ON = [0.0, 0.0, 0.0]  # ビューアの「重心のずれによる復原モーメント」ON の既定（空力中心 = 幾何中心）
WAKE_FORCE = 0.15
WAKE_MOMENT = 0.001

ATTITUDES: dict[str, dict[str, Any]] = {
    "face_up": {
        "position_world_m": [0.0, 0.0, 1.45], "speed_kmh": 40.0, "launch_azimuth_deg": 0.0, "launch_elevation_deg": 5.0,
        "orientation_euler_deg": [0.0, -10.0, 0.0], "angular_velocity": {"rpm": 0, "axis_body": [0.0, 0.0, 1.0]},
    },
    "face_forward": {
        "position_world_m": [0.0, 0.0, 1.45], "speed_kmh": 50.0, "launch_azimuth_deg": 0.0, "launch_elevation_deg": 2.0,
        "orientation_euler_deg": [0.0, 60.0, 0.0], "angular_velocity": {"rpm": 0, "axis_body": [0.0, 0.0, 1.0]},
    },
}
PRESETS = ("overhand_drop", "flick_flat", "sidearm_flat", "spec_example_tumble")


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    base: str = "face_up"  # ATTITUDES のキー、または examples/ のプリセット名
    overrides: dict[str, Any] = field(default_factory=dict)
    keep_series: bool = False
    tags: dict[str, Any] = field(default_factory=dict)


def build_raw(case: Case) -> dict[str, Any]:
    raw = load_raw_config(ROOT / "examples" / f"{case.base if case.base in PRESETS else 'overhand_drop'}.yaml")
    common = {"simulation.solver": "rk4", "simulation.fixed_step_sec": FIXED_STEP_SEC, "simulation.output_hz": OUTPUT_HZ}
    if case.base in ATTITUDES:
        common["initial_state"] = ATTITUDES[case.base]
    return apply_overrides(apply_overrides(raw, common), case.overrides)


def wake(force: float = WAKE_FORCE, moment: float = 0.0, seed: int = 1, **extra: Any) -> dict[str, Any]:
    return {"aerodynamics.wake": {"force_coefficient": force, "moment_coefficient": moment, "seed": seed, **extra}}


def moments(cp: list[float] | None = None, tumble: float = 0.0, cm_alpha: float = 0.0) -> dict[str, Any]:
    return {"aerodynamics.moments.center_of_pressure_body_m": cp, "aerodynamics.moments.tumble_damping": tumble,
            "aerodynamics.moments.cm_pitch_alpha_per_rad": cm_alpha}


# ------------------------------------------------------------ 計測（純粋関数。tests/test_issue7_study.py で検証）


def detrended_peak_to_peak(t: np.ndarray, y: np.ndarray) -> float:
    """直線（最小二乗）を引いた残りの最大幅。一方向に流れるだけなら小さく、揺れていれば大きい。"""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t.size < 3:
        return 0.0
    residual = y - np.polyval(np.polyfit(t, y, 1), t)
    return float(residual.max() - residual.min())


def direction_reversals(v: np.ndarray, deadband: float = VY_DEADBAND_M_S) -> int:
    """速度の向きが入れ替わった回数。|v| < deadband の区間は向きなしとして飛ばす。"""
    signs = np.sign(np.asarray(v, dtype=float)[np.abs(v) >= deadband])
    return int(np.count_nonzero(np.diff(signs)))


# ------------------------------------------------------------ 実行


def run_case(case: Case) -> dict[str, Any]:
    result = run_simulation(config_from_dict(build_raw(case), name=case.name))
    frames = result.frames
    t = np.array([f.state.time_sec for f in frames])
    pos = np.array([f.state.position_world_m for f in frames])
    vel = np.array([f.state.velocity_world_m_s for f in frames])
    normals = np.array([f.normal_world for f in frames])
    omega = np.array([f.state.angular_velocity_body_rad_s for f in frames])
    tilt = np.degrees(np.arccos(np.clip(normals @ normals[0], -1.0, 1.0)))
    row = {
        "name": case.name, "group": case.group, "base": case.base, **case.tags,
        "termination": result.termination_reason,
        "flight_time_s": float(t[-1]),
        "end_x_m": float(pos[-1, 0]), "end_y_mm": float(pos[-1, 1] * 1e3), "end_z_m": float(pos[-1, 2]),
        "y_abs_max_mm": float(np.max(np.abs(pos[:, 1])) * 1e3),
        "y_detrended_p2p_mm": detrended_peak_to_peak(t, pos[:, 1]) * 1e3,
        "vy_reversals": direction_reversals(vel[:, 1]),
        "max_normal_change_deg": max_angle_change_deg(normals),
        "pitch_rotation_deg": pitch_rotation_deg(normals),
        "normal_z_min": float(normals[:, 2].min()),
        "max_omega_inplane_rad_s": float(np.max(np.hypot(omega[:, 0], omega[:, 1]))),
    }
    out: dict[str, Any] = {"row": row}
    if case.keep_series:
        comp = {k: np.array([quat_to_matrix(f.state.orientation_body_to_world) @ f.aero.moment_components_body_nm[k]
                             for f in frames])[:, 1] for k in frames[0].aero.moment_components_body_nm}
        out["series"] = {
            "t": t.tolist(), "x": pos[:, 0].tolist(), "y": pos[:, 1].tolist(), "z": pos[:, 2].tolist(),
            "tilt": tilt.tolist(),
            "pitch": np.degrees(np.unwrap(np.arctan2(normals[:, 2], normals[:, 0]))).tolist(),
            "normal_to_wind": [f.aero.diagnostics["normal_to_wind_deg"] for f in frames],
            "omega_inplane": np.hypot(omega[:, 0], omega[:, 1]).tolist(),
            **{f"moment_y_{k}": v.tolist() for k, v in comp.items()},
        }
    return out


def static_pitch_moment(offsets_deg: np.ndarray, cp: list[float] | None, cm_alpha: float = 0.0) -> list[float]:
    """面法線を進行方向から上へ offset だけ傾けた姿勢での world Y まわりの空力モーメント（40 km/h・無回転）。

    offset = 0 は閉じた天面が進行方向へ正対、90° は面が上向き、180° は開口が進行方向へ正対。
    + は面法線を下げる向きなので、M > 0 なら offset を減らす（天面を前へ向ける）。
    """
    cfg = config_from_dict(build_raw(Case("static", "static", overrides=moments(cp, cm_alpha=cm_alpha))))
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    v = cfg.initial_state.velocity_world_m_s
    wind_elev = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    out = []
    for offset in offsets_deg:
        q = euler_deg_to_quat(np.array([0.0, 90.0 - (wind_elev + offset), 0.0]))
        res = model.evaluate(RigidBodyState(0.0, np.zeros(3), v, q, np.zeros(3)), cfg.cap, cfg.environment)
        out.append(float((quat_to_matrix(q) @ res.moment_body_nm)[1]))
    return out


def equilibria(offsets_deg: np.ndarray, moment: list[float]) -> list[dict[str, Any]]:
    """モーメントが 0 を横切る角度と、そこで安定か（傾きが正 = 姿勢を戻す向き）。"""
    m = np.asarray(moment, dtype=float)
    x = np.asarray(offsets_deg, dtype=float)
    nonzero = np.flatnonzero(m != 0.0)
    found = []
    for i, j in zip(nonzero[:-1], nonzero[1:]):
        if m[i] * m[j] > 0.0:
            continue
        # 隣り合う 2 点なら線形補間、間にちょうど 0 の点があればその中央
        a = x[i] - m[i] * (x[j] - x[i]) / (m[j] - m[i]) if j == i + 1 else float(np.mean(x[i + 1:j]))
        found.append({"offset_deg": float(a), "stable": bool(m[j] > m[i])})
    return found


def build_cases(quick: bool = False) -> list[Case]:
    seeds = range(1, 4) if quick else range(1, 21)
    cases = [
        # --- Phase 1: 1 回転の再現
        Case("face_up/no_moment", "flip", keep_series=True),
        Case("face_up/cp_on", "flip", overrides=moments(CP_ON), keep_series=True),
        Case("face_forward/cp_on", "flip", "face_forward", moments(CP_ON), keep_series=True),
    ]
    # --- Phase 1: 減衰と腕の長さ（空力中心の body Z。重心は +2.5 mm にあるので 0 mm で腕 −2.5 mm）
    for cp_z in (1.5, 0.0, -2.5) if not quick else (0.0,):
        for td in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0) if not quick else (0.0, 2.0):
            cases.append(Case(f"damping/cp_z={cp_z}/td={td}", "damping", overrides=moments([0.0, 0.0, cp_z * 1e-3], td),
                              tags={"cp_z_mm": cp_z, "tumble_damping": td}))
    # --- Phase 1: ピッチモーメント係数（面が上向きの姿勢では α ≈ 0 のまわり。負で復元）
    for cm in (-0.2, -0.05, -0.01, 0.01, 0.05):
        for td in (0.0, 2.0):
            cases.append(Case(f"cm_pitch/cm={cm}/td={td}", "cm_pitch", overrides=moments(None, td, cm),
                              tags={"cm_pitch_alpha_per_rad": cm, "tumble_damping": td}))
    # --- Phase 1: 姿勢の乱れだけで横に揺れるか
    perturbations = {
        "bank_5deg": {"initial_state.orientation_euler_deg": [5.0, -10.0, 0.0]},
        "yaw_5deg": {"initial_state.orientation_euler_deg": [0.0, -10.0, 5.0]},
        "roll_rate_1": {"initial_state.angular_velocity": {"omega_world_rad_s": [1.0, 0.0, 0.0]}},
        "yaw_rate_1": {"initial_state.angular_velocity": {"omega_world_rad_s": [0.0, 0.0, 1.0]}},
        "spin_30rpm_bank_5deg": {"initial_state.orientation_euler_deg": [5.0, -10.0, 0.0],
                                 "initial_state.angular_velocity": {"rpm": 30, "axis_body": [0.0, 0.0, 1.0]}},
    }
    moment_sets = {"no_moment": moments(), "cp_on": moments(CP_ON), "cp_on_td2": moments(CP_ON, 2.0),
                   "cm-0.05": moments(None, 0.0, -0.05), "cm-0.05_td2": moments(None, 2.0, -0.05)}
    for p_name, p in perturbations.items():
        for m_name, m in moment_sets.items():
            cases.append(Case(f"attitude_only/{p_name}/{m_name}", "attitude_only", overrides={**m, **p},
                              tags={"perturbation": p_name, "moments": m_name}))
    # --- Phase 2: 横力だけ（複数シード・2 姿勢）
    for base in ATTITUDES:
        for seed in seeds:
            cases.append(Case(f"wake_force/{base}/seed={seed}", "wake_force", base, wake(seed=seed),
                              keep_series=seed <= 6, tags={"seed": seed}))
    # --- Phase 2: 係数・速度への比例
    for c in (0.05, 0.1, 0.15, 0.3):
        cases.append(Case(f"wake_scale/C={c}", "wake_scale", overrides=wake(force=c, seed=3), tags={"force_coefficient": c}))
    for kmh in (30.0, 40.0, 50.0):
        cases.append(Case(f"wake_speed/v={kmh}", "wake_speed", overrides={**wake(seed=3), "initial_state.speed_kmh": kmh},
                          tags={"speed_kmh": kmh}))
    # --- Phase 2: 横力だけ／モーメントだけ／両方（モーメントは減衰つき・減衰なし・復原モーメントつきで比べる）
    variants = {
        "force_only": wake(),
        "moment_only_td2": {**wake(0.0, WAKE_MOMENT), **moments(None, 2.0)},
        "both_td2": {**wake(WAKE_FORCE, WAKE_MOMENT), **moments(None, 2.0)},
        "both_td0": {**wake(WAKE_FORCE, WAKE_MOMENT), **moments()},
        "both_cp_on_td2": {**wake(WAKE_FORCE, WAKE_MOMENT), **moments(CP_ON, 2.0)},
        "both_cm-0.05_td2": {**wake(WAKE_FORCE, WAKE_MOMENT), **moments(None, 2.0, -0.05)},
    }
    for v_name, v in variants.items():
        for seed in list(seeds)[:10]:
            w = {**v["aerodynamics.wake"], "seed": seed}
            cases.append(Case(f"wake_terms/{v_name}/seed={seed}", "wake_terms", overrides={**v, "aerodynamics.wake": w},
                              keep_series=seed == 3, tags={"variant": v_name, "seed": seed}))
    # --- Phase 3: 回転数への依存（横力は on − off の差で見る）
    for rpm in (0, 30, 100, 300, 1000, 1500):
        spin = {"initial_state.angular_velocity": {"rpm": rpm, "axis_body": [0.0, 0.0, 1.0]}}
        for label, w in (("off", {}), ("on", wake(seed=3))):
            cases.append(Case(f"spin/rpm={rpm}/{label}", "spin", overrides={**spin, **w}, tags={"rpm": rpm, "wake": label}))
    # --- Phase 3: 通常球のプリセットに足したときの影響
    for preset in PRESETS:
        cases.append(Case(f"preset/{preset}/off", "preset", preset, tags={"preset": preset, "wake": "off"}))
        for seed in list(seeds)[:10]:
            cases.append(Case(f"preset/{preset}/seed={seed}", "preset", preset, wake(seed=seed),
                              tags={"preset": preset, "wake": "on", "seed": seed}))
    return cases


def preset_differences(cases: list[Case], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """通常球: 後流の項を足したときの終了位置のずれ（off との差）。終了は捕手面か着地。"""
    rows = {c.name: r["row"] for c, r in zip(cases, results) if c.group == "preset"}
    out = []
    for preset in PRESETS:
        off = rows[f"preset/{preset}/off"]
        on = [r for n, r in rows.items() if n.startswith(f"preset/{preset}/seed=")]
        dy = np.array([r["end_y_mm"] - off["end_y_mm"] for r in on])
        dz = np.array([(r["end_z_m"] - off["end_z_m"]) * 1e3 for r in on])
        dx = np.array([(r["end_x_m"] - off["end_x_m"]) * 1e3 for r in on])
        out.append({"preset": preset, "termination": off["termination"], "seeds": len(on),
                    "same_termination": all(r["termination"] == off["termination"] for r in on),
                    "dy_mm_rms": float(np.sqrt(np.mean(dy**2))), "dy_mm_max": float(np.max(np.abs(dy))),
                    "dz_mm_max": float(np.max(np.abs(dz))), "dx_mm_max": float(np.max(np.abs(dx))),
                    "dnormal_deg_max": float(max(abs(r["max_normal_change_deg"] - off["max_normal_change_deg"]) for r in on))})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="動作確認用の少数ケースだけ計算する")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cases = build_cases(args.quick)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, cases))
    args.output.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    table = pd.DataFrame([r["row"] for r in results])
    table.to_csv(args.output / "runs.csv", index=False)
    series = {c.name: r["series"] for c, r in zip(cases, results) if "series" in r}
    (args.output / "series.json").write_text(json.dumps(series), encoding="utf-8")

    offsets = np.arange(0.0, 180.5, 1.0)
    static = {"offset_deg": offsets.tolist()}
    checks: dict[str, Any] = {"static_pitch_moment": static, "equilibria": {}}
    for name, (cp, cm) in {"cp_on": (CP_ON, 0.0), "cm_alpha_-0.05": (None, -0.05)}.items():
        static[name] = static_pitch_moment(offsets, cp, cm)
        # 0° と 180°（正対）は揚力の向きが退化するので、平衡点は 1°〜179° で探す
        checks["equilibria"][name] = equilibria(offsets[1:-1], static[name][1:-1])
    checks["preset_differences"] = preset_differences(cases, results)
    cfg = config_from_dict(build_raw(Case("ref", "ref")))
    checks["reference"] = {
        "face_up_speed_m_s": float(np.linalg.norm(cfg.initial_state.velocity_world_m_s)),
        "transverse_inertia_kg_m2": float(cfg.cap.inertia_tensor_body_kg_m2[0, 0]),
        "weight_n": cfg.cap.mass_kg * cfg.environment.gravity_m_s2,
        "band_wavelength_m": [cfg.cap.outer_diameter_m / k for k in WakeParams().reduced_frequencies],
        "band_hz_at_release": [k * float(np.linalg.norm(cfg.initial_state.velocity_world_m_s)) / cfg.cap.outer_diameter_m
                               for k in WakeParams().reduced_frequencies],
    }
    (args.output / "checks.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8")
    if not args.quick:
        from issue7_plots import write_figures  # noqa: PLC0415  同じ scripts/ 内

        write_figures(table, series, checks, args.output)
    print(f"{len(cases)} 条件 → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
