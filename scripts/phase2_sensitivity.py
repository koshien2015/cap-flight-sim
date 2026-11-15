"""Issue #1 Phase 2: 現行モデルの感度分析（力学モデルは変えない）。

ナックル（閉じた天面を打者へ正対・前縁を上げる・無回転）を基準に、速度・前縁の傾き・回転数・回転軸・
圧力中心・ピッチモーメント係数・減衰係数・左右の初期の乱れを1つずつ振り、姿勢と軌道がどう変わるかを調べる。
対比としてストレート（overhand_drop）も振る。

    uv run python scripts/phase2_sensitivity.py            # output/phase2_sensitivity/ に CSV・図・表を書く
    uv run python scripts/phase2_sensitivity.py --quick    # 動作確認用の少数ケース

方針:
- 積分は固定刻み RK4（DOP853 はタンブリング系で刻み制御の揺らぎが ~1e-4 出るため）。
- 着地時刻が条件で大きく変わるので、比較は着地点ではなく「同じ時刻」「同じ前方距離」で行う。
- 振った値はすべて assumed（実測ではない）。前縁 30° は投手の感覚値。
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cap_flight_sim.config import apply_overrides, config_from_dict, load_raw_config  # noqa: E402
from cap_flight_sim.coordinates import euler_deg_to_quat, quat_to_matrix  # noqa: E402
from cap_flight_sim.simulation import run_simulation  # noqa: E402

OUT_DIR = ROOT / "output/phase2_sensitivity"
OUTPUT_HZ = 1000  # 比較用の出力間隔（積分刻みとは別。積分は fixed_step_sec=1e-4）
TIMES_SEC = (0.1, 0.2, 0.3)
STATIONS_M = (2.0, 3.0)


# ------------------------------------------------------------ 条件


@dataclass(frozen=True)
class Case:
    """1 球の条件。既定値はナックルの基準（50km/h・前縁30°・無回転・圧力中心のずれなし）。"""

    name: str
    group: str = "knuckle"
    pitch: str = "knuckle"  # knuckle / straight
    speed_kmh: float = 50.0
    tilt_deg: float = 30.0  # 前縁を上げる角（面法線の仰角）
    yaw_deg: float = 0.0  # 面法線の方位の乱れ（+ = 投手から見て左）
    rpm: float = 0.0
    spin_axis_body: tuple[float, float, float] = (0.0, 0.0, 1.0)
    omega_extra_world: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 初期角速度の乱れ [rad/s]
    cp_dz_mm: float | None = None  # 圧力中心 − 重心（body Z, mm）。None = 一致
    cm_pitch_alpha_per_rad: float = 0.0
    spin_damping: float = 0.0
    tumble_damping: float = 0.0
    magnus: bool = True
    lift: bool = True
    fixed_step_sec: float = 1e-4
    tags: dict[str, Any] = field(default_factory=dict)


def build_raw(case: Case) -> dict[str, Any]:
    base = load_raw_config(ROOT / "examples/overhand_drop.yaml")
    com = config_from_dict(base).cap.center_of_mass_body_m
    if case.pitch == "straight" and (case.tilt_deg != 30.0 or case.yaw_deg != 0.0
                                     or case.spin_axis_body != (0.0, 0.0, 1.0) or any(case.omega_extra_world)):
        raise ValueError("straight は overhand_drop の姿勢・回転軸を使う。tilt/yaw/spin_axis/omega_extra は指定できない")
    overrides: dict[str, Any] = {
        "simulation.solver": "rk4",
        "simulation.fixed_step_sec": case.fixed_step_sec,
        "simulation.output_hz": OUTPUT_HZ,
        "aerodynamics.enable.magnus": case.magnus,
        "aerodynamics.enable.lift": case.lift,
        "aerodynamics.moments.cm_pitch_alpha_per_rad": case.cm_pitch_alpha_per_rad,
        "aerodynamics.moments.spin_damping": case.spin_damping,
        "aerodynamics.moments.tumble_damping": case.tumble_damping,
        "aerodynamics.moments.center_of_pressure_body_m": (
            None if case.cp_dz_mm is None else [float(x) for x in com + np.array([0.0, 0.0, case.cp_dz_mm * 1e-3])]
        ),
    }
    if case.pitch == "knuckle":
        q = euler_deg_to_quat(np.array([0.0, 90.0 - case.tilt_deg, case.yaw_deg]))
        spin = np.asarray(case.spin_axis_body, dtype=float)
        spin = spin / np.linalg.norm(spin) * case.rpm * 2.0 * np.pi / 60.0
        omega_body = spin + quat_to_matrix(q).T @ np.asarray(case.omega_extra_world, dtype=float)
        overrides["initial_state"] = {
            "position_world_m": [0.0, 0.0, 1.45],
            "speed_kmh": case.speed_kmh,
            "launch_azimuth_deg": 0.0,
            "launch_elevation_deg": 2.0,
            "orientation_quaternion_wxyz": [float(x) for x in q],
            "angular_velocity": {"omega_body_rad_s": [float(x) for x in omega_body]},
        }
    else:
        overrides["initial_state.speed_kmh"] = case.speed_kmh
        overrides["initial_state.angular_velocity"] = {"rpm": case.rpm, "axis_world": [0.0, 1.0, 0.0]}
    return apply_overrides(base, overrides)


# ------------------------------------------------------------ 計測（純粋関数。tests/test_phase2_metrics.py で検証）


def normal_angles_deg(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """面法線の (仰角, 方位角)。方位角は +Y（投手から見て左）向きを正。"""
    n = np.asarray(normals, dtype=float)
    elevation = np.degrees(np.arcsin(np.clip(n[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
    return elevation, azimuth


def value_at(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """単調増加の xs 上で ys を線形補間する。範囲外は NaN（着地後の値を作らない）。"""
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0 or x < xs[0] or x > xs[-1]:
        return float("nan")
    return float(np.interp(x, xs, np.asarray(ys, dtype=float)))


def extremum_period_sec(t: np.ndarray, angle: np.ndarray, min_excursion_deg: float = 0.1) -> float:
    """角度の極値の間隔の平均×2。揺れなら揺れの周期、縦に回り続けるなら（仰角が ±90° で折り返すため）半回転の時間×2。

    隣り合う極値の差が min_excursion_deg 未満のものは数値誤差として無視する。極値が2つ未満なら NaN。
    """
    a = np.asarray(angle, dtype=float)
    d = np.diff(a)
    idx = np.where((d[:-1] > 0) & (d[1:] <= 0) | (d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    kept: list[int] = []
    for i in idx:
        if abs(a[i] - a[kept[-1] if kept else 0]) >= min_excursion_deg:
            kept.append(int(i))
    if len(kept) < 2:
        return float("nan")
    return float(2.0 * np.mean(np.diff(np.asarray(t, dtype=float)[kept])))


def pitch_rotation_deg(normals: np.ndarray) -> float:
    """縦（X-Z 面内）の回転量。面法線の向き atan2(n_z, n_x) を連続化した最大振れ幅。180° を超えれば縦に回っている。"""
    n = np.asarray(normals, dtype=float)
    angle = np.degrees(np.unwrap(np.arctan2(n[:, 2], n[:, 0])))
    return float(np.max(np.abs(angle - angle[0])))


def max_angle_change_deg(normals: np.ndarray) -> float:
    n = np.asarray(normals, dtype=float)
    return float(np.degrees(np.max(np.arccos(np.clip(n @ n[0], -1.0, 1.0)))))


# ------------------------------------------------------------ 実行


def run_case(case: Case) -> dict[str, Any]:
    result = run_simulation(config_from_dict(build_raw(case), name=case.name))
    frames = result.frames
    t = np.array([f.state.time_sec for f in frames])
    pos = np.array([f.state.position_world_m for f in frames])
    normals = np.array([f.normal_world for f in frames])
    omega = np.array([f.state.angular_velocity_body_rad_s for f in frames])
    elevation, azimuth = normal_angles_deg(normals)
    row: dict[str, Any] = {
        **{k: v for k, v in asdict(case).items() if k != "tags"},
        **case.tags,
        "termination": result.termination_reason,
        "flight_time_s": float(t[-1]),
        "landing_x_m": float(pos[-1, 0]),
        "max_normal_change_deg": max_angle_change_deg(normals),
        "max_omega_inplane_rad_s": float(np.max(np.hypot(omega[:, 0], omega[:, 1]))),
        "extremum_period_s": extremum_period_sec(t, elevation),
        "pitch_rotation_deg": pitch_rotation_deg(normals),
        "reynolds_min": float(min(f.aero.diagnostics["reynolds_number"] for f in frames)),
        "reynolds_max": float(max(f.aero.diagnostics["reynolds_number"] for f in frames)),
    }
    for ts in TIMES_SEC:
        row[f"y_mm@{ts}s"] = value_at(t, pos[:, 1], ts) * 1e3
        row[f"elev_deg@{ts}s"] = value_at(t, elevation, ts)
        row[f"azim_deg@{ts}s"] = value_at(t, azimuth, ts)
    for xs in STATIONS_M:
        row[f"y_mm@x{xs}m"] = value_at(pos[:, 0], pos[:, 1], xs) * 1e3
        row[f"z_m@x{xs}m"] = value_at(pos[:, 0], pos[:, 2], xs)
    series = {"t": t.tolist(), "y": pos[:, 1].tolist(), "x": pos[:, 0].tolist(), "z": pos[:, 2].tolist(),
              "elev": elevation.tolist(), "azim": azimuth.tolist()}
    return {"row": row, "series": series}


def pitch_moment_world_y(case: Case, offsets_deg) -> list[float]:
    """面法線を相対風（初速方向）から仰角方向に offset だけずらした姿勢での、world Y まわりの空力モーメント。

    + は面法線を下げる向き。offset > 0 で M > 0 なら姿勢を戻す（復元）。
    """
    from cap_flight_sim.aerodynamics.factory import build_aerodynamic_model
    from cap_flight_sim.models import RigidBodyState

    cfg = config_from_dict(build_raw(case))
    model = build_aerodynamic_model(cfg.aerodynamics, cfg.cap)
    v = cfg.initial_state.velocity_world_m_s
    wind_elev = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    out = []
    for offset in offsets_deg:
        q = euler_deg_to_quat(np.array([0.0, 90.0 - (wind_elev + offset), 0.0]))
        res = model.evaluate(RigidBodyState(0.0, np.zeros(3), v, q, np.zeros(3)), cfg.cap, cfg.environment)
        out.append(float((quat_to_matrix(q) @ res.moment_body_nm)[1]))
    return out


def pitch_stiffness(case: Case, delta_deg: float = 0.2) -> dict[str, float]:
    """面を相対風に正対させた姿勢のまわりで、ピッチモーメントの傾き（剛性）を差分で求める。

    返り値の k [N·m/rad] が正なら復元（安定）、負なら不安定。周期の線形予測 T = 2π√(I_t / k) も返す。
    """
    m_plus, m_minus = pitch_moment_world_y(case, (delta_deg, -delta_deg))
    k = (m_plus - m_minus) / (2.0 * np.radians(delta_deg))
    i_t = float(config_from_dict(build_raw(case)).cap.inertia_tensor_body_kg_m2[0, 0])
    return {
        "moment_plus_nm": m_plus, "moment_minus_nm": m_minus, "stiffness_nm_per_rad": float(k),
        "transverse_inertia_kg_m2": i_t,
        "linear_period_s": float(2.0 * np.pi * np.sqrt(i_t / k)) if k > 0 else float("nan"),
    }


def build_cases(quick: bool = False) -> list[Case]:
    k = Case(name="knuckle_nominal", group="nominal")
    cp = replace(k, name="knuckle_cp_aft", cp_dz_mm=-2.5)  # 圧力中心を body 原点（天面から 2.5mm 開口側）に置く推測モデル
    cases = [k, cp]
    if quick:
        return cases + [replace(cp, name="quick_yaw", group="perturb_yaw", yaw_deg=1.0, tags={"eps": 1.0})]
    for dz in (-4.0, -2.5, -1.0, 1.0, 2.5, 4.0):
        cases.append(replace(k, name=f"cp_dz={dz}", group="cp_offset", cp_dz_mm=dz))
    for c in (-0.2, -0.05, 0.05, 0.2):
        cases.append(replace(k, name=f"cm_alpha={c}", group="cm_pitch_alpha", cm_pitch_alpha_per_rad=c))
    for base_name, base in (("cp_aft", cp), ("cp_fwd", replace(cp, cp_dz_mm=2.5))):
        for v in (30.0, 40.0, 60.0, 70.0):
            cases.append(replace(base, name=f"{base_name}/speed={v}", group=f"speed/{base_name}", speed_kmh=v))
        for tilt in (0.0, 10.0, 20.0, 45.0, 60.0):
            cases.append(replace(base, name=f"{base_name}/tilt={tilt}", group=f"tilt/{base_name}", tilt_deg=tilt))
        for rpm in (30.0, 100.0, 300.0, 1000.0):
            for magnus, lift, label in ((True, True, "all"), (False, True, "no_magnus"), (False, False, "no_magnus_lift")):
                cases.append(replace(base, name=f"{base_name}/rpm={rpm}/{label}", group=f"rpm/{base_name}/{label}",
                                     rpm=rpm, magnus=magnus, lift=lift))
        for axis_deg in (10.0, 45.0, 90.0):
            a = np.radians(axis_deg)
            cases.append(replace(base, name=f"{base_name}/axis_tilt={axis_deg}", group=f"spin_axis/{base_name}",
                                 rpm=300.0, spin_axis_body=(float(np.sin(a)), 0.0, float(np.cos(a)))))
        for td in (0.01, 0.05, 0.2):
            cases.append(replace(base, name=f"{base_name}/tumble_damping={td}", group=f"damping/{base_name}",
                                 tumble_damping=td))
    cases.append(replace(cp, name="cp_aft/spin_damping=0.05/rpm=300", group="damping/cp_aft", rpm=300.0, spin_damping=0.05))
    # 左右の対称を崩す初期の乱れ。±ε の対で、応答が ε に比例するか（線形）・時間とともに膨らむかを見る
    for base_name, base in (("cp_aft", cp), ("cp_fwd", replace(cp, cp_dz_mm=2.5)), ("no_moment", k)):
        cases.append(replace(base, name=f"{base_name}/eps=0", group=f"perturb/{base_name}", tags={"eps": 0.0, "kind": "ref"}))
        for eps in (1e-3, 1e-2, 1e-1, 1.0, 3.0):
            for sign in (1.0, -1.0):
                cases.append(replace(base, name=f"{base_name}/yaw={sign * eps}", group=f"perturb/{base_name}",
                                     yaw_deg=sign * eps, tags={"eps": sign * eps, "kind": "yaw_deg"}))
            cases.append(replace(base, name=f"{base_name}/yaw_rate={eps}", group=f"perturb/{base_name}",
                                 omega_extra_world=(0.0, 0.0, eps), tags={"eps": eps, "kind": "yaw_rate_rad_s"}))
    # ストレート（対比）: 回転で姿勢が保たれるか
    for dz in (None, -2.5):
        tag = "cp_none" if dz is None else "cp_aft"
        for rpm in (0.0, 300.0, 600.0, 1200.0, 2400.0):
            cases.append(replace(k, name=f"straight/{tag}/rpm={rpm}", group=f"straight/{tag}", pitch="straight",
                                 speed_kmh=58.0, rpm=rpm, cp_dz_mm=dz))
    # 刻み幅の確認（代表的な極端ケースを半分の刻みで再計算）
    for c in [replace(cp, cp_dz_mm=4.0), replace(cp, cp_dz_mm=2.5, yaw_deg=3.0), replace(cp, rpm=1000.0)]:
        cases.append(replace(c, name=f"step_half/{c.cp_dz_mm}/{c.yaw_deg}/{c.rpm}", group="step_check",
                             fixed_step_sec=5e-5, tags={"pair": f"{c.cp_dz_mm}/{c.yaw_deg}/{c.rpm}"}))
        cases.append(replace(c, name=f"step_full/{c.cp_dz_mm}/{c.yaw_deg}/{c.rpm}", group="step_check",
                             tags={"pair": f"{c.cp_dz_mm}/{c.yaw_deg}/{c.rpm}"}))
    return cases


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
    (args.output / "series.json").write_text(
        json.dumps({c.name: r["series"] for c, r in zip(cases, results)}), encoding="utf-8")
    check_cases = {
        "cp_aft_2.5mm": Case(name="s", cp_dz_mm=-2.5),
        "cp_fwd_2.5mm": Case(name="s", cp_dz_mm=2.5),
        "cm_alpha_+0.05": Case(name="s", cm_pitch_alpha_per_rad=0.05),
        "cm_alpha_-0.05": Case(name="s", cm_pitch_alpha_per_rad=-0.05),
    }
    checks: dict[str, Any] = {}
    for name, c in check_cases.items():
        coarse, fine = pitch_stiffness(c, 0.2), pitch_stiffness(c, 0.02)
        # 差分の幅を 1/10 にしてもモーメントが 1/10 にならなければ、正対で不連続（剛性は定義できない）
        continuous = abs(fine["moment_plus_nm"]) < 0.2 * abs(coarse["moment_plus_nm"])
        checks[name] = {"delta=0.2deg": coarse, "delta=0.02deg": fine, "continuous_at_face_on": bool(continuous)}
        if not continuous:
            for v in (coarse, fine):
                v["stiffness_nm_per_rad"] = v["linear_period_s"] = None
    offsets = [0.5, 1, 2, 5, 8, 10, 11, 12, 13, 15, 20, 28, 40, 60]
    checks["moment_vs_offset_deg"] = {"offset_deg": offsets, **{
        name: pitch_moment_world_y(c, offsets) for name, c in check_cases.items() if name.startswith("cp_")}}
    (args.output / "checks.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8")
    if not args.quick:
        from phase2_plots import write_figures  # noqa: PLC0415  同じ scripts/ 内

        write_figures(table, {c.name: r["series"] for c, r in zip(cases, results)}, args.output)
    print(f"{len(cases)} 条件 → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
