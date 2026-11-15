"""通常球の回帰基準（凍結ベースライン）を生成・照合する。

Issue #1 Phase 0。TS 照合用ゴールデン（generate_ts_goldens.py）は Python の物理を変えるたびに再生成されるため、
「既存の通常球の軌道が変わったこと」自体は検出しない。こちらは意図して更新しない限り固定される基準で、
CLI・実験・フィッティングが使う既定の積分器（solve_ivp / DOP853）の結果を記録する。

    uv run python scripts/update_regression_baseline.py           # 現行結果と基準の差を表示（書き換えない）
    uv run python scripts/update_regression_baseline.py --write   # 基準を書き換える（変更理由を説明できるときだけ）

tests/test_regression_baseline.py が許容誤差 TOLERANCES で照合する。
許容誤差は浮動小数の揺らぎ（初期位置 1e-12 m の摂動で ≤3e-11）より十分大きく、
物理・積分設定の変更（例: spec_example_tumble は最大刻み半減で面法線が ~7e-5 変わる）より十分小さい値にした。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cap_flight_sim.config import apply_overrides, config_from_dict, load_raw_config  # noqa: E402
from cap_flight_sim.outputs.summaries import compute_summary  # noqa: E402
from cap_flight_sim.simulation import Frame, run_simulation  # noqa: E402

BASELINE_PATH = ROOT / "tests/regression/normal_pitches_baseline.json"

# 通常球の代表プリセット。低スピン域（ナックル想定）は Phase 3〜4 で変わり得るため含めない。
PRESETS = (
    "overhand_drop",
    "flick_flat",
    "sidearm_flat",
    "spec_example_tumble",
    "ballistic_no_aero",
    "lookup_table_example",
)
OUTPUT_HZ = 60  # 出力の間引きのみ。積分は dense output から補間するので軌道は変わらない

# 量ごとの絶対許容誤差（SI 単位。面法線は単位ベクトル成分）
TOLERANCES: dict[str, float] = {
    "timeSec": 1e-9,
    "position": 1e-7,
    "velocity": 1e-6,
    "normal": 1e-7,
    "omegaWorld": 1e-5,
    "summary": 1e-7,
}
SUMMARY_KEYS = (
    "flightTimeSec",
    "lateralBreakFromLaunchLineM",
    "dropFromLaunchLineM",
    "maxAbsLateralBreakM",
    "faceNormalChangeDeg",
    "maxFaceNormalChangeDeg",
)


def _floats(v: Any) -> list[float]:
    return [float(x) for x in np.asarray(v).ravel()]


def _state(frame: Frame) -> dict[str, list[float]]:
    return {
        "position": _floats(frame.state.position_world_m),
        "velocity": _floats(frame.state.velocity_world_m_s),
        "normal": _floats(frame.normal_world),
        "omegaWorld": _floats(frame.omega_world_rad_s),
    }


def run_case(name: str) -> dict[str, Any]:
    raw = apply_overrides(load_raw_config(ROOT / "examples" / f"{name}.yaml"), {"simulation.output_hz": OUTPUT_HZ})
    result = run_simulation(config_from_dict(raw, name=name))
    summary = compute_summary(result)
    return {
        "name": name,
        "solver": f"{result.config.settings.solver}/{result.config.settings.method}",
        "terminationReason": result.termination_reason,
        "events": [
            {"type": e["type"], "timeSec": float(e["timeSec"]), **(_state(e["frame"]) if "frame" in e else {})}
            for e in result.events
        ],
        "summary": {k: float(summary[k]) for k in SUMMARY_KEYS},
        "frames": [{"timeSec": float(f.state.time_sec), **_state(f)} for f in result.frames],
    }


def build_baseline() -> dict[str, Any]:
    return {
        "generator": "scripts/update_regression_baseline.py",
        "note": "通常球の凍結回帰基準。物理を変えて差が出たら、差と根拠を示したうえで --write で更新する",
        "outputHz": OUTPUT_HZ,
        "cases": [run_case(name) for name in PRESETS],
    }


def _max_abs(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """基準との差のうち許容誤差を超えたものを人が読める行で返す（空なら一致）。"""
    problems: list[str] = []
    exp_cases = {c["name"]: c for c in expected["cases"]}
    act_cases = {c["name"]: c for c in actual["cases"]}
    if set(exp_cases) != set(act_cases):
        problems.append(f"ケース集合が違う: 基準 {sorted(exp_cases)} / 現行 {sorted(act_cases)}")
    for name in sorted(set(exp_cases) & set(act_cases)):
        problems.extend(f"{name}: {p}" for p in _compare_case(exp_cases[name], act_cases[name]))
    return problems


def _compare_case(exp: dict[str, Any], act: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in ("solver", "terminationReason"):
        if exp[key] != act[key]:
            problems.append(f"{key}: {exp[key]} → {act[key]}")
    if [e["type"] for e in exp["events"]] != [e["type"] for e in act["events"]]:
        problems.append(f"イベント列: {[e['type'] for e in exp['events']]} → {[e['type'] for e in act['events']]}")
    else:
        for e, a in zip(exp["events"], act["events"]):
            problems.extend(_compare_fields(f"event {e['type']}", e, a))
    for key in SUMMARY_KEYS:
        if (d := abs(exp["summary"][key] - act["summary"][key])) > TOLERANCES["summary"]:
            problems.append(f"summary.{key}: {exp['summary'][key]:.9g} → {act['summary'][key]:.9g} (|Δ|={d:.2e})")
    if len(exp["frames"]) != len(act["frames"]):
        problems.append(f"フレーム数: {len(exp['frames'])} → {len(act['frames'])}")
        return problems
    for key in ("timeSec", "position", "velocity", "normal", "omegaWorld"):
        d = _max_abs([f[key] for f in exp["frames"]], [f[key] for f in act["frames"]])
        if d > TOLERANCES[key]:
            problems.append(f"frames.{key}: max|Δ|={d:.2e} > {TOLERANCES[key]:.0e}")
    return problems


def _compare_fields(label: str, exp: dict[str, Any], act: dict[str, Any]) -> list[str]:
    problems = []
    for key in ("timeSec", "position", "velocity", "normal", "omegaWorld"):
        if key not in exp:
            continue
        d = _max_abs(exp[key], act[key])
        if d > TOLERANCES[key]:
            problems.append(f"{label}.{key}: {exp[key]} → {act[key]} (max|Δ|={d:.2e})")
    return problems


def render(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="基準ファイルを現行結果で書き換える")
    args = parser.parse_args()
    actual = build_baseline()
    if BASELINE_PATH.exists():
        problems = compare(json.loads(BASELINE_PATH.read_text(encoding="utf-8")), actual)
        print("\n".join(problems) if problems else "基準と一致（許容誤差内）")
    else:
        problems = ["基準ファイルが無い"]
        print(problems[0])
    if args.write:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(render(actual), encoding="utf-8")
        print(f"wrote {BASELINE_PATH.relative_to(ROOT)}")
        return 0
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
