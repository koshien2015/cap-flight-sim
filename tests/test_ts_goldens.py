"""ブラウザ版（TS 移植）との整合を Python 側から守るテスト。

- ゴールデンの古さ検出: Python の物理を変えたのに scripts/generate_ts_goldens.py を再実行していないと失敗する。
  失敗したら再生成し、viewer 側で `pnpm test` を通すこと（TS 側の追従が必要）。
- UI 書き出し JSON の再現: ブラウザが書き出した条件を Python CLI で計算し、TS と同じ終了位置になることを確認する。
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

from cap_flight_sim.config import config_from_dict
from cap_flight_sim.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    spec = importlib.util.spec_from_file_location("generate_ts_goldens", ROOT / "scripts/generate_ts_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 浮動小数は OS（libm・numpy の SIMD 経路）で最下位ビットが変わるため、バイト一致ではなく数値で比べる。
# キー・文字列・真偽値・null・配列長は完全一致を要求する（分岐が変わったら許容誤差では吸収しない）。
GOLDEN_REL_TOL = 1e-12


def _diff_json(expected, actual, path="$"):
    """(構造の不一致リスト, 最悪の数値差 (相対誤差, パス)) を返す。"""
    if isinstance(expected, bool) or isinstance(actual, bool) or expected is None or actual is None \
            or isinstance(expected, str) or isinstance(actual, str):
        return ([] if expected == actual else [f"{path}: {expected!r} → {actual!r}"]), (0.0, path)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        rel = abs(expected - actual) / max(1.0, abs(expected))
        return [], (rel if np.isfinite(rel) else float("inf"), path)
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return [f"{path}: キーが違う {sorted(expected.keys() ^ actual.keys())}"], (0.0, path)
        items = [(expected[k], actual[k], f"{path}.{k}") for k in expected]
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: 長さ {len(expected)} → {len(actual)}"], (0.0, path)
        items = [(e, a, f"{path}[{i}]") for i, (e, a) in enumerate(zip(expected, actual))]
    else:
        return [f"{path}: 型が違う {type(expected).__name__} → {type(actual).__name__}"], (0.0, path)
    mismatches, worst = [], (0.0, path)
    for e, a, p in items:
        m, w = _diff_json(e, a, p)
        mismatches.extend(m)
        worst = max(worst, w, key=lambda x: x[0])
    return mismatches, worst


def test_ts_goldens_are_up_to_date():
    gen = _generator()
    stale = []
    for path, data in gen.build_all().items():
        name = str(path.relative_to(ROOT))
        if not path.exists():
            stale.append(f"{name}: ファイルが無い")
            continue
        mismatches, (rel, where) = _diff_json(json.loads(path.read_text(encoding="utf-8")), json.loads(gen.render(data)))
        stale.extend(f"{name}: {m}" for m in mismatches[:5])
        if not (rel <= GOLDEN_REL_TOL):
            stale.append(f"{name}: 最大相対差 {rel:.2e} @ {where}")
    assert not stale, "ゴールデンが古い → uv run python scripts/generate_ts_goldens.py を実行し、viewer で pnpm test\n" + "\n".join(stale)


def test_golden_diff_detects_structure_and_value_changes():
    mismatches, (rel, _) = _diff_json({"a": [1.0, True, None, "s"]}, {"a": [1.0, True, None, "s"]})
    assert mismatches == [] and rel == 0.0
    assert _diff_json({"a": True}, {"a": False})[0]
    assert _diff_json({"a": [1.0]}, {"a": [1.0, 2.0]})[0]
    assert _diff_json({"a": None}, {"a": 0.0})[0]
    assert _diff_json({"a": 1.0}, {"a": 1.0 + 1e-9})[1][0] > GOLDEN_REL_TOL

def test_browser_exported_config_reproduces_in_python():
    fixture = json.loads((ROOT / "viewer/tests/fixtures/exported-config.json").read_text(encoding="utf-8"))
    result = run_simulation(config_from_dict(fixture["config"]))
    expected = fixture["expected"]
    assert result.termination_reason == expected["terminationReason"]
    final = result.frames[-1].state
    assert abs(final.time_sec - expected["finalTime"]) < 1e-9
    assert np.allclose(final.position_world_m, expected["finalPosition"], atol=1e-8)
