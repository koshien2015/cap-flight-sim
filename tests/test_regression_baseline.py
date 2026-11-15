"""通常球の凍結回帰基準（Issue #1 Phase 0）。

既定の積分器（solve_ivp / DOP853）で代表プリセットを計算し、tests/regression/normal_pitches_baseline.json と照合する。
失敗したら物理・設定のどこが軌道を変えたかを確認し、意図した変更なら差と根拠を記録したうえで
    uv run python scripts/update_regression_baseline.py --write
で基準を更新する。
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("update_regression_baseline", ROOT / "scripts/update_regression_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baseline_module():
    return _module()


@pytest.fixture(scope="module")
def expected(baseline_module):
    path = baseline_module.BASELINE_PATH
    assert path.exists(), f"{path.relative_to(ROOT)} が無い → scripts/update_regression_baseline.py --write"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def actual(baseline_module):
    return baseline_module.build_baseline()


def test_normal_pitches_match_frozen_baseline(baseline_module, expected, actual):
    problems = baseline_module.compare(expected, actual)
    assert not problems, "通常球の軌道が基準から変わった:\n" + "\n".join(problems)


def test_baseline_covers_landing_or_catcher_state(expected):
    for case in expected["cases"]:
        terminal = [e for e in case["events"] if e["type"] in ("ground", "catcher_plane")]
        assert terminal, f"{case['name']}: 着地／捕手面到達の状態が記録されていない"
        assert {"position", "normal", "velocity", "omegaWorld"} <= set(terminal[-1])


@pytest.mark.parametrize(
    ("key", "delta"),
    [("position", 1e-6), ("normal", 1e-6), ("velocity", 1e-5), ("omegaWorld", 1e-4), ("timeSec", 1e-8)],
)
def test_compare_detects_changes_above_tolerance(baseline_module, expected, key, delta):
    mutated = copy.deepcopy(expected)
    frame = mutated["cases"][0]["frames"][-1]
    frame[key] = frame[key] + delta if key == "timeSec" else [frame[key][0] + delta, *frame[key][1:]]
    assert baseline_module.compare(expected, mutated)


def test_compare_detects_event_and_summary_changes(baseline_module, expected):
    mutated = copy.deepcopy(expected)
    case = mutated["cases"][0]
    case["events"][-1]["position"][1] += 1e-6
    case["summary"]["lateralBreakFromLaunchLineM"] += 1e-6
    problems = baseline_module.compare(expected, mutated)
    assert any("event" in p for p in problems)
    assert any("lateralBreakFromLaunchLineM" in p for p in problems)


@pytest.mark.parametrize("where", ["frame", "event", "summary"])
def test_compare_reports_nan(baseline_module, expected, where):
    mutated = copy.deepcopy(expected)
    case = mutated["cases"][0]
    if where == "frame":
        case["frames"][-1]["position"][0] = float("nan")
    elif where == "event":
        case["events"][-1]["normal"][0] = float("nan")
    else:
        case["summary"]["flightTimeSec"] = float("nan")
    assert baseline_module.compare(expected, mutated)


def test_compare_reports_missing_event_state(baseline_module, expected):
    mutated = copy.deepcopy(expected)
    del mutated["cases"][0]["events"][-1]["normal"]
    assert any("記録項目" in p for p in baseline_module.compare(expected, mutated))
