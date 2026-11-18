"""scripts/issue7_wake_study.py の計測関数と、診断の要点。全スイープは重いので pytest では回さない。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study():
    sys.path.insert(0, str(ROOT / "scripts"))  # 同じ scripts/ 内の phase2_sensitivity を引けるようにする
    spec = importlib.util.spec_from_file_location("issue7_wake_study", ROOT / "scripts/issue7_wake_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass が自モジュールを引けるように登録する
    spec.loader.exec_module(module)
    return module


def test_detrended_peak_to_peak_ignores_a_steady_drift(study):
    t = np.linspace(0.0, 1.0, 1001)
    assert study.detrended_peak_to_peak(t, 0.3 * t + 0.1) == pytest.approx(0.0, abs=1e-12)
    swaying = 0.3 * t + 0.02 * np.cos(2 * np.pi * 3.0 * t)
    assert study.detrended_peak_to_peak(t, swaying) == pytest.approx(0.04, rel=0.05)


def test_direction_reversals_counts_sign_changes_outside_the_deadband(study):
    t = np.linspace(0.0, 1.0, 2001)
    assert study.direction_reversals(np.cos(2 * np.pi * 2.0 * t)) == 4
    assert study.direction_reversals(np.full(10, 0.5)) == 0
    assert study.direction_reversals(1e-5 * np.cos(2 * np.pi * 2.0 * t)) == 0  # 止まっているとみなす速さ


def test_equilibria_reports_zero_crossings_and_stability(study):
    x = np.linspace(0.0, 180.0, 181)
    found = study.equilibria(x, list(np.sin(np.radians(2.0 * (x - 45.0)))))
    assert [round(e["offset_deg"]) for e in found] == [45, 135]
    assert [e["stable"] for e in found] == [True, False]


def test_restoring_moment_has_no_equilibrium_near_face_up(study):
    """復原モーメント（圧力中心のずれ）の安定な平衡は天面が前を向く姿勢の近くにあり、面が上向きの姿勢には無い。"""
    offsets = np.arange(1.0, 180.0, 1.0)
    found = study.equilibria(offsets, study.static_pitch_moment(offsets, study.CP_ON))
    stable = [e["offset_deg"] for e in found if e["stable"]]
    assert len(stable) == 1 and 5.0 < stable[0] < 20.0
    assert not any(60.0 < e["offset_deg"] < 120.0 for e in found)


def test_face_up_release_turns_over_with_the_restoring_moment_but_not_with_the_wake_force(study):
    coarse = {"simulation.fixed_step_sec": 1e-3}
    flipped = study.run_case(study.Case("flip", "t", overrides={**study.moments(study.CP_ON), **coarse}))["row"]
    assert flipped["pitch_rotation_deg"] > 180.0 and flipped["normal_z_min"] < -0.9
    swayed = study.run_case(study.Case("sway", "t", overrides={**study.wake(seed=3), **coarse}))["row"]
    assert swayed["max_normal_change_deg"] < 1e-3
    assert swayed["vy_reversals"] >= 2 and swayed["y_detrended_p2p_mm"] > 10.0
