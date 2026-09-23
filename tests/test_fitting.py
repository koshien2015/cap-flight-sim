import numpy as np
import pytest

from cap_flight_sim.config import apply_overrides, config_from_dict
from cap_flight_sim.fitting.losses import weighted_loss, weighted_residuals
from cap_flight_sim.fitting.optimizer import FitParameter, fit
from cap_flight_sim.fitting.trajectory_loader import ObservedTrajectory
from cap_flight_sim.simulation import run_simulation


def _observe(raw, times, noise, seed):
    result = run_simulation(config_from_dict(raw), sample_times=times)
    pos = np.array([f.state.position_world_m for f in result.frames])
    rng = np.random.default_rng(seed)
    return ObservedTrajectory("synthetic", times, pos + rng.normal(scale=noise, size=pos.shape), np.ones(len(times)), True)


@pytest.fixture
def short_raw(base_raw):
    base_raw["simulation"] = {"duration_sec": 0.4, "termination": {"stop_at_ground": False, "stop_at_catcher_plane": False}}
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 1500, "axis_world": [0, 0, 1]}
    return base_raw


def test_loss_is_confidence_weighted():
    obs = ObservedTrajectory("o", np.array([0.0, 0.1]), np.zeros((2, 3)), np.array([1.0, 0.25]), False)
    sim = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    assert weighted_loss(obs, sim) == pytest.approx(1.0 * 1.0 + 0.25 * 4.0)
    assert weighted_residuals(obs, sim[:1]).size == 6  # 欠損点も罰則付きで残る


def test_recovers_synthetic_coefficients(short_raw):
    truth = apply_overrides(short_raw, {"aerodynamics.drag.cd_projected": 0.6, "aerodynamics.magnus.coefficient": 1.8})
    obs = _observe(truth, np.arange(0.0, 0.4, 1 / 60), noise=0.002, seed=0)
    params = [FitParameter("aerodynamics.drag.cd_projected", 1.0, 0.1, 2.0),
              FitParameter("aerodynamics.magnus.coefficient", 1.0, 0.0, 4.0)]
    report = fit(short_raw, params, [obs], [], max_nfev=30)
    values = report["summary"]["parameters"]
    assert values["aerodynamics.drag.cd_projected"]["value"] == pytest.approx(0.6, abs=0.03)
    assert values["aerodynamics.magnus.coefficient"]["value"] == pytest.approx(1.8, abs=0.1)
    warnings = " ".join(report["summary"]["warnings"])
    assert "1 球のみ" in warnings and "合成" in warnings and "評価用軌道なし" in warnings


def test_reports_bound_hit(short_raw):
    truth = apply_overrides(short_raw, {"aerodynamics.drag.cd_projected": 1.5})
    obs = _observe(truth, np.arange(0.0, 0.4, 1 / 30), noise=0.0, seed=0)
    report = fit(short_raw, [FitParameter("aerodynamics.drag.cd_projected", 0.5, 0.1, 1.0)], [obs], [], max_nfev=20)
    assert report["fittedParameters"][0]["atBound"] is True
    assert any("境界" in w for w in report["summary"]["warnings"])
