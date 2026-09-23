import numpy as np
import pytest

from cap_flight_sim.config import config_from_dict
from cap_flight_sim.simulation import run_simulation

ALL_OFF = {"drag": False, "lift": False, "magnus": False, "moments": False}


def test_catcher_plane_event_fires_at_pitch_distance(base_raw):
    base_raw["aerodynamics"]["enable"] = ALL_OFF
    result = run_simulation(config_from_dict(base_raw))
    assert result.termination_reason == "catcher_plane"
    ev = result.catcher_plane_event
    assert ev["frame"].state.position_world_m[0] == pytest.approx(9.22, abs=1e-6)
    v0 = result.config.initial_state.velocity_world_m_s
    assert ev["timeSec"] == pytest.approx(9.22 / v0[0], rel=1e-6)
    assert result.frames[-1].state.time_sec == pytest.approx(ev["timeSec"], abs=1e-9)


def test_ground_event_stops(base_raw):
    base_raw["aerodynamics"]["enable"] = ALL_OFF
    base_raw["initial_state"]["speed_m_s"] = 5.0
    result = run_simulation(config_from_dict(base_raw))
    assert result.termination_reason == "ground"
    assert result.frames[-1].state.position_world_m[2] == pytest.approx(0.0, abs=1e-6)
    assert any("到達せず" in w for w in result.warnings)


def test_non_finite_state_recorded_as_abnormal(base_raw):
    from cap_flight_sim.models import AerodynamicResult

    class ExplodingModel:
        name = "exploding"

        def evaluate(self, state, cap, env):
            bad = np.full(3, np.nan) if state.time_sec > 0.05 else np.zeros(3)
            return AerodynamicResult(bad, np.zeros(3), bad, np.zeros(3), np.zeros(3), {}, {})

        def describe(self):
            return {"name": self.name}

    result = run_simulation(config_from_dict(base_raw), model=ExplodingModel())
    assert result.termination_reason == "abnormal_non_finite"
    assert result.frames, "直前までの結果が保持されること"
    assert any("異常終了" in w for w in result.warnings)


def test_overspeed_event(base_raw):
    base_raw["aerodynamics"]["enable"] = ALL_OFF
    base_raw["simulation"]["termination"] = {"max_speed_m_s": 16.5, "stop_at_catcher_plane": False}
    result = run_simulation(config_from_dict(base_raw))
    assert result.termination_reason == "abnormal_speed"
