import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cap_flight_sim.config import ConfigError, apply_overrides, config_from_dict, load_config
from cap_flight_sim.coordinates import quat_to_matrix, rpm_to_rad_s
from cap_flight_sim.outputs.serializers import write_result_bundle
from cap_flight_sim.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]


def _omega_world(cfg):
    return quat_to_matrix(cfg.initial_state.orientation_body_to_world) @ cfg.initial_state.angular_velocity_body_rad_s


def test_kmh_and_rpm_converted_to_si(base_raw):
    base_raw["initial_state"].pop("speed_m_s")
    base_raw["initial_state"]["speed_kmh"] = 57.6
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 600, "axis_world": [0, 0, 1]}
    cfg = config_from_dict(base_raw)
    assert np.linalg.norm(cfg.initial_state.velocity_world_m_s) == pytest.approx(16.0)
    assert np.allclose(_omega_world(cfg), [0, 0, rpm_to_rad_s(600)])
    assert cfg.input_units["speed"]["input_unit"] == "km/h"


def test_world_axis_is_converted_to_body_rates(base_raw):
    base_raw["initial_state"]["face_normal_world"] = [0, 1, 0]
    base_raw["initial_state"]["angular_velocity"] = {"rad_s": 100.0, "axis_world": [0, 1, 0]}
    cfg = config_from_dict(base_raw)
    assert np.allclose(cfg.initial_state.angular_velocity_body_rad_s, [0, 0, 100.0], atol=1e-12)


def test_clockwise_requires_viewpoint(base_raw):
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 600, "rotation_sense": "clockwise"}
    with pytest.raises(ConfigError, match="viewed_from_world"):
        config_from_dict(base_raw)


def test_clockwise_viewed_from_above_points_down(base_raw):
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 600, "rotation_sense": "clockwise", "viewed_from_world": [0, 0, 1]}
    assert _omega_world(config_from_dict(base_raw))[2] < 0


def test_conflicting_angular_velocity_forms_rejected(base_raw):
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 600, "axis_world": [0, 0, 1], "axis_body": [0, 0, 1]}
    with pytest.raises(ConfigError, match="conflicting"):
        config_from_dict(base_raw)


def test_parameter_sources_are_tracked(base_raw):
    base_raw["cap"]["mass_kg"] = {"value": 0.0023, "source": "measured", "note": "scale"}
    cfg = config_from_dict(base_raw)
    assert cfg.cap.mass_kg == pytest.approx(0.0023)
    assert cfg.parameter_sources["cap.mass_kg"]["source"] == "measured"
    assert cfg.parameter_sources["aerodynamics.drag.cd_projected"]["source"] == "assumed"


def test_invalid_source_rejected(base_raw):
    base_raw["cap"]["mass_kg"] = {"value": 0.0023, "source": "cfd"}
    with pytest.raises(ConfigError):
        config_from_dict(base_raw)


def test_orientation_override_replaces_previous_form(base_raw):
    base_raw["initial_state"]["orientation_euler_deg"] = [0, 0, 0]
    base_raw["initial_state"].pop("face_normal_world")
    raw = apply_overrides(base_raw, {"initial_state.face_normal_world": [0, 1, 0]})
    assert "orientation_euler_deg" not in raw["initial_state"]
    assert "orientation_euler_deg" in base_raw["initial_state"], "元の dict を変更しない"


def test_all_examples_load():
    for path in sorted((ROOT / "examples").glob("[!_]*.yaml")):
        cfg = load_config(path)
        assert cfg.cap.mass_kg > 0, path


def test_result_bundle_contents(base_raw, tmp_path):
    base_raw["aerodynamics"]["enable"] = {"drag": False, "lift": False, "magnus": False, "moments": False}
    result = run_simulation(config_from_dict(base_raw))
    paths = write_result_bundle(result, tmp_path)
    doc = json.loads(paths["result"].read_text())
    for key in ("parameter_source", "aerodynamic_model", "warnings", "simulation_version", "coordinate_system"):
        assert key in doc
    cp = doc["summary"]["catcherPlane"]
    for key in ("timeSec", "yM", "zM", "velocityWorldMS", "quaternionWxyz", "rpm", "spinAxisWorld", "forcesN", "terminationReason"):
        assert key in cp
    df = pd.read_csv(paths["timeseries"])
    required = ["time", "x", "y", "z", "vx", "vy", "vz", "speed", "qw", "qx", "qy", "qz", "roll", "pitch", "yaw",
                "wx", "wy", "wz", "rpm", "angle_of_attack", "sideslip", "drag_x", "lift_y", "magnus_z",
                "total_force_x", "moment_z", "cd", "cl", "cy"]
    assert not set(required) - set(df.columns)
    viewer = json.loads(paths["viewer"].read_text())
    assert viewer["schemaVersion"] == "1.0.0"
    assert viewer["coordinateSystem"]["handedness"] == "right"
    assert len(viewer["frames"]) == len(df)
    assert {"release", "catcher_plane"} <= {e["type"] for e in viewer["events"]}
