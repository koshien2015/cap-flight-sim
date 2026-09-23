from __future__ import annotations

import copy
from typing import Any

import pytest

BASE_RAW: dict[str, Any] = {
    "cap": {"mass_kg": 0.0022, "outer_diameter_m": 0.03, "height_m": 0.015},
    "simulation": {"duration_sec": 2.0, "output_hz": 240},
    "initial_state": {
        "position_world_m": [0.0, 0.0, 1.45],
        "speed_m_s": 16.0,
        "launch_elevation_deg": 2.0,
        "face_normal_world": [0.0, 0.0, 1.0],
        "angular_velocity": {"rpm": 0.0, "axis_body": [0.0, 0.0, 1.0]},
    },
    "aerodynamics": {
        "model": "simple",
        "drag": {"cd_projected": 1.0},
        "lift": {"cl_alpha_per_rad": 1.4},
        "magnus": {"coefficient": 1.0, "max_coefficient": 0.5},
    },
}


@pytest.fixture
def base_raw() -> dict[str, Any]:
    return copy.deepcopy(BASE_RAW)
