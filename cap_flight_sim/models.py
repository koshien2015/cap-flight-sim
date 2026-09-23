"""データモデル（SI 単位で保持）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ParameterSource = Literal["measured", "fitted", "literature", "assumed"]
PARAMETER_SOURCES: tuple[str, ...] = ("measured", "fitted", "literature", "assumed")

# body +Z がキャップのどちら側を向くか（表裏非対称を扱うための区別）
SurfaceOrientation = Literal["body_z_is_top", "body_z_is_cavity"]


@dataclass(frozen=True)
class CapParameters:
    mass_kg: float
    outer_diameter_m: float
    height_m: float
    reference_area_m2: float
    center_of_mass_body_m: np.ndarray
    inertia_tensor_body_kg_m2: np.ndarray
    surface_orientation: SurfaceOrientation
    inertia_method: str

    @property
    def radius_m(self) -> float:
        return 0.5 * self.outer_diameter_m


@dataclass(frozen=True)
class Environment:
    air_density_kg_m3: float
    wind_velocity_world_m_s: np.ndarray
    gravity_m_s2: float

    @property
    def gravity_world_m_s2(self) -> np.ndarray:
        return np.array([0.0, 0.0, -self.gravity_m_s2])


@dataclass(frozen=True)
class FieldGeometry:
    pitch_distance_m: float = 9.22
    ground_z_m: float = 0.0


@dataclass
class RigidBodyState:
    time_sec: float
    position_world_m: np.ndarray  # shape (3,) 重心位置
    velocity_world_m_s: np.ndarray  # shape (3,)
    orientation_body_to_world: np.ndarray  # quaternion [w, x, y, z]
    angular_velocity_body_rad_s: np.ndarray  # shape (3,)

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position_world_m,
                self.velocity_world_m_s,
                self.orientation_body_to_world,
                self.angular_velocity_body_rad_s,
            ]
        )

    @staticmethod
    def from_vector(t: float, y: np.ndarray) -> "RigidBodyState":
        return RigidBodyState(
            time_sec=float(t),
            position_world_m=np.array(y[0:3], dtype=float),
            velocity_world_m_s=np.array(y[3:6], dtype=float),
            orientation_body_to_world=np.array(y[6:10], dtype=float),
            angular_velocity_body_rad_s=np.array(y[10:13], dtype=float),
        )


@dataclass
class AerodynamicResult:
    force_world_n: np.ndarray
    moment_body_nm: np.ndarray
    drag_world_n: np.ndarray
    lift_world_n: np.ndarray
    magnus_world_n: np.ndarray
    coefficients: dict[str, float]
    diagnostics: dict[str, float | str | bool]


@dataclass(frozen=True)
class ForceToggles:
    drag: bool = True
    lift: bool = True
    magnus: bool = True
    moments: bool = True


@dataclass(frozen=True)
class TerminationSettings:
    stop_at_ground: bool = True
    stop_at_catcher_plane: bool = True
    max_speed_m_s: float = 100.0
    max_angular_speed_rad_s: float = 5000.0


@dataclass(frozen=True)
class SimulationSettings:
    duration_sec: float
    output_hz: float
    max_step_sec: float
    max_step_reason: str
    solver: str  # "solve_ivp" | "rk4"
    method: str  # solve_ivp の手法 (RK45 / DOP853 / ...)
    rtol: float
    atol: float
    fixed_step_sec: float
    quaternion_drift_gain: float
    video_fps: float
    termination: TerminationSettings


@dataclass
class SimulationConfig:
    name: str
    label: str
    cap: CapParameters
    environment: Environment
    field_geometry: FieldGeometry
    settings: SimulationSettings
    initial_state: RigidBodyState
    aerodynamics: dict[str, Any]
    raw: dict[str, Any]
    parameter_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    input_units: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
