"""MVP 用の簡易係数モデル（すべて検証前の近似）。

抗力   F_D = q A_ref Cd(α) · e_drag
揚力   F_L = q A_ref Cl(α) · e_lift       （Cl は α の奇関数 + 非対称項 cl0）
マグヌス F_M = q A_ref C_M(S) · unit(ω⊥ × u)  （球・円柱の式を流用した暫定モデル）
モーメント
    M_cp   = (r_cp - r_com) × F_aero            空力中心と重心のずれ
    M_pitch = q A_ref D Cm(α) · e_pitch         姿勢の安定化/不安定化
    M_damp = -½ρ|u| A_ref D R (c_spin ω_n + c_tumble ω_t)  回転抗力
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..coordinates import world_to_body
from ..models import AerodynamicResult, CapParameters, Environment, ForceToggles, RigidBodyState
from .base import compute_flow_geometry, magnus_direction

DRAG_MODELS = ("projected_area", "constant", "quadratic_aoa")
POST_STALL_MODES = ("decay", "hold")
MAGNUS_MODELS = ("linear_spin_parameter", "constant")


@dataclass(frozen=True)
class DragParams:
    model: str = "projected_area"
    cd_base: float = 1.0  # constant / quadratic_aoa の α=0 値
    cd_projected: float = 1.0  # projected_area: 投影面積基準の Cd
    cd_alpha2_per_rad2: float = 0.0  # quadratic_aoa: Cd = cd_base + k α²


@dataclass(frozen=True)
class LiftParams:
    cl0: float = 0.0  # α=0 の揚力（表裏非対称）。body +Z 方向を正とする
    cl_alpha_per_rad: float = 0.0
    stall_angle_deg: float = 45.0
    post_stall: str = "decay"  # decay: 90° で 0 へ線形減衰 / hold: 失速時の値を保持


@dataclass(frozen=True)
class MagnusParams:
    model: str = "linear_spin_parameter"
    coefficient: float = 0.0  # linear: C_M = coefficient * S / constant: C_M = coefficient
    max_coefficient: float = 1.0


@dataclass(frozen=True)
class MomentParams:
    center_of_pressure_body_m: tuple[float, float, float] | None = None  # None = 重心と一致
    cm_pitch0: float = 0.0
    cm_pitch_alpha_per_rad: float = 0.0
    spin_damping: float = 0.0  # 面法線まわり
    tumble_damping: float = 0.0  # 面内軸まわり


@dataclass(frozen=True)
class SimpleAerodynamicModel:
    drag: DragParams = field(default_factory=DragParams)
    lift: LiftParams = field(default_factory=LiftParams)
    magnus: MagnusParams = field(default_factory=MagnusParams)
    moments: MomentParams = field(default_factory=MomentParams)
    toggles: ForceToggles = field(default_factory=ForceToggles)
    name: str = "simple_coefficient_model"

    def __post_init__(self) -> None:
        if self.drag.model not in DRAG_MODELS:
            raise ValueError(f"drag.model must be one of {DRAG_MODELS}")
        if self.lift.post_stall not in POST_STALL_MODES:
            raise ValueError(f"lift.post_stall must be one of {POST_STALL_MODES}")
        if self.magnus.model not in MAGNUS_MODELS:
            raise ValueError(f"magnus.model must be one of {MAGNUS_MODELS}")
        if not 0.0 < self.lift.stall_angle_deg < 90.0:
            raise ValueError("lift.stall_angle_deg must be in (0, 90)")

    # ------------------------------------------------------------ 係数関数

    def drag_coefficient(self, alpha_rad: float, cap: CapParameters) -> tuple[float, float]:
        """(Cd[A_ref 基準], 投影面積)。"""
        area_proj = projected_area_m2(alpha_rad, cap)
        p = self.drag
        if p.model == "constant":
            return p.cd_base, area_proj
        if p.model == "quadratic_aoa":
            return p.cd_base + p.cd_alpha2_per_rad2 * alpha_rad**2, area_proj
        return p.cd_projected * area_proj / cap.reference_area_m2, area_proj

    def lift_coefficient(self, alpha_rad: float) -> float:
        p = self.lift
        stall = np.deg2rad(p.stall_angle_deg)
        a = abs(alpha_rad)
        if a <= stall:
            odd = p.cl_alpha_per_rad * a
        elif p.post_stall == "hold":
            odd = p.cl_alpha_per_rad * stall
        else:
            odd = p.cl_alpha_per_rad * stall * (0.5 * np.pi - a) / (0.5 * np.pi - stall)
        return float(p.cl0 + np.sign(alpha_rad) * odd)

    def magnus_coefficient(self, spin_parameter: float) -> float:
        p = self.magnus
        if spin_parameter <= 0.0:
            return 0.0
        value = p.coefficient if p.model == "constant" else p.coefficient * spin_parameter
        return float(np.clip(value, -p.max_coefficient, p.max_coefficient))

    def pitch_moment_coefficient(self, alpha_rad: float) -> float:
        return self.moments.cm_pitch0 + self.moments.cm_pitch_alpha_per_rad * alpha_rad

    # ------------------------------------------------------------ 評価

    def evaluate(self, state: RigidBodyState, cap: CapParameters, environment: Environment) -> AerodynamicResult:
        flow = compute_flow_geometry(state, cap, environment)
        zero = np.zeros(3)
        if flow.is_stationary:
            return _stationary_result()

        qa = flow.dynamic_pressure_pa * cap.reference_area_m2
        alpha = flow.angle_of_attack_rad

        cd, area_proj = self.drag_coefficient(alpha, cap)
        drag = -qa * cd * flow.u_hat_world if self.toggles.drag else zero

        cl = self.lift_coefficient(alpha)
        # Cl は body +Z 基準。e_lift は n の相対風垂直成分なので Cl>0 で n 側へ働く。
        # α<0（n 側から風が当たる）では奇関数部が負になり、力は -n 側へ向く。
        lift = qa * cl * flow.lift_dir_world if self.toggles.lift else zero

        cm_coef = self.magnus_coefficient(flow.spin_parameter)
        magnus = qa * cm_coef * magnus_direction(flow) if self.toggles.magnus else zero

        force = drag + lift + magnus
        moment, cm_pitch = self._moment_body(state, cap, environment, flow, force) if self.toggles.moments else (zero, 0.0)

        return AerodynamicResult(
            force_world_n=force,
            moment_body_nm=moment,
            drag_world_n=drag,
            lift_world_n=lift,
            magnus_world_n=magnus,
            coefficients={"cd": cd, "cl": cl, "cy": cm_coef, "cm_pitch": cm_pitch},
            diagnostics={
                "speed_m_s": flow.speed_m_s,
                "dynamic_pressure_pa": flow.dynamic_pressure_pa,
                "angle_of_attack_deg": float(np.rad2deg(alpha)),
                "sideslip_deg": float(np.rad2deg(flow.sideslip_rad)),
                "normal_to_wind_deg": flow.normal_to_wind_deg,
                "projected_area_m2": area_proj,
                "spin_parameter": flow.spin_parameter,
                "lift_degenerate_face_on": flow.is_face_on,
                "stationary": False,
            },
        )

    def _moment_body(
        self, state: RigidBodyState, cap: CapParameters, environment: Environment, flow, force_world: np.ndarray
    ) -> tuple[np.ndarray, float]:
        p = self.moments
        q = state.orientation_body_to_world
        moment = np.zeros(3)
        if p.center_of_pressure_body_m is not None:
            arm = np.asarray(p.center_of_pressure_body_m) - cap.center_of_mass_body_m
            moment = moment + np.cross(arm, world_to_body(q, force_world))

        cm_pitch = self.pitch_moment_coefficient(flow.angle_of_attack_rad)
        qad = flow.dynamic_pressure_pa * cap.reference_area_m2 * cap.outer_diameter_m
        moment = moment + qad * cm_pitch * flow.pitch_axis_body

        omega = state.angular_velocity_body_rad_s
        omega_n = np.array([0.0, 0.0, omega[2]])
        omega_t = np.array([omega[0], omega[1], 0.0])
        damp_scale = (
            0.5 * environment.air_density_kg_m3 * flow.speed_m_s
            * cap.reference_area_m2 * cap.outer_diameter_m * cap.radius_m
        )
        moment = moment - damp_scale * (p.spin_damping * omega_n + p.tumble_damping * omega_t)
        return moment, cm_pitch

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": "provisional_unvalidated",
            "equations": {
                "drag": "F_D = -q A_ref Cd(alpha) u_hat",
                "lift": "F_L = q A_ref Cl(alpha) e_lift, e_lift = unit(n - (n.u_hat) u_hat)",
                "magnus": "F_M = q A_ref C_M(S) unit(omega_perp x u), S = R |omega_perp| / |u|",
                "moment": "(r_cp - r_com) x F + q A_ref D Cm(alpha) e_pitch - 0.5 rho |u| A_ref D R (c_spin w_n + c_tumble w_t)",
            },
            "notes": [
                "マグヌス式は球・円柱の式をキャップへ流用した暫定近似",
                "projected_area 抗力は円筒の幾何投影面積による検証前の近似",
            ],
            "drag": vars(self.drag),
            "lift": vars(self.lift),
            "magnus": vars(self.magnus),
            "moments": {**vars(self.moments)},
            "toggles": vars(self.toggles),
        }


def projected_area_m2(alpha_rad: float, cap: CapParameters) -> float:
    """円筒を相対風方向から見た幾何投影面積 π R² |sin α| + D h |cos α|。"""
    return float(
        np.pi * cap.radius_m**2 * abs(np.sin(alpha_rad)) + cap.outer_diameter_m * cap.height_m * abs(np.cos(alpha_rad))
    )


def _stationary_result() -> AerodynamicResult:
    zero = np.zeros(3)
    return AerodynamicResult(
        force_world_n=zero, moment_body_nm=zero, drag_world_n=zero, lift_world_n=zero, magnus_world_n=zero,
        coefficients={"cd": 0.0, "cl": 0.0, "cy": 0.0, "cm_pitch": 0.0},
        diagnostics={
            "speed_m_s": 0.0, "dynamic_pressure_pa": 0.0, "angle_of_attack_deg": 0.0, "sideslip_deg": 0.0,
            "normal_to_wind_deg": float("nan"), "projected_area_m2": 0.0, "spin_parameter": 0.0,
            "lift_degenerate_face_on": False, "stationary": True,
        },
    )
