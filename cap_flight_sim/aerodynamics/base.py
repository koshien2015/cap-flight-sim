"""空力モデルの共通インターフェースと流れ場の幾何量。

符号規約（ボディ座標、n = body +Z = 面法線）:
    u      : 空気に対するキャップの速度 = v_cap - wind（= -relative_air_velocity）
    迎角 α : α = atan2(-u_z, sqrt(u_x² + u_y²))  [-90°, +90°]
             α > 0 は相対風が body -Z 側（n と反対側の面）へ当たる状態。
             フリスビーを n=上 で水平に投げ、やや下向きに進むとき α > 0。
             面をひっくり返す（n → -n）と α の符号も反転する。
    横滑り角 β : β = atan2(u_y, u_x)  面内での進行方向（body X から body Y へ測る）。
             軸対称なキャップでは β はスピンとともに回転し、空気力には直接効かない。
    法線角 θn : n と相対風（-u）のなす角。θn = 90° がエッジオン。

空力座標（モーメント・ルックアップテーブル用）:
    e_drag  = -û                     （抗力方向）
    e_lift  = n - (n·û)û を正規化     （相対風に垂直、n と û を含む面内）
    e_inplane = u の面内成分の単位ベクトル（ロール軸）
    e_pitch = e_inplane × n          （正のモーメントで α が増える向き）
    e_yaw   = n                      （スピン軸）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..coordinates import quat_to_matrix, safe_unit, world_to_body
from ..models import AerodynamicResult, CapParameters, Environment, RigidBodyState

SPEED_EPS_M_S = 1e-6
DEGENERACY_EPS = 1e-6


@dataclass(frozen=True)
class FlowGeometry:
    speed_m_s: float
    dynamic_pressure_pa: float
    u_world: np.ndarray  # 空気に対するキャップ速度
    u_hat_world: np.ndarray
    u_body: np.ndarray
    angle_of_attack_rad: float
    sideslip_rad: float
    normal_to_wind_deg: float
    normal_world: np.ndarray
    omega_world: np.ndarray
    lift_dir_world: np.ndarray  # 退化時は零
    inplane_dir_body: np.ndarray  # 退化時は零
    pitch_axis_body: np.ndarray  # 退化時は零
    omega_perp_world: np.ndarray  # u に垂直な角速度成分
    spin_parameter: float
    is_stationary: bool
    is_face_on: bool


def compute_flow_geometry(state: RigidBodyState, cap: CapParameters, environment: Environment) -> FlowGeometry:
    q = state.orientation_body_to_world
    rotation = quat_to_matrix(q)
    normal_world = rotation[:, 2]
    omega_world = rotation @ state.angular_velocity_body_rad_s
    relative_air = environment.wind_velocity_world_m_s - state.velocity_world_m_s
    u_world = -relative_air
    speed = float(np.linalg.norm(u_world))
    stationary = speed < SPEED_EPS_M_S
    q_dyn = 0.5 * environment.air_density_kg_m3 * speed**2

    if stationary:
        zero = np.zeros(3)
        return FlowGeometry(
            speed, q_dyn, u_world, zero, zero, 0.0, 0.0, float("nan"), normal_world, omega_world,
            zero, zero, zero, zero, 0.0, True, False,
        )

    u_hat = u_world / speed
    u_body = world_to_body(q, u_world)
    inplane = np.hypot(u_body[0], u_body[1])
    alpha = float(np.arctan2(-u_body[2], inplane))
    beta = float(np.arctan2(u_body[1], u_body[0])) if inplane > DEGENERACY_EPS * speed else 0.0
    cos_n = float(np.clip(np.dot(normal_world, -u_hat), -1.0, 1.0))

    lift_raw = normal_world - np.dot(normal_world, u_hat) * u_hat
    face_on = np.linalg.norm(lift_raw) < DEGENERACY_EPS
    lift_dir = np.zeros(3) if face_on else lift_raw / np.linalg.norm(lift_raw)

    inplane_body = np.array([u_body[0], u_body[1], 0.0])
    inplane_body = safe_unit(inplane_body, DEGENERACY_EPS * speed)
    pitch_axis_body = np.cross(inplane_body, np.array([0.0, 0.0, 1.0]))

    omega_perp = omega_world - np.dot(omega_world, u_hat) * u_hat
    spin_parameter = cap.radius_m * float(np.linalg.norm(omega_perp)) / speed

    return FlowGeometry(
        speed_m_s=speed,
        dynamic_pressure_pa=q_dyn,
        u_world=u_world,
        u_hat_world=u_hat,
        u_body=u_body,
        angle_of_attack_rad=alpha,
        sideslip_rad=beta,
        normal_to_wind_deg=float(np.rad2deg(np.arccos(cos_n))),
        normal_world=normal_world,
        omega_world=omega_world,
        lift_dir_world=lift_dir,
        inplane_dir_body=inplane_body,
        pitch_axis_body=pitch_axis_body,
        omega_perp_world=omega_perp,
        spin_parameter=spin_parameter,
        is_stationary=False,
        is_face_on=bool(face_on),
    )


def magnus_direction(flow: FlowGeometry) -> np.ndarray:
    """ω⊥ × u の単位ベクトル（u は空気に対するキャップ速度）。回転なし・静止時は零。"""
    return safe_unit(np.cross(flow.omega_perp_world, flow.u_world))


class AerodynamicModel(Protocol):
    name: str

    def evaluate(self, state: RigidBodyState, cap: CapParameters, environment: Environment) -> AerodynamicResult:
        ...

    def describe(self) -> dict:
        """出力 JSON に記録するモデル説明（式・係数・出典区分）。"""
        ...
