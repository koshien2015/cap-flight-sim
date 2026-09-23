"""剛体の慣性近似と運動方程式。"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .coordinates import quat_derivative, quat_normalize
from .models import AerodynamicResult, CapParameters, Environment, RigidBodyState

INERTIA_METHODS = ("thin_walled_cup", "solid_cylinder", "thin_disk")


class NonFiniteStateError(RuntimeError):
    """状態量・力に NaN/Inf が現れた。"""


def approximate_mass_properties(
    mass_kg: float,
    diameter_m: float,
    height_m: float,
    method: str,
    body_z_is_top: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """一様密度近似による (重心[body], 重心まわり慣性テンソル[body], 説明文)。

    body 原点 = 円筒の幾何中心（高さ方向の中央）。
    - thin_walled_cup: 天板（薄円盤）+ 側壁（薄肉円筒）の一様シェル。質量は面積比で配分。
                       天板が偏るため重心は天板側へずれる。
    - solid_cylinder : 中実円筒。
    - thin_disk      : 厚さを無視した薄円盤。
    """
    r = 0.5 * diameter_m
    h = height_m
    if method == "solid_cylinder":
        izz = 0.5 * mass_kg * r**2
        ixx = mass_kg * (3 * r**2 + h**2) / 12.0
        com = np.zeros(3)
        desc = "solid_cylinder: 一様中実円筒 Izz=mR²/2, Ixx=m(3R²+h²)/12"
    elif method == "thin_disk":
        izz = 0.5 * mass_kg * r**2
        ixx = 0.25 * mass_kg * r**2
        com = np.zeros(3)
        desc = "thin_disk: 一様薄円盤 Izz=mR²/2, Ixx=mR²/4（厚さ無視）"
    elif method == "thin_walled_cup":
        area_top = np.pi * r**2
        area_wall = 2 * np.pi * r * h
        m_top = mass_kg * area_top / (area_top + area_wall)
        m_wall = mass_kg - m_top
        z_top = 0.5 * h if body_z_is_top else -0.5 * h
        z_c = m_top * z_top / mass_kg
        izz = 0.5 * m_top * r**2 + m_wall * r**2
        ixx = (0.25 * m_top * r**2 + m_top * (z_top - z_c) ** 2) + (
            m_wall * (0.5 * r**2 + h**2 / 12.0) + m_wall * z_c**2
        )
        com = np.array([0.0, 0.0, z_c])
        desc = (
            "thin_walled_cup: 天板(薄円盤)+側壁(薄肉円筒)の一様シェル、質量は面積比配分 "
            f"(天板 {m_top / mass_kg:.2f}, 側壁 {m_wall / mass_kg:.2f})。ネジ山・リブは無視"
        )
    else:
        raise ValueError(f"unknown inertia method: {method} (choose from {INERTIA_METHODS})")
    return com, np.diag([ixx, ixx, izz]), desc


def translational_acceleration(
    cap: CapParameters, environment: Environment, aero: AerodynamicResult
) -> np.ndarray:
    """m dv/dt = F_gravity + F_aero。"""
    return environment.gravity_world_m_s2 + aero.force_world_n / cap.mass_kg


def angular_acceleration_body(
    inertia_body: np.ndarray, inertia_inv_body: np.ndarray, omega_body: np.ndarray, moment_body: np.ndarray
) -> np.ndarray:
    """オイラーの剛体方程式 I dω/dt + ω × (Iω) = M（ボディ座標）。"""
    return inertia_inv_body @ (moment_body - np.cross(omega_body, inertia_body @ omega_body))


AeroFunction = Callable[[RigidBodyState], AerodynamicResult]


def make_state_derivative(
    cap: CapParameters,
    environment: Environment,
    aero_fn: AeroFunction,
    drift_gain: float,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """13 要素状態 y=[r, v, q(wxyz), ω_body] の時間微分関数を生成する。

    RK4 / solve_ivp 双方で同じ関数を使う。力の記録には使わない（試行点でも呼ばれるため）。
    """
    inertia = cap.inertia_tensor_body_kg_m2
    inertia_inv = np.linalg.inv(inertia)

    def derivative(t: float, y: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(y)):
            raise NonFiniteStateError(f"non-finite state at t={t:.6f}")
        q_raw = y[6:10]
        state = RigidBodyState(
            time_sec=t,
            position_world_m=y[0:3],
            velocity_world_m_s=y[3:6],
            orientation_body_to_world=quat_normalize(q_raw),
            angular_velocity_body_rad_s=y[10:13],
        )
        aero = aero_fn(state)
        acc = translational_acceleration(cap, environment, aero)
        alpha = angular_acceleration_body(inertia, inertia_inv, state.angular_velocity_body_rad_s, aero.moment_body_nm)
        dq = quat_derivative(q_raw, state.angular_velocity_body_rad_s, drift_gain)
        dy = np.concatenate([state.velocity_world_m_s, acc, dq, alpha])
        if not np.all(np.isfinite(dy)):
            raise NonFiniteStateError(f"non-finite derivative at t={t:.6f}")
        return dy

    return derivative
