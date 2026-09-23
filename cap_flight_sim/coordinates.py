"""座標・クォータニオン・単位変換（副作用のない純関数のみ）。

ワールド座標（右手系・FLU）:
    +X: 投手 → 捕手
    +Y: 投手から見て「左」
    +Z: 鉛直上
    ※ 指示書原文の「+Y=右」は左手系になるため、右手系を保つよう +Y=左 に確定した。
      「右へ曲がる」は -Y 方向の変位として現れる。

ボディ座標:
    body +Z: キャップ円形面の法線（surface_orientation で天面側か空洞側かを明示）
    body X-Y: キャップ円形面

クォータニオン:
    [w, x, y, z]（スカラー先頭）。ボディ座標 → ワールド座標の回転を表す。
    v_world = R(q) @ v_body
    scipy.spatial.transform.Rotation は [x, y, z, w] 順。変換は wxyz_to_xyzw / xyzw_to_wxyz に集約する。

オイラー角（表示・入力専用。積分には使わない）:
    [roll, pitch, yaw] [deg]、内在的 Z-Y'-X''（航空機の yaw→pitch→roll）
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    右手系 FLU のため、+pitch は body +X を下（-Z）へ向ける回転になる。
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12

# ---------------------------------------------------------------- 単位変換


def kmh_to_m_s(v: float) -> float:
    return float(v) / 3.6


def m_s_to_kmh(v: float) -> float:
    return float(v) * 3.6


def rpm_to_rad_s(rpm: float) -> float:
    return float(rpm) * 2.0 * np.pi / 60.0


def rad_s_to_rpm(w: float) -> float:
    return float(w) * 60.0 / (2.0 * np.pi)


# ---------------------------------------------------------------- ベクトル


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def safe_unit(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    """単位ベクトル。ほぼゼロなら零ベクトルを返す（NaN を出さない）。"""
    arr = np.asarray(v, dtype=float)
    n = np.linalg.norm(arr)
    if n < eps:
        return np.zeros_like(arr)
    return arr / n


def component_perpendicular(v: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    """v から direction_unit 方向成分を除いた垂直成分。"""
    return v - np.dot(v, direction_unit) * direction_unit


# ---------------------------------------------------------------- クォータニオン [w, x, y, z]


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < EPS or not np.isfinite(n):
        raise ValueError(f"quaternion cannot be normalized: {q}")
    return np.asarray(q, dtype=float) / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 積 a ⊗ b。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """ボディ→ワールド回転行列。q は正規化済みを想定。"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """回転行列 → [w, x, y, z]（w >= 0 に正規化）。"""
    m = np.asarray(m, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = quat_normalize(np.array(q))
    return q if q[0] >= 0 else -q


def body_to_world(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q) @ v_body


def world_to_body(q: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q).T @ v_world


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    u = safe_unit(np.asarray(axis, dtype=float))
    if not u.any():
        return quat_identity()
    h = 0.5 * angle_rad
    return np.concatenate([[np.cos(h)], np.sin(h) * u])


def quat_derivative(q: np.ndarray, omega_body: np.ndarray, drift_gain: float = 0.0) -> np.ndarray:
    """dq/dt = 0.5 q ⊗ [0, ω_body] + k (1 - |q|²) q。

    第2項はノルムを 1 へ引き戻す拘束安定化項（solve_ivp のステップ間で正規化できないため）。
    """
    dq = 0.5 * quat_multiply(q, np.concatenate([[0.0], omega_body]))
    if drift_gain:
        dq = dq + drift_gain * (1.0 - float(np.dot(q, q))) * q
    return dq


def euler_deg_to_quat(roll_pitch_yaw_deg: np.ndarray) -> np.ndarray:
    """[roll, pitch, yaw] deg（内在的 Z-Y'-X''）→ [w, x, y, z]。"""
    roll, pitch, yaw = np.deg2rad(np.asarray(roll_pitch_yaw_deg, dtype=float))
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch)
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll)
    return quat_normalize(quat_multiply(quat_multiply(qz, qy), qx))


def quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → [roll, pitch, yaw] deg（表示専用。pitch=±90° 付近でジンバルロック）。"""
    m = quat_to_matrix(quat_normalize(q))
    pitch = np.arcsin(np.clip(-m[2, 0], -1.0, 1.0))
    if abs(m[2, 0]) < 1.0 - 1e-9:
        roll = np.arctan2(m[2, 1], m[2, 2])
        yaw = np.arctan2(m[1, 0], m[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-m[0, 1], m[1, 1])
    return np.rad2deg(np.array([roll, pitch, yaw]))


def quat_from_face_normal(normal_world: np.ndarray, body_x_hint_world: np.ndarray | None = None) -> np.ndarray:
    """body +Z を normal_world に合わせる姿勢。body +X は hint を面内へ射影した方向。"""
    z = safe_unit(np.asarray(normal_world, dtype=float))
    if not z.any():
        raise ValueError("face_normal_world must be non-zero")
    hint = np.array([1.0, 0.0, 0.0]) if body_x_hint_world is None else np.asarray(body_x_hint_world, dtype=float)
    x = component_perpendicular(hint, z)
    if np.linalg.norm(x) < 1e-6:
        fallback = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = component_perpendicular(fallback, z)
    x = safe_unit(x)
    y = np.cross(z, x)
    return matrix_to_quat(np.column_stack([x, y, z]))


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """本パッケージ [w,x,y,z] → scipy / three.js 系 [x,y,z,w]。変換はここだけで行う。"""
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = safe_unit(a), safe_unit(b)
    if not ua.any() or not ub.any():
        return float("nan")
    return float(np.rad2deg(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0))))
