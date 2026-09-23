"""結果の要約指標（着弾点だけに頼らない比較用）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..coordinates import angle_between_deg, quat_to_euler_deg, rad_s_to_rpm, safe_unit
from ..simulation import Frame, SimulationResult


def _vec(v: np.ndarray) -> list[float]:
    return [float(x) for x in v]


def frame_snapshot(frame: Frame, mass_kg: float, gravity_m_s2: float) -> dict[str, Any]:
    s = frame.state
    omega = frame.omega_world_rad_s
    return {
        "timeSec": s.time_sec,
        "positionWorldM": _vec(s.position_world_m),
        "yM": float(s.position_world_m[1]),
        "zM": float(s.position_world_m[2]),
        "velocityWorldMS": _vec(s.velocity_world_m_s),
        "speedMS": float(np.linalg.norm(s.velocity_world_m_s)),
        "quaternionWxyz": _vec(s.orientation_body_to_world),
        "eulerRollPitchYawDegDisplayOnly": _vec(quat_to_euler_deg(s.orientation_body_to_world)),
        "faceNormalWorld": _vec(frame.normal_world),
        "rpm": rad_s_to_rpm(float(np.linalg.norm(omega))),
        "spinAxisWorld": _vec(safe_unit(omega)),
        "forcesN": {
            "drag": _vec(frame.aero.drag_world_n),
            "lift": _vec(frame.aero.lift_world_n),
            "magnus": _vec(frame.aero.magnus_world_n),
            "aeroTotal": _vec(frame.aero.force_world_n),
            "gravity": [0.0, 0.0, -mass_kg * gravity_m_s2],
        },
        "momentBodyNm": _vec(frame.aero.moment_body_nm),
        "coefficients": {k: float(v) for k, v in frame.aero.coefficients.items()},
    }


def compute_summary(result: SimulationResult) -> dict[str, Any]:
    frames = result.frames
    cfg = result.config
    mass = cfg.cap.mass_kg
    g = cfg.environment.gravity_m_s2
    t = np.array([f.state.time_sec for f in frames])
    pos = np.array([f.state.position_world_m for f in frames])
    vel = np.array([f.state.velocity_world_m_s for f in frames])
    force = np.array([f.aero.force_world_n for f in frames])
    normals = np.array([f.normal_world for f in frames])
    rpm = np.array([rad_s_to_rpm(np.linalg.norm(f.omega_world_rad_s)) for f in frames])

    p0, v0 = pos[0], vel[0]
    line = p0 + np.outer(t, v0)  # 初速方向の直線（初期投射方向）
    gravity_only = line - np.outer(0.5 * g * t**2, [0.0, 0.0, 1.0])
    lateral_break = pos[:, 1] - line[:, 1]  # 投射方向と分離した横変化

    catcher = result.catcher_plane_event
    catcher_block = None
    if catcher is not None:
        catcher_block = {**frame_snapshot(catcher["frame"], mass, g), "terminationReason": result.termination_reason}
        tc = catcher["timeSec"]
        pc = catcher["frame"].state.position_world_m
        catcher_block["lateralBreakFromLaunchLineM"] = float(pc[1] - (p0[1] + v0[1] * tc))
        catcher_block["dropFromLaunchLineM"] = float((p0[2] + v0[2] * tc) - pc[2])

    return {
        "terminationReason": result.termination_reason,
        "reachedCatcherPlane": catcher is not None,
        "flightTimeSec": float(t[-1]),
        "finalPositionWorldM": _vec(pos[-1]),
        "finalSpeedMS": float(np.linalg.norm(vel[-1])),
        "initialSpeedMS": float(np.linalg.norm(v0)),
        "initialLateralVelocityMS": float(v0[1]),
        "initialVerticalVelocityMS": float(v0[2]),
        "meanAeroLateralAccelMS2": float(np.mean(force[:, 1]) / mass),
        "meanAeroVerticalAccelMS2": float(np.mean(force[:, 2]) / mass),
        "maxAbsLateralDisplacementM": float(np.max(np.abs(pos[:, 1] - p0[1]))),
        "lateralBreakFromLaunchLineM": float(lateral_break[-1]),
        "maxAbsLateralBreakM": float(np.max(np.abs(lateral_break))),
        "dropFromLaunchLineM": float(line[-1, 2] - pos[-1, 2]),
        "verticalAeroEffectVsGravityOnlyM": float(pos[-1, 2] - gravity_only[-1, 2]),
        "faceNormalChangeDeg": angle_between_deg(normals[0], normals[-1]),
        "maxFaceNormalChangeDeg": float(max(angle_between_deg(normals[0], n) for n in normals)),
        "rpmInitial": float(rpm[0]),
        "rpmFinal": float(rpm[-1]),
        "meanDragN": float(np.mean(np.linalg.norm([f.aero.drag_world_n for f in frames], axis=1))),
        "meanLiftN": float(np.mean(np.linalg.norm([f.aero.lift_world_n for f in frames], axis=1))),
        "meanMagnusN": float(np.mean(np.linalg.norm([f.aero.magnus_world_n for f in frames], axis=1))),
        "weightN": float(mass * g),
        "catcherPlane": catcher_block,
        "notes": [
            "lateralBreak は初速方向の直線からの横ずれ（初期投射方向による横移動を含まない）",
            "Y 正は投手から見て左。右への変化は負値",
        ],
    }
