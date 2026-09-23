"""JSON / CSV / three.js ビューア用 JSON の出力。

ビューア JSON も物理座標（+X 捕手・+Y 左・+Z 上）のまま保存する。表示座標への変換はビューア側の
viewer/src/domain/coordinate-transform.ts の一か所だけで行う。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import SIMULATION_VERSION
from ..coordinates import quat_to_euler_deg, rad_s_to_rpm
from ..simulation import SimulationResult
from .summaries import compute_summary, frame_snapshot

VIEWER_SCHEMA_VERSION = "1.0.0"
RESULT_SCHEMA_VERSION = "1.0.0"

COORDINATE_SYSTEM = {
    "handedness": "right",
    "forwardAxis": "+X",
    "leftAxis": "+Y",
    "rightAxis": "-Y",
    "upAxis": "+Z",
    "unit": "meter",
    "origin": "投手リリース直下の地面を原点とする想定（初期位置で指定）",
    "note": "指示書原文の '+Y=右' は左手系になるため、右手系を保つよう +Y=投手から見て左 に確定",
    "quaternion": "[w, x, y, z], body→world",
    "body": "body +Z = キャップ円形面の法線, body X-Y = 円形面",
    "eulerDisplayOnly": "[roll, pitch, yaw] deg, intrinsic Z-Y'-X'' (R = Rz(yaw) Ry(pitch) Rx(roll))",
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def timeseries_dataframe(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for f in result.frames:
        s = f.state
        a = f.aero
        roll, pitch, yaw = quat_to_euler_deg(s.orientation_body_to_world)
        omega = s.angular_velocity_body_rad_s
        rows.append(
            {
                "time": s.time_sec,
                "x": s.position_world_m[0], "y": s.position_world_m[1], "z": s.position_world_m[2],
                "vx": s.velocity_world_m_s[0], "vy": s.velocity_world_m_s[1], "vz": s.velocity_world_m_s[2],
                "speed": float(np.linalg.norm(s.velocity_world_m_s)),
                "speed_kmh": float(np.linalg.norm(s.velocity_world_m_s)) * 3.6,
                "qw": s.orientation_body_to_world[0], "qx": s.orientation_body_to_world[1],
                "qy": s.orientation_body_to_world[2], "qz": s.orientation_body_to_world[3],
                "roll": roll, "pitch": pitch, "yaw": yaw,
                "wx": omega[0], "wy": omega[1], "wz": omega[2],
                "wx_world": f.omega_world_rad_s[0], "wy_world": f.omega_world_rad_s[1], "wz_world": f.omega_world_rad_s[2],
                "rpm": rad_s_to_rpm(float(np.linalg.norm(omega))),
                "normal_x": f.normal_world[0], "normal_y": f.normal_world[1], "normal_z": f.normal_world[2],
                "angle_of_attack": a.diagnostics.get("angle_of_attack_deg", np.nan),
                "sideslip": a.diagnostics.get("sideslip_deg", np.nan),
                "normal_to_wind_deg": a.diagnostics.get("normal_to_wind_deg", np.nan),
                "spin_parameter": a.diagnostics.get("spin_parameter", np.nan),
                "drag_x": a.drag_world_n[0], "drag_y": a.drag_world_n[1], "drag_z": a.drag_world_n[2],
                "lift_x": a.lift_world_n[0], "lift_y": a.lift_world_n[1], "lift_z": a.lift_world_n[2],
                "magnus_x": a.magnus_world_n[0], "magnus_y": a.magnus_world_n[1], "magnus_z": a.magnus_world_n[2],
                "total_force_x": a.force_world_n[0], "total_force_y": a.force_world_n[1], "total_force_z": a.force_world_n[2],
                "moment_x": a.moment_body_nm[0], "moment_y": a.moment_body_nm[1], "moment_z": a.moment_body_nm[2],
                "cd": a.coefficients.get("cd", np.nan),
                "cl": a.coefficients.get("cl", np.nan),
                "cy": a.coefficients.get("cy", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_result_document(result: SimulationResult) -> dict[str, Any]:
    cfg = result.config
    s0 = cfg.initial_state
    mass = cfg.cap.mass_kg
    g = cfg.environment.gravity_m_s2
    df = timeseries_dataframe(result)
    return _jsonable(
        {
            "schemaVersion": RESULT_SCHEMA_VERSION,
            "simulation_version": SIMULATION_VERSION,
            "name": cfg.name,
            "label": cfg.label,
            "coordinate_system": COORDINATE_SYSTEM,
            "status": "hypothesis_calculation",
            "disclaimer": "結果は仮定した空力係数モデルに依存する計算値であり、物理的事実として断定しない",
            "config": {
                "input": cfg.raw,
                "si": {
                    "cap": {
                        "massKg": mass,
                        "outerDiameterM": cfg.cap.outer_diameter_m,
                        "heightM": cfg.cap.height_m,
                        "referenceAreaM2": cfg.cap.reference_area_m2,
                        "centerOfMassBodyM": cfg.cap.center_of_mass_body_m,
                        "inertiaTensorBodyKgM2": cfg.cap.inertia_tensor_body_kg_m2,
                        "inertiaMethod": cfg.cap.inertia_method,
                        "surfaceOrientation": cfg.cap.surface_orientation,
                    },
                    "environment": {
                        "airDensityKgM3": cfg.environment.air_density_kg_m3,
                        "windVelocityWorldMS": cfg.environment.wind_velocity_world_m_s,
                        "gravityMS2": g,
                    },
                    "field": vars(cfg.field_geometry),
                    "initialState": {
                        "positionWorldM": s0.position_world_m,
                        "velocityWorldMS": s0.velocity_world_m_s,
                        "quaternionWxyz": s0.orientation_body_to_world,
                        "angularVelocityBodyRadS": s0.angular_velocity_body_rad_s,
                    },
                    "settings": {**vars(cfg.settings), "termination": vars(cfg.settings.termination)},
                },
                "inputUnits": cfg.input_units,
            },
            "parameter_source": cfg.parameter_sources,
            "aerodynamic_model": result.model_description,
            "assumptions": cfg.assumptions,
            "warnings": result.warnings,
            "summary": compute_summary(result),
            "events": [
                {"type": e["type"], "timeSec": e["timeSec"], **({"state": frame_snapshot(e["frame"], mass, g)} if "frame" in e else {})}
                for e in result.events
            ],
            "timeseries": df.to_dict(orient="list"),
        }
    )


def build_viewer_document(result: SimulationResult) -> dict[str, Any]:
    cfg = result.config
    frames = []
    for f in result.frames:
        s = f.state
        a = f.aero
        frames.append(
            {
                "timeSec": s.time_sec,
                "positionM": s.position_world_m,
                "quaternionWxyz": s.orientation_body_to_world,
                "velocityMS": s.velocity_world_m_s,
                "angularVelocityWorldRadS": f.omega_world_rad_s,
                "faceNormalWorld": f.normal_world,
                "rpm": rad_s_to_rpm(float(np.linalg.norm(f.omega_world_rad_s))),
                "angleOfAttackDeg": a.diagnostics.get("angle_of_attack_deg"),
                "sideslipDeg": a.diagnostics.get("sideslip_deg"),
                "forcesN": {
                    "drag": a.drag_world_n,
                    "lift": a.lift_world_n,
                    "magnus": a.magnus_world_n,
                    "total": a.force_world_n,
                },
            }
        )
    return _jsonable(
        {
            "schemaVersion": VIEWER_SCHEMA_VERSION,
            "name": cfg.name,
            "label": cfg.label,
            "coordinateSystem": COORDINATE_SYSTEM,
            "field": {"pitchDistanceM": cfg.field_geometry.pitch_distance_m, "groundZM": cfg.field_geometry.ground_z_m},
            "cap": {
                "diameterM": cfg.cap.outer_diameter_m,
                "heightM": cfg.cap.height_m,
                "modelUri": None,
                "bodyNormalAxis": "+Z",
                "surfaceOrientation": cfg.cap.surface_orientation,
            },
            "frames": frames,
            "events": [{"type": e["type"], "timeSec": e["timeSec"]} for e in result.events],
            "terminationReason": result.termination_reason,
            "warnings": result.warnings,
            "visualExaggeration": None,
        }
    )


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)


def write_result_bundle(result: SimulationResult, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"result": out / "result.json", "timeseries": out / "timeseries.csv", "viewer": out / "viewer.json"}
    write_json(paths["result"], build_result_document(result))
    timeseries_dataframe(result).to_csv(paths["timeseries"], index=False)
    write_json(paths["viewer"], build_viewer_document(result))
    return paths


def load_result_document(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
