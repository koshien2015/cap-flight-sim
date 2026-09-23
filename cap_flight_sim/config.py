"""YAML 設定の読み込みと SI 変換。

数値パラメータは次のどちらの書き方も可:
    mass_kg: 0.0022
    mass_kg: {value: 0.0022, source: measured, note: "電子天秤 0.1mg"}
source を省略した数値は assumed として記録する（仮定値と実測値を混在させないため）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .coordinates import (
    angle_between_deg,
    euler_deg_to_quat,
    kmh_to_m_s,
    quat_from_face_normal,
    quat_normalize,
    quat_to_matrix,
    rpm_to_rad_s,
    safe_unit,
)
from .models import (
    PARAMETER_SOURCES,
    CapParameters,
    Environment,
    FieldGeometry,
    RigidBodyState,
    SimulationConfig,
    SimulationSettings,
    TerminationSettings,
)
from .rigid_body import approximate_mass_properties

TRACKED_SECTIONS = ("cap", "environment", "aerodynamics")
DEFAULT_VIDEO_FPS = 240.0
DEFAULT_DRIFT_GAIN = 10.0
MAX_STEP_CEILING_SEC = 0.002


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- 汎用ユーティリティ


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level must be a mapping")
    return data


ORIENTATION_KEYS = ("orientation_quaternion_wxyz", "orientation_euler_deg", "face_normal_world", "body_x_hint_world")
REPLACE_WHOLE_KEYS = ("angular_velocity",)  # 書式が複数あるため継承時は丸ごと置き換える


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """base を変更せずに override を再帰的に重ねた新しい dict を返す。

    姿勢指定（ORIENTATION_KEYS）は override 側に 1 つでもあれば base 側の姿勢指定を捨てる。
    """
    merged = copy.deepcopy(base)
    if any(k in override for k in ORIENTATION_KEYS[:3]):
        merged = {k: v for k, v in merged.items() if k not in ORIENTATION_KEYS}
    for key, value in override.items():
        if key in REPLACE_WHOLE_KEYS:
            merged[key] = copy.deepcopy(value)
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and not _is_param_spec(value):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """{"aerodynamics.lift.cl0": 0.1} 形式のドット区切り上書きを適用した新しい dict。"""
    result = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = result
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        if keys[-1] in ORIENTATION_KEYS[:3]:
            for other in ORIENTATION_KEYS:
                node.pop(other, None)
        node[keys[-1]] = copy.deepcopy(value)
    return result


def get_dotted(raw: dict[str, Any], dotted: str) -> Any:
    node: Any = raw
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(dotted)
        node = node[k]
    return _unwrap(node)[0] if _is_param_spec(node) else node


def _is_param_spec(value: Any) -> bool:
    return isinstance(value, dict) and "value" in value and set(value) <= {"value", "source", "note"}


def _unwrap(value: Any) -> tuple[Any, str | None, str | None]:
    if _is_param_spec(value):
        source = value.get("source")
        if source is not None and source not in PARAMETER_SOURCES:
            raise ConfigError(f"invalid source '{source}', choose from {PARAMETER_SOURCES}")
        return value["value"], source, value.get("note")
    return value, None, None


def resolve_parameters(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """{value, source} を素の値へ展開し、追跡対象セクションの数値の出典を集める。"""
    sources: dict[str, dict[str, Any]] = {}

    def walk(node: Any, path: str, tracked: bool) -> Any:
        value, source, note = _unwrap(node)
        if isinstance(value, dict):
            return {k: walk(v, f"{path}.{k}" if path else k, tracked or k in TRACKED_SECTIONS) for k, v in value.items()}
        if isinstance(value, list):
            resolved = [walk(v, f"{path}[{i}]", False) for i, v in enumerate(value)]
            if tracked and resolved and all(isinstance(v, (int, float)) for v in resolved):
                sources[path] = {"value": resolved, "source": source or "assumed", **({"note": note} if note else {})}
            return resolved
        if tracked and isinstance(value, (int, float)) and not isinstance(value, bool):
            sources[path] = {"value": value, "source": source or "assumed", **({"note": note} if note else {})}
        return value

    return walk(raw, "", False), sources


# ---------------------------------------------------------------- セクション変換


def _vec(value: Any, name: str, size: int = 3) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,):
        raise ConfigError(f"{name} must have {size} elements, got {value}")
    return arr


def build_cap(section: dict[str, Any], assumptions: list[str]) -> CapParameters:
    mass = float(section["mass_kg"])
    diameter = float(section["outer_diameter_m"])
    height = float(section["height_m"])
    if mass <= 0 or diameter <= 0 or height <= 0:
        raise ConfigError("cap mass/diameter/height must be positive")
    surface = section.get("surface_orientation", "body_z_is_top")
    if surface not in ("body_z_is_top", "body_z_is_cavity"):
        raise ConfigError("cap.surface_orientation must be body_z_is_top or body_z_is_cavity")

    area = section.get("reference_area_m2")
    if area is None:
        area = np.pi * (0.5 * diameter) ** 2
        assumptions.append(
            f"reference_area_m2 未指定のため円形面積 πR² = {area:.4e} m² を使用（エッジオン時の実効投影面積とは別物）"
        )

    method = section.get("inertia_method", "thin_walled_cup")
    com_approx, inertia_approx, desc = approximate_mass_properties(
        mass, diameter, height, method, surface == "body_z_is_top"
    )
    com_raw = section.get("center_of_mass_body_m")
    com = com_approx if com_raw is None else _vec(com_raw, "center_of_mass_body_m")
    if com_raw is None:
        assumptions.append(f"重心は慣性近似 {method} から算出: body {np.round(com, 6).tolist()} m")

    inertia_raw = section.get("inertia_tensor_body_kg_m2")
    if inertia_raw is None:
        inertia = inertia_approx
        inertia_method = desc
        assumptions.append(f"慣性テンソル未指定のため近似: {desc}")
    else:
        inertia = np.asarray(inertia_raw, dtype=float)
        if inertia.shape == (3,):
            inertia = np.diag(inertia)
        if inertia.shape != (3, 3) or not np.allclose(inertia, inertia.T):
            raise ConfigError("inertia_tensor_body_kg_m2 must be 3 values (diagonal) or a symmetric 3x3 matrix")
        if np.any(np.linalg.eigvalsh(inertia) <= 0):
            raise ConfigError("inertia tensor must be positive definite")
        inertia_method = "specified"

    return CapParameters(
        mass_kg=mass,
        outer_diameter_m=diameter,
        height_m=height,
        reference_area_m2=float(area),
        center_of_mass_body_m=com,
        inertia_tensor_body_kg_m2=inertia,
        surface_orientation=surface,
        inertia_method=inertia_method,
    )


def build_environment(section: dict[str, Any]) -> Environment:
    return Environment(
        air_density_kg_m3=float(section.get("air_density_kg_m3", 1.204)),
        wind_velocity_world_m_s=_vec(section.get("wind_velocity_world_m_s", [0.0, 0.0, 0.0]), "wind_velocity_world_m_s"),
        gravity_m_s2=float(section.get("gravity_m_s2", 9.80665)),
    )


def build_velocity(section: dict[str, Any], units: dict[str, Any]) -> np.ndarray:
    """初速ベクトル。方位角は +Y（投手から見て左）向きを正、仰角は上向きを正。"""
    if "velocity_world_m_s" in section:
        return _vec(section["velocity_world_m_s"], "velocity_world_m_s")
    if "speed_kmh" in section:
        speed = kmh_to_m_s(section["speed_kmh"])
        units["speed"] = {"input": section["speed_kmh"], "input_unit": "km/h", "si": speed, "si_unit": "m/s"}
    elif "speed_m_s" in section:
        speed = float(section["speed_m_s"])
    else:
        raise ConfigError("initial_state needs speed_kmh, speed_m_s or velocity_world_m_s")
    az = np.deg2rad(float(section.get("launch_azimuth_deg", 0.0)))
    el = np.deg2rad(float(section.get("launch_elevation_deg", 0.0)))
    return speed * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def build_orientation(section: dict[str, Any]) -> np.ndarray:
    keys = [k for k in ("orientation_quaternion_wxyz", "orientation_euler_deg", "face_normal_world") if k in section]
    if len(keys) != 1:
        raise ConfigError(
            "initial_state needs exactly one of orientation_quaternion_wxyz / orientation_euler_deg / face_normal_world"
        )
    key = keys[0]
    if key == "orientation_quaternion_wxyz":
        return quat_normalize(_vec(section[key], key, 4))
    if key == "orientation_euler_deg":
        return euler_deg_to_quat(_vec(section[key], key))
    hint = section.get("body_x_hint_world")
    return quat_from_face_normal(_vec(section[key], key), None if hint is None else _vec(hint, "body_x_hint_world"))


def build_angular_velocity_body(spec: dict[str, Any] | None, q: np.ndarray, units: dict[str, Any]) -> np.ndarray:
    """角速度（ボディ座標）。ワールド指定は ω_body = Rᵀ ω_world で変換する。

    書式:
        omega_world_rad_s: [wx, wy, wz]
        omega_body_rad_s:  [wx, wy, wz]
        rpm | rad_s + axis_world | axis_body          （右手の法則で軸方向を正）
        rpm | rad_s + rotation_sense + viewed_from_world
            viewed_from_world: キャップから観測者への方向。その位置から見た時計/反時計回り。
    """
    if spec is None:
        return np.zeros(3)
    forms = [k for k in ("omega_world_rad_s", "omega_body_rad_s", "rotation_sense", "axis_world", "axis_body") if k in spec]
    if len(forms) > 1:
        raise ConfigError(f"angular_velocity has conflicting forms {forms}; use exactly one")
    rotation = quat_to_matrix(q)
    if "omega_world_rad_s" in spec:
        return rotation.T @ _vec(spec["omega_world_rad_s"], "omega_world_rad_s")
    if "omega_body_rad_s" in spec:
        return _vec(spec["omega_body_rad_s"], "omega_body_rad_s")

    if "rpm" in spec:
        rate = rpm_to_rad_s(spec["rpm"])
        units["spin"] = {"input": spec["rpm"], "input_unit": "rpm", "si": rate, "si_unit": "rad/s"}
    elif "rad_s" in spec:
        rate = float(spec["rad_s"])
    else:
        raise ConfigError("angular_velocity needs omega_world_rad_s, omega_body_rad_s, rpm or rad_s")

    if "rotation_sense" in spec:
        sense = spec["rotation_sense"]
        if sense not in ("clockwise", "counterclockwise"):
            raise ConfigError("rotation_sense must be clockwise or counterclockwise")
        if "viewed_from_world" not in spec:
            raise ConfigError("rotation_sense requires viewed_from_world (direction from cap to observer)")
        toward_viewer = safe_unit(_vec(spec["viewed_from_world"], "viewed_from_world"))
        if not toward_viewer.any():
            raise ConfigError("viewed_from_world must be non-zero")
        # 観測者から見て反時計回り = 観測者方向を向く角速度ベクトル（右手の法則）
        axis_world = toward_viewer if sense == "counterclockwise" else -toward_viewer
        return rotation.T @ (rate * axis_world)

    if "axis_world" in spec:
        axis = safe_unit(_vec(spec["axis_world"], "axis_world"))
        if not axis.any():
            raise ConfigError("axis_world must be non-zero")
        return rotation.T @ (rate * axis)
    if "axis_body" in spec:
        axis = safe_unit(_vec(spec["axis_body"], "axis_body"))
        if not axis.any():
            raise ConfigError("axis_body must be non-zero")
        return rate * axis
    raise ConfigError("angular_velocity with rpm/rad_s needs axis_world, axis_body or rotation_sense+viewed_from_world")


def build_settings(section: dict[str, Any], omega_body: np.ndarray, warnings: list[str]) -> SimulationSettings:
    video_fps = float(section.get("video_fps", DEFAULT_VIDEO_FPS))
    spin = float(np.linalg.norm(omega_body))
    period = 2 * np.pi / spin if spin > 0 else np.inf
    auto = min(period / 20.0, 1.0 / (4.0 * video_fps), MAX_STEP_CEILING_SEC)
    given = section.get("max_step_sec")
    if given is None:
        max_step = auto
        reason = f"auto: min(回転周期/20={period / 20:.2e}, 1/(4·fps)={1 / (4 * video_fps):.2e}, {MAX_STEP_CEILING_SEC})"
    else:
        max_step = float(given)
        reason = "specified"
        if max_step > period / 10.0:
            warnings.append(f"max_step_sec={max_step} は回転周期 {period:.4f}s の1/10より大きい（回転の解像度不足の恐れ）")

    solver = section.get("solver", "solve_ivp")
    if solver not in ("solve_ivp", "rk4"):
        raise ConfigError("simulation.solver must be solve_ivp or rk4")
    term = section.get("termination", {}) or {}
    return SimulationSettings(
        duration_sec=float(section.get("duration_sec", 2.0)),
        output_hz=float(section.get("output_hz", 240.0)),
        max_step_sec=max_step,
        max_step_reason=reason,
        solver=solver,
        method=str(section.get("method", "DOP853")),
        rtol=float(section.get("rtol", 1e-7)),
        atol=float(section.get("atol", 1e-9)),
        fixed_step_sec=float(section.get("fixed_step_sec", min(max_step, 1e-4))),
        quaternion_drift_gain=float(section.get("quaternion_drift_gain", DEFAULT_DRIFT_GAIN)),
        video_fps=video_fps,
        termination=TerminationSettings(
            stop_at_ground=bool(term.get("stop_at_ground", True)),
            stop_at_catcher_plane=bool(term.get("stop_at_catcher_plane", True)),
            max_speed_m_s=float(term.get("max_speed_m_s", 100.0)),
            max_angular_speed_rad_s=float(term.get("max_angular_speed_rad_s", 5000.0)),
        ),
    )


# ---------------------------------------------------------------- エントリポイント


def config_from_dict(raw_input: dict[str, Any], name: str = "run") -> SimulationConfig:
    raw, sources = resolve_parameters(raw_input)
    assumptions: list[str] = []
    warnings: list[str] = []
    units: dict[str, Any] = {}

    for section in ("cap", "initial_state"):
        if section not in raw:
            raise ConfigError(f"missing section: {section}")

    cap = build_cap(raw["cap"], assumptions)
    environment = build_environment(raw.get("environment", {}) or {})
    field_raw = raw.get("field", {}) or {}
    field_geometry = FieldGeometry(
        pitch_distance_m=float(field_raw.get("pitch_distance_m", 9.22)),
        ground_z_m=float(field_raw.get("ground_z_m", 0.0)),
    )

    init = raw["initial_state"]
    q0 = build_orientation(init)
    omega_body = build_angular_velocity_body(init.get("angular_velocity"), q0, units)
    initial_state = RigidBodyState(
        time_sec=0.0,
        position_world_m=_vec(init.get("position_world_m", [0.0, 0.0, 1.45]), "position_world_m"),
        velocity_world_m_s=build_velocity(init, units),
        orientation_body_to_world=q0,
        angular_velocity_body_rad_s=omega_body,
    )
    settings = build_settings(raw.get("simulation", {}) or {}, omega_body, warnings)

    normal0 = quat_to_matrix(q0)[:, 2]
    if np.linalg.norm(omega_body) > 0:
        tilt = angle_between_deg(quat_to_matrix(q0) @ omega_body, normal0)
        spin_axis_angle = min(tilt, 180.0 - tilt)
        assumptions.append(f"初期回転軸と面法線のなす角 = {spin_axis_angle:.1f}°（0°=面法線まわりの自転＝フリスビー/車輪型、90°=直径まわりのタンブリング）")
    assumptions.append(f"初期面法線(world) = {np.round(normal0, 4).tolist()}")

    aero = raw.get("aerodynamics", {}) or {}
    if not any(k.startswith("aerodynamics.") for k in sources) and aero.get("model", "simple") == "simple":
        warnings.append("空力係数が未指定: 既定値（抗力のみ・揚力/マグヌス/モーメント=0）で計算")

    return SimulationConfig(
        name=str(raw.get("name", name)),
        label=str(raw.get("label", "virtual")),
        cap=cap,
        environment=environment,
        field_geometry=field_geometry,
        settings=settings,
        initial_state=initial_state,
        aerodynamics=aero,
        raw=raw_input,
        parameter_sources=sources,
        input_units=units,
        assumptions=assumptions,
        warnings=warnings,
    )


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """YAML を読み、extends の継承と係数表パスの相対解決を適用した raw dict。"""
    p = Path(path)
    raw = load_yaml(p)
    if "extends" in raw:
        base = load_raw_config((p.parent / raw["extends"]).resolve())
        raw = deep_merge(base, {k: v for k, v in raw.items() if k != "extends"})
    table = (raw.get("aerodynamics") or {}).get("lookup_table") or {}
    path_value = table.get("path")
    if path_value and not Path(path_value).is_absolute():
        raw = apply_overrides(raw, {"aerodynamics.lookup_table.path": str((p.parent / path_value).resolve())})
    return raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SimulationConfig:
    raw = load_raw_config(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return config_from_dict(raw, name=Path(path).stem)
