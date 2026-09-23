"""CSV / Parquet の空力係数表を読み込むモデル（CFD・風洞・実測との交換口）。

係数ファイルのスキーマ（schema_version 1.0.0）:
    軸列（使うものだけ・直交格子であること）:
        speed_m_s, angle_of_attack_deg, sideslip_deg, spin_parameter
    カテゴリ列（任意）:
        surface_orientation  … body_z_is_top / body_z_is_cavity。キャップ設定と一致する行だけを使う
    係数列（無い列は 0 とみなし警告）:
        cd, cl, cy, cm_pitch, cm_roll, cm_yaw
    メタ列（任意・全行同一値）:
        source … measured / fitted / literature / assumed

力・モーメントの方向は base.py の空力座標に従う:
    抗力 e_drag、揚力 e_lift、横力 cy は unit(ω⊥ × u)（マグヌス成分として出力）、
    cm_roll / cm_pitch / cm_yaw は e_inplane / e_pitch / n まわり、基準長は外径 D。
OpenFOAM 等の固有形式はここへ持ち込まず、事前にこのスキーマへ変換すること。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from ..models import AerodynamicResult, CapParameters, Environment, ForceToggles, RigidBodyState
from .base import compute_flow_geometry, magnus_direction

TABLE_SCHEMA_VERSION = "1.0.0"
AXIS_COLUMNS = ("speed_m_s", "angle_of_attack_deg", "sideslip_deg", "spin_parameter")
COEFFICIENT_COLUMNS = ("cd", "cl", "cy", "cm_pitch", "cm_roll", "cm_yaw")
OUT_OF_RANGE_POLICIES = ("clamp", "nearest", "error")


class OutOfTableRangeError(ValueError):
    pass


@dataclass
class CoefficientTable:
    axes: tuple[str, ...]
    grid: tuple[np.ndarray, ...]
    values: dict[str, np.ndarray]
    source: str
    path: str
    missing_coefficients: tuple[str, ...]
    _linear: dict[str, RegularGridInterpolator] = field(default_factory=dict, repr=False)
    _nearest: dict[str, RegularGridInterpolator] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name, arr in self.values.items():
            self._linear[name] = RegularGridInterpolator(self.grid, arr, method="linear")
            self._nearest[name] = RegularGridInterpolator(self.grid, arr, method="nearest")

    def lookup(self, point: dict[str, float], policy: str) -> tuple[dict[str, float], bool]:
        """(係数, 範囲外だったか)。"""
        coords = np.array([point[a] for a in self.axes], dtype=float)
        lo = np.array([g[0] for g in self.grid])
        hi = np.array([g[-1] for g in self.grid])
        out = bool(np.any(coords < lo) or np.any(coords > hi))
        if out and policy == "error":
            raise OutOfTableRangeError(f"query {dict(zip(self.axes, coords))} outside table {self.path}")
        clipped = np.clip(coords, lo, hi)
        interps = self._nearest if (out and policy == "nearest") else self._linear
        result = {name: float(interp(clipped)[0]) for name, interp in interps.items()}
        for name in self.missing_coefficients:
            result[name] = 0.0
        return result, out


def load_coefficient_table(path: str | Path, surface_orientation: str | None = None) -> CoefficientTable:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if "surface_orientation" in df.columns and surface_orientation is not None:
        df = df[df["surface_orientation"] == surface_orientation]
        if df.empty:
            raise ValueError(f"{p}: no rows for surface_orientation={surface_orientation}")

    axes = tuple(a for a in AXIS_COLUMNS if a in df.columns)
    if not axes:
        raise ValueError(f"{p}: at least one axis column required from {AXIS_COLUMNS}")
    present = tuple(c for c in COEFFICIENT_COLUMNS if c in df.columns)
    if not present:
        raise ValueError(f"{p}: at least one coefficient column required from {COEFFICIENT_COLUMNS}")
    missing = tuple(c for c in COEFFICIENT_COLUMNS if c not in df.columns)

    grid = tuple(np.sort(df[a].unique()).astype(float) for a in axes)
    for a, g in zip(axes, grid):
        if len(g) < 2:
            raise ValueError(f"{p}: axis {a} needs at least 2 distinct values (drop the column if constant)")
    expected = int(np.prod([len(g) for g in grid]))
    if len(df) != expected or df.duplicated(list(axes)).any():
        raise ValueError(f"{p}: table must be a full rectangular grid without duplicates ({len(df)} rows, expected {expected})")

    ordered = df.sort_values(list(axes))
    shape = tuple(len(g) for g in grid)
    values = {c: ordered[c].to_numpy(dtype=float).reshape(shape) for c in present}

    source = "assumed"
    if "source" in df.columns:
        sources = df["source"].unique()
        if len(sources) != 1:
            raise ValueError(f"{p}: mixed 'source' values {list(sources)} are not allowed")
        source = str(sources[0])

    return CoefficientTable(axes=axes, grid=grid, values=values, source=source, path=str(p), missing_coefficients=missing)


@dataclass
class LookupTableAerodynamicModel:
    table: CoefficientTable
    out_of_range: str = "clamp"
    toggles: ForceToggles = field(default_factory=ForceToggles)
    name: str = "lookup_table_model"
    out_of_range_count: int = 0

    def __post_init__(self) -> None:
        if self.out_of_range not in OUT_OF_RANGE_POLICIES:
            raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE_POLICIES}")

    def evaluate(self, state: RigidBodyState, cap: CapParameters, environment: Environment) -> AerodynamicResult:
        flow = compute_flow_geometry(state, cap, environment)
        zero = np.zeros(3)
        if flow.is_stationary:
            coeffs = {c: 0.0 for c in COEFFICIENT_COLUMNS}
            return AerodynamicResult(zero, zero, zero, zero, zero, coeffs, {"stationary": True, "out_of_table_range": False})

        point = {
            "speed_m_s": flow.speed_m_s,
            "angle_of_attack_deg": float(np.rad2deg(flow.angle_of_attack_rad)),
            "sideslip_deg": float(np.rad2deg(flow.sideslip_rad)),
            "spin_parameter": flow.spin_parameter,
        }
        coeffs, out = self.table.lookup(point, self.out_of_range)
        if out:
            self.out_of_range_count += 1

        qa = flow.dynamic_pressure_pa * cap.reference_area_m2
        drag = -qa * coeffs["cd"] * flow.u_hat_world if self.toggles.drag else zero
        lift = qa * coeffs["cl"] * flow.lift_dir_world if self.toggles.lift else zero
        magnus = qa * coeffs["cy"] * magnus_direction(flow) if self.toggles.magnus else zero
        force = drag + lift + magnus

        moment = zero
        if self.toggles.moments:
            qad = qa * cap.outer_diameter_m
            moment = qad * (
                coeffs["cm_roll"] * flow.inplane_dir_body
                + coeffs["cm_pitch"] * flow.pitch_axis_body
                + coeffs["cm_yaw"] * np.array([0.0, 0.0, 1.0])
            )

        return AerodynamicResult(
            force_world_n=force,
            moment_body_nm=moment,
            drag_world_n=drag,
            lift_world_n=lift,
            magnus_world_n=magnus,
            coefficients=coeffs,
            diagnostics={
                **{k: float(v) for k, v in point.items()},
                "normal_to_wind_deg": flow.normal_to_wind_deg,
                "out_of_table_range": out,
                "stationary": False,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table_schema_version": TABLE_SCHEMA_VERSION,
            "table_path": self.table.path,
            "axes": list(self.table.axes),
            "axis_ranges": {a: [float(g[0]), float(g[-1])] for a, g in zip(self.table.axes, self.table.grid)},
            "coefficient_source": self.table.source,
            "missing_coefficients_treated_as_zero": list(self.table.missing_coefficients),
            "out_of_range_policy": self.out_of_range,
            "toggles": vars(self.toggles),
        }
