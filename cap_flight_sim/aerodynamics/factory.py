"""空力モデルの生成（設定 dict → モデル）。"""

from __future__ import annotations

from typing import Any

from ..models import CapParameters, ForceToggles
from .base import AerodynamicModel
from .lookup_table import LookupTableAerodynamicModel, load_coefficient_table
from .simple import DragParams, LiftParams, MagnusParams, MomentParams, SimpleAerodynamicModel


def build_aerodynamic_model(section: dict[str, Any], cap: CapParameters) -> AerodynamicModel:
    enable = section.get("enable", {}) or {}
    toggles = ForceToggles(**{k: bool(v) for k, v in enable.items()})
    model = section.get("model", "simple")
    if model == "simple":
        moments = dict(section.get("moments", {}) or {})
        cp = moments.get("center_of_pressure_body_m")
        if cp is not None:
            moments["center_of_pressure_body_m"] = tuple(float(v) for v in cp)
        return SimpleAerodynamicModel(
            drag=DragParams(**(section.get("drag", {}) or {})),
            lift=LiftParams(**(section.get("lift", {}) or {})),
            magnus=MagnusParams(**(section.get("magnus", {}) or {})),
            moments=MomentParams(**moments),
            toggles=toggles,
        )
    if model == "lookup_table":
        spec = section.get("lookup_table", {}) or {}
        if "path" not in spec:
            raise ValueError("aerodynamics.lookup_table.path is required")
        table = load_coefficient_table(spec["path"], cap.surface_orientation)
        return LookupTableAerodynamicModel(table=table, out_of_range=spec.get("out_of_range", "clamp"), toggles=toggles)
    raise ValueError(f"unknown aerodynamics.model: {model} (simple | lookup_table)")
