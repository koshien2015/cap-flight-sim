"""簡易モデルを係数表スキーマ（lookup_table.py 参照）の CSV に書き出す。

CFD 側と交換するファイル形式の見本であり、中身は簡易モデルの仮定値（source=assumed）。
回転減衰モーメントは角速度依存のため表に含めない（cm_roll/cm_yaw は 0）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from .factory import build_aerodynamic_model
from .simple import SimpleAerodynamicModel

AOA_GRID_DEG = np.arange(-90.0, 90.0 + 1e-9, 5.0)
SPIN_PARAMETER_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])


def tabulate_simple_model(config_path: str | Path, output_path: str | Path) -> Path:
    config = load_config(config_path)
    model = build_aerodynamic_model(config.aerodynamics, config.cap)
    if not isinstance(model, SimpleAerodynamicModel):
        raise ValueError("tabulate requires aerodynamics.model: simple")
    rows = []
    for aoa in AOA_GRID_DEG:
        alpha = np.deg2rad(aoa)
        cd, _ = model.drag_coefficient(alpha, config.cap)
        for s in SPIN_PARAMETER_GRID:
            rows.append(
                {
                    "angle_of_attack_deg": aoa,
                    "spin_parameter": s,
                    "surface_orientation": config.cap.surface_orientation,
                    "cd": cd,
                    "cl": model.lift_coefficient(alpha),
                    "cy": model.magnus_coefficient(float(s)),
                    "cm_pitch": model.pitch_moment_coefficient(alpha),
                    "cm_roll": 0.0,
                    "cm_yaw": 0.0,
                    "source": "assumed",
                }
            )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
