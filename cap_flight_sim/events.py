"""終了・通過イベント。g(t, y) の符号変化で検出する。

NaN は g で扱わず（根探索が壊れるため）、状態微分で NonFiniteStateError を送出して異常終了として記録する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .models import FieldGeometry, TerminationSettings

GROUND = "ground"
CATCHER_PLANE = "catcher_plane"
MAX_TIME = "max_time"
OVERSPEED = "abnormal_speed"
OVERSPIN = "abnormal_angular_speed"
NON_FINITE = "abnormal_non_finite"
SOLVER_FAILED = "solver_failed"


@dataclass(frozen=True)
class EventSpec:
    name: str
    function: Callable[[float, np.ndarray], float]
    direction: int  # +1: 負→正 で発火, -1: 正→負 で発火
    terminal: bool


def build_events(field: FieldGeometry, termination: TerminationSettings) -> list[EventSpec]:
    return [
        EventSpec(GROUND, lambda t, y: y[2] - field.ground_z_m, -1, termination.stop_at_ground),
        EventSpec(CATCHER_PLANE, lambda t, y: y[0] - field.pitch_distance_m, +1, termination.stop_at_catcher_plane),
        EventSpec(OVERSPEED, lambda t, y: termination.max_speed_m_s - np.linalg.norm(y[3:6]), -1, True),
        EventSpec(OVERSPIN, lambda t, y: termination.max_angular_speed_rad_s - np.linalg.norm(y[10:13]), -1, True),
    ]


def _crossed(g0: float, g1: float, direction: int) -> bool:
    if direction > 0:
        return g0 < 0.0 <= g1
    return g0 > 0.0 >= g1


def locate_event(
    spec: EventSpec, t0: float, t1: float, interp: Callable[[float], np.ndarray]
) -> float | None:
    """区間 [t0, t1] 内の発火時刻（なければ None）。"""
    g0 = spec.function(t0, interp(t0))
    g1 = spec.function(t1, interp(t1))
    if not _crossed(g0, g1, spec.direction):
        return None
    if g1 == 0.0:
        return t1
    return float(brentq(lambda t: spec.function(t, interp(t)), t0, t1, xtol=1e-12, rtol=1e-10))
