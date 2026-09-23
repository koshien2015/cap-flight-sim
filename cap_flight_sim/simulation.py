"""1球分のシミュレーション実行。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import SIMULATION_VERSION
from .aerodynamics.base import AerodynamicModel
from .aerodynamics.factory import build_aerodynamic_model
from .coordinates import quat_normalize, quat_to_matrix
from .events import CATCHER_PLANE, MAX_TIME, NON_FINITE, SOLVER_FAILED, build_events, locate_event
from .integrators import Segment, SolverFailedError, adaptive_segments, rk4_segments
from .models import AerodynamicResult, RigidBodyState, SimulationConfig
from .rigid_body import NonFiniteStateError, make_state_derivative


@dataclass
class Frame:
    state: RigidBodyState
    omega_world_rad_s: np.ndarray
    normal_world: np.ndarray
    aero: AerodynamicResult


@dataclass
class SimulationResult:
    config: SimulationConfig
    model_description: dict[str, Any]
    frames: list[Frame]
    events: list[dict[str, Any]]
    termination_reason: str
    warnings: list[str] = field(default_factory=list)
    simulation_version: str = SIMULATION_VERSION

    @property
    def catcher_plane_event(self) -> dict[str, Any] | None:
        return next((e for e in self.events if e["type"] == CATCHER_PLANE), None)


def make_frame(t: float, y: np.ndarray, model: AerodynamicModel, config: SimulationConfig) -> Frame:
    q = quat_normalize(y[6:10])
    state = RigidBodyState(t, np.array(y[0:3]), np.array(y[3:6]), q, np.array(y[10:13]))
    rotation = quat_to_matrix(q)
    return Frame(
        state=state,
        omega_world_rad_s=rotation @ state.angular_velocity_body_rad_s,
        normal_world=rotation[:, 2],
        aero=model.evaluate(state, config.cap, config.environment),
    )


def run_simulation(
    config: SimulationConfig,
    model: AerodynamicModel | None = None,
    sample_times: np.ndarray | None = None,
) -> SimulationResult:
    """1球を計算する。sample_times を渡すと出力フレームをその時刻（計算範囲内）で作る。"""
    model = model or build_aerodynamic_model(config.aerodynamics, config.cap)
    settings = config.settings

    def aero_fn(state: RigidBodyState) -> AerodynamicResult:
        return model.evaluate(state, config.cap, config.environment)

    fun = make_state_derivative(config.cap, config.environment, aero_fn, settings.quaternion_drift_gain)
    y0 = config.initial_state.to_vector()
    t_end = settings.duration_sec
    if settings.solver == "rk4":
        segments_iter = rk4_segments(fun, 0.0, y0, t_end, settings.fixed_step_sec)
    else:
        segments_iter = adaptive_segments(
            fun, 0.0, y0, t_end, settings.method, settings.rtol, settings.atol, settings.max_step_sec
        )

    event_specs = build_events(config.field_geometry, settings.termination)
    segments: list[Segment] = []
    event_hits: list[tuple[str, float, Segment]] = []
    reason = MAX_TIME
    warnings = list(config.warnings)

    try:
        for seg in segments_iter:
            hits = sorted(
                ((t_hit, spec)
                for spec in event_specs
                if (t_hit := locate_event(spec, seg.t0, seg.t1, seg.interp)) is not None
            ), key=lambda hit: hit[0])
            terminal_time = None
            for t_hit, spec in hits:
                if terminal_time is not None and t_hit > terminal_time:
                    break
                event_hits.append((spec.name, t_hit, seg))
                if spec.terminal:
                    terminal_time = t_hit
                    reason = spec.name
            if terminal_time is not None:
                segments.append(Segment(seg.t0, terminal_time, seg.interp))
                break
            segments.append(seg)
    except NonFiniteStateError as exc:
        reason = NON_FINITE
        warnings.append(f"異常終了: {exc}")
    except SolverFailedError as exc:
        reason = SOLVER_FAILED
        warnings.append(f"積分器失敗: {exc}")

    t_final = segments[-1].t1 if segments else 0.0
    if sample_times is None:
        times = _output_times(t_final, settings.output_hz)
    else:
        times = np.asarray(sample_times, dtype=float)
        times = times[(times >= 0.0) & (times <= t_final)]
    frames = _sample_frames(segments, y0, times, model, config)

    events: list[dict[str, Any]] = [{"type": "release", "timeSec": 0.0}]
    for name, t_hit, seg in event_hits:
        frame = make_frame(t_hit, seg.interp(t_hit), model, config)
        events.append({"type": name, "timeSec": t_hit, "frame": frame})
    if reason in (MAX_TIME, NON_FINITE, SOLVER_FAILED):
        events.append({"type": reason, "timeSec": t_final})

    if reason != CATCHER_PLANE and not any(n == CATCHER_PLANE for n, _, _ in event_hits):
        warnings.append(f"捕手面(X={config.field_geometry.pitch_distance_m}m)に到達せず終了: {reason}")
    out_of_table = sum(bool(f.aero.diagnostics.get("out_of_table_range")) for f in frames)
    if out_of_table:
        warnings.append(f"係数表の範囲外参照 {out_of_table}/{len(frames)} フレーム（policy={getattr(model, 'out_of_range', '?')}）")
    if any(bool(f.aero.diagnostics.get("lift_degenerate_face_on")) for f in frames):
        warnings.append("面が相対風に正対（揚力方向が退化）したフレームあり: 揚力 0 として処理")

    return SimulationResult(
        config=config,
        model_description=model.describe(),
        frames=frames,
        events=events,
        termination_reason=reason,
        warnings=warnings,
    )


def _output_times(t_final: float, output_hz: float) -> np.ndarray:
    times = np.arange(0.0, t_final, 1.0 / output_hz)
    if times.size == 0 or t_final - times[-1] > 1e-12:
        times = np.append(times, t_final)
    return times


def _sample_frames(segments: list[Segment], y0: np.ndarray, times: np.ndarray, model, config) -> list[Frame]:
    if not segments:
        return [make_frame(0.0, y0, model, config)]
    ends = np.array([s.t1 for s in segments])
    frames = []
    for t in times:
        idx = min(int(np.searchsorted(ends, t - 1e-15)), len(segments) - 1)
        y = y0 if t == 0.0 else segments[idx].interp(float(t))
        frames.append(make_frame(float(t), y, model, config))
    return frames
