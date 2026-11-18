"""後流の非定常性による時間変動の横力・モーメント（Issue #7 の実験的モデル。既定 OFF）。

流れは解いていない。位相がランダムな数本の正弦波の和で、相対風に垂直な力とモーメントを与えるだけの近似で、
係数・周波数帯・回転への依存はすべて仮定値（実測と照合していない）。

    横力       F_w = q A_proj(α) C_w g(S) [s1(Φ) e1 + s2(Φ) e2]
    モーメント M_w = q A_proj(α) D C_mw g(S) [s3(Φ) e1 + s4(Φ) e2]
    位相       dΦ/dt = 2π |u| / D     （直径 1 つ分進むごとに 2π）
    信号       s_j(Φ) = Σ_i w_i sin(k_i Φ + φ_ji) / sqrt(Σ_i w_i² / 2)   （RMS = 1）
    回転       g(S) = 1 / (1 + (S / S0)²),  S = R |ω| / |u|

- e1, e2 は相対風に垂直な world 固定の基底（水平飛行では e1 = 左、e2 = 上）。面法線は使わない
  （正対で退化し、回転とともに向きが回るため）。
- 周波数は無次元 k = f D / |u| で与えるので、速度が落ちると揺れも遅くなる。速度は飛行中に変わるため、
  時刻ではなく積分した位相 Φ を状態として持つ。
- 乱数は位相のずれ φ を決めるときに一度だけ引く。力の評価中には引かない
  （RK4 や適応刻みは 1 ステップに何度も評価するので、評価のたびに引くと結果が決定的でなくなる）。
- 横力は重心に作用させ、モーメントには寄与させない。姿勢を回す作用は M_w として別に持つ。
- 回転数のしきい値や球種の分岐は持たない。g(S) は連続で、S0 = None なら回転に依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from ..coordinates import safe_unit
from ..models import CapParameters

UINT32 = 0xFFFFFFFF
SIGNAL_COUNT = 4  # 横力 2 軸 + モーメント 2 軸
WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_LEFT = np.array([0.0, 1.0, 0.0])
BASIS_EPS = 1e-9


@dataclass(frozen=True)
class WakeParams:
    force_coefficient: float = 0.0  # C_w（RMS、投影面積基準）。0 = 横力なし
    moment_coefficient: float = 0.0  # C_mw（RMS、投影面積 × 外径基準）。0 = モーメントなし
    seed: int = 1
    # k = f D / |u|。揺れ 1 回の飛行距離は D / k なので、D = 30 mm で 5.0 / 2.7 / 1.5 m（40 km/h なら約 2.2 / 4.1 / 7.4 Hz）。
    # 9.22 m の投球で数回揺れて見える帯域に置いた仮定値で、渦放出の周波数（k ~ 0.1 の桁）とは別物
    reduced_frequencies: tuple[float, ...] = (0.006, 0.011, 0.02)
    weights: tuple[float, ...] = (1.0, 0.8, 0.5)
    spin_parameter_scale: float | None = 0.05  # S0。None = 回転に依存しない

    def __post_init__(self) -> None:
        if not (_is_number(self.force_coefficient) and _is_number(self.moment_coefficient)) \
                or self.force_coefficient < 0.0 or self.moment_coefficient < 0.0:
            raise ValueError("wake.force_coefficient / moment_coefficient must be finite numbers >= 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= UINT32:
            raise ValueError("wake.seed must be an integer in [0, 2^32)")
        if not self.reduced_frequencies or len(self.reduced_frequencies) != len(self.weights):
            raise ValueError("wake.reduced_frequencies and wake.weights must be non-empty and the same length")
        if any(not _is_positive(v) for v in (*self.reduced_frequencies, *self.weights)):
            raise ValueError("wake.reduced_frequencies and wake.weights must be positive finite numbers")
        if self.spin_parameter_scale is not None and not _is_positive(self.spin_parameter_scale):
            raise ValueError("wake.spin_parameter_scale must be a positive finite number or null")

    @property
    def enabled(self) -> bool:
        return self.force_coefficient > 0.0 or self.moment_coefficient > 0.0


def _is_number(value: Any) -> bool:
    """有限の数値か（真偽値・文字列・NaN・Inf は数値として扱わない。ブラウザ版の検証と同じ）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _is_positive(value: Any) -> bool:
    return _is_number(value) and value > 0.0


def build_wake_params(section: dict[str, Any] | None) -> WakeParams:
    if section is None:
        return WakeParams()
    if not isinstance(section, dict):
        raise ValueError("aerodynamics.wake must be a mapping")
    unknown = set(section) - set(WakeParams.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown key aerodynamics.wake.{sorted(unknown)[0]}")
    values = dict(section)
    for key in ("reduced_frequencies", "weights"):
        if key in values:
            if not isinstance(values[key], (list, tuple)):
                raise ValueError(f"wake.{key} must be a list")
            values[key] = tuple(values[key])
    return WakeParams(**values)


def mulberry32(seed: int) -> Callable[[], float]:
    """シード付き擬似乱数 [0, 1)。ブラウザ版（viewer/src/physics/wake.ts）と同じ列を返す。"""
    state = seed & UINT32

    def imul(a: int, b: int) -> int:
        return (a * b) & UINT32

    def draw() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & UINT32
        t = imul(state ^ (state >> 15), state | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & UINT32
        return (t ^ (t >> 14)) / 4294967296

    return draw


@lru_cache(maxsize=64)
def phase_offsets(seed: int, band_count: int) -> tuple[tuple[float, ...], ...]:
    """信号ごと（横力 e1・e2、モーメント e1・e2）× 正弦波ごとの位相のずれ [rad]。"""
    rng = mulberry32(seed)
    return tuple(tuple(rng() * 2.0 * np.pi for _ in range(band_count)) for _ in range(SIGNAL_COUNT))


def band_signal(offsets: tuple[float, ...], reduced_frequencies: tuple[float, ...], weights: tuple[float, ...], phase: float) -> float:
    total = 0.0
    power = 0.0
    for offset, k, w in zip(offsets, reduced_frequencies, weights):
        total += w * np.sin(k * phase + offset)
        power += w * w
    return float(total / np.sqrt(0.5 * power))


def spin_attenuation(spin_parameter: float, scale: float | None) -> float:
    if scale is None:
        return 1.0
    return float(1.0 / (1.0 + (spin_parameter / scale) ** 2))


def wake_basis(u_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """相対風に垂直な world 固定の基底 (e1, e2)。水平飛行では e1 = 左（+Y）、e2 = 上（+Z）。"""
    e1 = safe_unit(np.cross(WORLD_UP, u_hat), BASIS_EPS)
    if not e1.any():  # 真上・真下へ飛ぶとき
        e1 = WORLD_LEFT
    return e1, np.cross(u_hat, e1)


@dataclass(frozen=True)
class WakeLoads:
    force_world_n: np.ndarray
    moment_world_nm: np.ndarray
    phase_rate_rad_s: float
    spin_attenuation: float


def evaluate_wake(
    params: WakeParams,
    cap: CapParameters,
    u_hat: np.ndarray,
    speed_m_s: float,
    dynamic_pressure_pa: float,
    projected_area_m2: float,
    angular_speed_rad_s: float,
    phase_rad: float,
) -> WakeLoads:
    offsets = phase_offsets(params.seed, len(params.reduced_frequencies))
    signals = [band_signal(o, params.reduced_frequencies, params.weights, phase_rad) for o in offsets]
    gain = spin_attenuation(cap.radius_m * angular_speed_rad_s / speed_m_s, params.spin_parameter_scale)
    e1, e2 = wake_basis(u_hat)
    scale = dynamic_pressure_pa * projected_area_m2 * gain
    force = scale * params.force_coefficient * (signals[0] * e1 + signals[1] * e2)
    moment = scale * cap.outer_diameter_m * params.moment_coefficient * (signals[2] * e1 + signals[3] * e2)
    return WakeLoads(force, moment, 2.0 * np.pi * speed_m_s / cap.outer_diameter_m, gain)


def describe_wake(params: WakeParams) -> dict[str, Any]:
    return {
        "status": "experimental_unvalidated",
        "enabled": params.enabled,
        "equations": {
            "force": "F_w = q A_proj(alpha) C_w g(S) [s1(Phi) e1 + s2(Phi) e2], e1/e2 perpendicular to u (world-fixed)",
            "moment": "M_w = q A_proj(alpha) D C_mw g(S) [s3(Phi) e1 + s4(Phi) e2]",
            "phase": "dPhi/dt = 2 pi |u| / D",
            "signal": "s_j = sum_i w_i sin(k_i Phi + phi_ji) / sqrt(sum_i w_i^2 / 2)",
            "spin": "g(S) = 1 / (1 + (S / S0)^2), S = R |omega| / |u|",
        },
        "notes": [
            "流れは解いていない。係数・周波数帯・回転への依存はすべて仮定値で、実測と未照合",
            "横力は重心に作用させ、モーメントには寄与させない（姿勢を回す作用は M_w として別項）",
        ],
        "force_coefficient": params.force_coefficient,
        "moment_coefficient": params.moment_coefficient,
        "seed": params.seed,
        "reduced_frequencies": list(params.reduced_frequencies),
        "weights": list(params.weights),
        "spin_parameter_scale": params.spin_parameter_scale,
    }
