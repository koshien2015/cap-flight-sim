"""数値積分。どちらの積分器も同じ状態微分関数を使い、区間ごとの補間関数を返す。

- solve_ivp 系: scipy.integrate の OdeSolver（solve_ivp が内部で使うクラス）を直接ステップ実行する。
  NaN 発生時に直前までの結果を保持して異常終了を記録するため、solve_ivp 本体ではなくクラスを使う。
- rk4: 固定刻み古典 RK4（比較・デバッグ用）。区間補間は 3 次エルミート。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, Radau

Derivative = Callable[[float, np.ndarray], np.ndarray]

SOLVER_CLASSES = {"RK45": RK45, "RK23": RK23, "DOP853": DOP853, "Radau": Radau, "BDF": BDF, "LSODA": LSODA}


class SolverFailedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    interp: Callable[[float], np.ndarray]


def adaptive_segments(
    fun: Derivative, t0: float, y0: np.ndarray, t_end: float, method: str, rtol: float, atol: float, max_step: float
) -> Iterator[Segment]:
    if method not in SOLVER_CLASSES:
        raise ValueError(f"unknown method {method}; choose from {list(SOLVER_CLASSES)}")
    solver = SOLVER_CLASSES[method](fun, t0, y0, t_end, max_step=max_step, rtol=rtol, atol=atol)
    while solver.status == "running":
        t_prev = solver.t
        message = solver.step()
        if solver.status == "failed":
            raise SolverFailedError(message or "solver failed")
        dense = solver.dense_output()
        yield Segment(t_prev, solver.t, dense)


def rk4_segments(fun: Derivative, t0: float, y0: np.ndarray, t_end: float, dt: float) -> Iterator[Segment]:
    t = t0
    y = np.asarray(y0, dtype=float)
    f = fun(t, y)
    while t < t_end - 1e-15:
        h = min(dt, t_end - t)
        k1 = f
        k2 = fun(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = fun(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = fun(t + h, y + h * k3)
        y1 = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        f1 = fun(t + h, y1)
        yield Segment(t, t + h, _hermite(t, t + h, y, y1, f, f1))
        t, y, f = t + h, y1, f1


def _hermite(t0: float, t1: float, y0: np.ndarray, y1: np.ndarray, f0: np.ndarray, f1: np.ndarray):
    h = t1 - t0

    def interp(t: float) -> np.ndarray:
        s = (t - t0) / h
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1

    return interp
