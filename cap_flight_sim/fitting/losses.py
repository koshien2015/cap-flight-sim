"""損失: loss = Σ confidence_i · ||position_sim(t_i) - position_observed_i||²"""

from __future__ import annotations

import numpy as np

from .trajectory_loader import ObservedTrajectory

# シミュレーションが観測時刻より前に終わった点に与える残差 [m]（黙って捨てずに大きく罰する）
MISSING_SAMPLE_PENALTY_M = 10.0


def weighted_residuals(observed: ObservedTrajectory, simulated_positions: np.ndarray) -> np.ndarray:
    """least_squares 用の残差ベクトル sqrt(c_i)·(p_sim - p_obs)（3N 要素）。"""
    n_obs = len(observed.time_sec)
    n_sim = len(simulated_positions)
    diff = np.full((n_obs, 3), MISSING_SAMPLE_PENALTY_M)
    if n_sim:
        diff[:n_sim] = simulated_positions[:n_obs] - observed.position_m[:n_sim]
    return (np.sqrt(observed.confidence)[:, None] * diff).ravel()


def weighted_loss(observed: ObservedTrajectory, simulated_positions: np.ndarray) -> float:
    r = weighted_residuals(observed, simulated_positions)
    return float(np.dot(r, r))


def rmse_m(observed: ObservedTrajectory, simulated_positions: np.ndarray) -> float:
    n = min(len(observed.time_sec), len(simulated_positions))
    if n == 0:
        return float("inf")
    err = np.linalg.norm(simulated_positions[:n] - observed.position_m[:n], axis=1)
    return float(np.sqrt(np.mean(err**2)))
