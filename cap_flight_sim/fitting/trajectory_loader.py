"""観測軌道 CSV の読み込みと、検証用の合成観測軌道の生成。

観測 CSV（ワールド座標・SI）:
    time_sec,x_m,y_m,z_m,confidence
    time_sec はリリースを 0 とする。座標は本パッケージの右手系（+Y=投手から見て左）。
    画像座標や 2D 動画からの 3D 復元はこのパッケージの範囲外（2D から完全な 3D 姿勢は得られない）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..simulation import run_simulation

REQUIRED_COLUMNS = ("time_sec", "x_m", "y_m", "z_m")


@dataclass(frozen=True)
class ObservedTrajectory:
    name: str
    time_sec: np.ndarray
    position_m: np.ndarray  # (N, 3)
    confidence: np.ndarray  # (N,)
    is_synthetic: bool


def load_trajectory(path: str | Path) -> ObservedTrajectory:
    p = Path(path)
    df = pd.read_csv(p, comment="#")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).sort_values("time_sec")
    if len(df) < 3:
        raise ValueError(f"{p}: need at least 3 observations")
    confidence = df["confidence"].to_numpy(float) if "confidence" in df.columns else np.ones(len(df))
    if np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError(f"{p}: confidence must be within [0, 1]")
    with open(p, encoding="utf-8") as f:
        synthetic = "synthetic" in f.readline().lower()
    return ObservedTrajectory(
        name=p.stem,
        time_sec=df["time_sec"].to_numpy(float),
        position_m=df[["x_m", "y_m", "z_m"]].to_numpy(float),
        confidence=confidence,
        is_synthetic=synthetic,
    )


def synthesize_observation(
    config_path: str | Path, output_path: str | Path, fps: float, noise_m: float, seed: int
) -> Path:
    """シミュレーション結果に等方ガウス雑音を加えた「仮想の観測」CSV。実測ではない。"""
    config = load_config(config_path)
    result = run_simulation(config)
    t_end = result.frames[-1].state.time_sec
    times = np.arange(0.0, t_end, 1.0 / fps)
    sampled = run_simulation(config, sample_times=times)
    rng = np.random.default_rng(seed)
    pos = np.array([f.state.position_world_m for f in sampled.frames])
    noisy = pos + rng.normal(scale=noise_m, size=pos.shape)
    confidence = np.clip(rng.normal(0.9, 0.05, size=len(pos)), 0.5, 1.0)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# synthetic observation from {Path(config_path).name}, noise_m={noise_m}, fps={fps}, seed={seed}\n")
        pd.DataFrame(
            {"time_sec": times[: len(pos)], "x_m": noisy[:, 0], "y_m": noisy[:, 1], "z_m": noisy[:, 2], "confidence": confidence}
        ).to_csv(f, index=False, float_format="%.6f")
    return out
