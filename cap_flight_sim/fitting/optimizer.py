"""観測軌道への係数フィッティング（scipy.optimize.least_squares, 境界付き）。

フィッティング設定（YAML）:
    base_config: ../examples/flick_flat.yaml
    parameters:
      - path: aerodynamics.drag.cd_projected
        initial: 0.8
        bounds: [0.1, 2.0]
      - path: initial_state.speed_kmh   # 初期条件の微修正も同じ書式
        initial: 50
        bounds: [45, 55]
    max_nfev: 200

全学習軌道で共通のパラメータを推定する（各軌道の初期条件は base_config で共通）。
推定値・標準誤差・相関・境界到達・条件数を出力し、識別不能性の兆候を警告する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares

from ..config import apply_overrides, config_from_dict, get_dotted, load_raw_config, load_yaml
from ..outputs.serializers import write_json
from ..simulation import run_simulation
from .losses import rmse_m, weighted_loss, weighted_residuals
from .trajectory_loader import ObservedTrajectory, load_trajectory

BOUND_TOLERANCE_RATIO = 1e-3
CORRELATION_WARNING = 0.95
CONDITION_WARNING = 1e8


@dataclass(frozen=True)
class FitParameter:
    path: str
    initial: float
    lower: float
    upper: float


def parse_parameters(spec: list[dict[str, Any]], base_raw: dict[str, Any]) -> list[FitParameter]:
    params = []
    for item in spec:
        path = item["path"]
        initial = float(item["initial"]) if "initial" in item else float(get_dotted(base_raw, path))
        lower, upper = (float(v) for v in item["bounds"])
        if not lower < upper or not lower <= initial <= upper:
            raise ValueError(f"{path}: need lower < upper and initial within bounds")
        params.append(FitParameter(path, initial, lower, upper))
    if not params:
        raise ValueError("fitting needs at least one parameter")
    return params


def _fit_raw(base_raw: dict[str, Any], observations: list[ObservedTrajectory]) -> dict[str, Any]:
    """フィット中は地面・捕手面で止めず、最終観測時刻まで計算する。"""
    t_max = max(float(o.time_sec[-1]) for o in observations)
    return apply_overrides(
        base_raw,
        {
            "simulation.duration_sec": t_max + 1e-3,
            "simulation.termination.stop_at_ground": False,
            "simulation.termination.stop_at_catcher_plane": False,
        },
    )


def simulate_positions(raw: dict[str, Any], params: list[FitParameter], values: np.ndarray, obs: ObservedTrajectory) -> np.ndarray:
    overridden = apply_overrides(raw, {p.path: float(v) for p, v in zip(params, values)})
    result = run_simulation(config_from_dict(overridden), sample_times=obs.time_sec)
    return np.array([f.state.position_world_m for f in result.frames]).reshape(-1, 3)


def fit(
    base_raw: dict[str, Any],
    params: list[FitParameter],
    train: list[ObservedTrajectory],
    test: list[ObservedTrajectory],
    max_nfev: int = 200,
) -> dict[str, Any]:
    raw = _fit_raw(base_raw, train + test)
    x0 = np.array([p.initial for p in params])
    lower = np.array([p.lower for p in params])
    upper = np.array([p.upper for p in params])

    def residuals(x: np.ndarray) -> np.ndarray:
        return np.concatenate([weighted_residuals(o, simulate_positions(raw, params, x, o)) for o in train])

    solution = least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac", diff_step=1e-4, max_nfev=max_nfev)
    stats = _uncertainty(solution, len(params))

    warnings: list[str] = []
    fitted = []
    for i, p in enumerate(params):
        span = p.upper - p.lower
        at_lower = solution.x[i] - p.lower < BOUND_TOLERANCE_RATIO * span
        at_upper = p.upper - solution.x[i] < BOUND_TOLERANCE_RATIO * span
        if at_lower or at_upper:
            warnings.append(f"{p.path} が境界に到達（{'下限' if at_lower else '上限'}）: 真値が範囲外か、他パラメータと代替関係の可能性")
        fitted.append(
            {
                "path": p.path,
                "value": float(solution.x[i]),
                "initial": p.initial,
                "bounds": [p.lower, p.upper],
                "standardError": stats["std"][i],
                "atBound": bool(at_lower or at_upper),
                "source": "fitted",
            }
        )
    corr = stats["correlation"]
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            if corr is not None and abs(corr[i][j]) > CORRELATION_WARNING:
                warnings.append(f"{params[i].path} と {params[j].path} の相関 {corr[i][j]:+.3f}: 識別困難（同じ軌道を別の組合せで説明可能）")
    if stats["conditionNumber"] > CONDITION_WARNING:
        warnings.append(f"ヤコビアン条件数 {stats['conditionNumber']:.2e}: パラメータが軌道から一意に決まらない恐れ")
    if len(train) == 1:
        warnings.append("学習軌道が 1 球のみ: 推定値を一般則として扱わないこと")
    if not test:
        warnings.append("評価用軌道なし: 汎化性能は未評価")
    if any(o.is_synthetic for o in train + test):
        warnings.append("合成（仮想）観測データを含む: 実測による検証ではない")

    def evaluate(obs_list: list[ObservedTrajectory], x: np.ndarray) -> list[dict[str, Any]]:
        rows = []
        for o in obs_list:
            sim = simulate_positions(raw, params, x, o)
            rows.append({"name": o.name, "rmseM": rmse_m(o, sim), "weightedLoss": weighted_loss(o, sim),
                         "coverage": len(sim) / len(o.time_sec), "synthetic": o.is_synthetic})
        return rows

    return {
        "summary": {
            "success": bool(solution.success),
            "message": solution.message,
            "nfev": int(solution.nfev),
            "parameters": {f["path"]: {"value": f["value"], "standardError": f["standardError"], "atBound": f["atBound"]} for f in fitted},
            "train": evaluate(train, solution.x),
            "test": evaluate(test, solution.x),
            "warnings": warnings,
        },
        "fittedParameters": fitted,
        "correlation": corr,
        "conditionNumber": stats["conditionNumber"],
        "initialTrain": evaluate(train, x0),
        "loss": "sum_i confidence_i * ||p_sim(t_i) - p_obs_i||^2",
        "_x": solution.x,
    }


def _uncertainty(solution, n_params: int) -> dict[str, Any]:
    jac = solution.jac
    m = len(solution.fun)
    dof = max(m - n_params, 1)
    s2 = 2.0 * solution.cost / dof
    jtj = jac.T @ jac
    cond = float(np.linalg.cond(jtj)) if np.all(np.isfinite(jtj)) else float("inf")
    try:
        cov = np.linalg.inv(jtj) * s2
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        denom = np.outer(std, std)
        corr = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), 0.0)
        return {"std": [float(v) for v in std], "correlation": corr.tolist(), "conditionNumber": cond}
    except np.linalg.LinAlgError:
        return {"std": [float("inf")] * n_params, "correlation": None, "conditionNumber": float("inf")}


def run_fit_from_files(
    config_path: str | Path, train_paths: list[str], output_dir: str | Path, test_paths: list[str]
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    fit_cfg = load_yaml(cfg_path)
    base_raw = load_raw_config((cfg_path.parent / fit_cfg["base_config"]).resolve())
    params = parse_parameters(fit_cfg["parameters"], base_raw)
    train = [load_trajectory(p) for p in train_paths]
    test = [load_trajectory(p) for p in test_paths]
    report = fit(base_raw, params, train, test, int(fit_cfg.get("max_nfev", 200)))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = report.pop("_x")
    write_json(out / "fit_report.json", {"fitConfig": fit_cfg, **report})
    fitted_raw = apply_overrides(
        base_raw, {p.path: {"value": float(v), "source": "fitted", "note": f"fit to {','.join(o.name for o in train)}"}
                   for p, v in zip(params, x)}
    )
    with open(out / "fitted_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(fitted_raw, f, allow_unicode=True, sort_keys=False)
    _plot_fit(base_raw, params, x, train + test, out)
    return report


def _plot_fit(base_raw, params, x, observations, out: Path) -> None:
    from ..outputs import plots  # noqa: F401  フォント設定を適用
    import matplotlib.pyplot as plt

    raw = _fit_raw(base_raw, observations)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for o in observations:
        sim = simulate_positions(raw, params, x, o)
        for ax, idx, lab in ((axes[0], 1, "Y [m]（+=左）"), (axes[1], 2, "Z [m]")):
            ax.scatter(o.position_m[:, 0], o.position_m[:, idx], s=8, alpha=0.6, label=f"観測 {o.name}")
            ax.plot(sim[:, 0], sim[:, idx], label=f"フィット {o.name}")
            ax.set_xlabel("X [m]")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
    axes[0].set_title("上面 X-Y（フィット値は fitted、一般則ではない）", fontsize=10)
    axes[1].set_title("側面 X-Z", fontsize=10)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fit_trajectory.png", dpi=130)
    plt.close(fig)
