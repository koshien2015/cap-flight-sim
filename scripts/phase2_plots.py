"""phase2_sensitivity.py の結果から図を描く（計算はしない）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cap_flight_sim.outputs import plots as _style  # noqa: E402,F401  日本語フォント設定を流用

NOTE = "（係数はすべて仮定値。前縁30°は投手の感覚値）"


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    path = out / name
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_cp_sweep(table: pd.DataFrame, series: dict[str, Any], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["knuckle_nominal"] + table.loc[table.group == "cp_offset", "name"].tolist()
    for name in names:
        s = series[name]
        label = "圧力中心＝重心" if name == "knuckle_nominal" else f"圧力中心 {name.split('=')[1]} mm（+ = 天面側）"
        ax.plot(s["t"], s["elev"], label=label)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("面法線の仰角 [deg]")
    ax.set_title("ナックル: 圧力中心の位置と縦の姿勢変化\n" + NOTE, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, out, "fig1_cp_sweep_elevation.png")


def fig_perturbation(series: dict[str, Any], out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, base in zip(axes, ("no_moment", "cp_aft", "cp_fwd")):
        ref = series[f"{base}/eps=0"]
        for eps in (1e-3, 1e-1, 1.0, 3.0):
            s = series[f"{base}/yaw_rate={eps}"]
            n = min(len(s["t"]), len(ref["t"]))
            dy = (np.array(s["y"][:n]) - np.array(ref["y"][:n])) / eps * 1e3
            ax.plot(s["t"][:n], dy, label=f"ε = {eps} rad/s")
        ax.set_title({"no_moment": "モーメントなし", "cp_aft": "圧力中心が開口側 2.5mm",
                      "cp_fwd": "圧力中心が天面側 2.5mm"}[base], fontsize=10)
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("横変位の差 / ε [mm per rad/s]")
    axes[0].legend(fontsize=7)
    fig.suptitle("初期のヨー角速度 ε に対する横変位の応答（線が重なれば ε に比例＝線形応答）\n" + NOTE, fontsize=10)
    return _save(fig, out, "fig2_perturbation_linearity.png")


def fig_spin(table: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = [
        ("rpm/cp_aft/all", "knuckle_cp_aft", "ナックル・圧力中心 開口側"),
        ("rpm/cp_fwd/all", None, "ナックル・圧力中心 天面側"),
        ("straight/cp_aft", None, "ストレート・圧力中心 開口側"),
    ]
    for group, zero_name, label in rows:
        sub = table[table.group == group].copy()
        if zero_name:
            sub = pd.concat([table[table.name == zero_name], sub])
        if group == "rpm/cp_fwd/all":
            sub = pd.concat([table[table.name == "cp_dz=2.5"], sub])
        sub = sub.sort_values("rpm")
        ax.plot(sub["rpm"].clip(lower=10), sub["max_normal_change_deg"], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("回転数 [rpm]（0 rpm は 10 の位置に表示）")
    ax.set_ylabel("面法線の最大変化 [deg]")
    ax.set_title("回転数と姿勢の保たれ方\n" + NOTE, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out, "fig3_spin_stabilization.png")


def fig_rpm_attribution(table: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, base in zip(axes, ("cp_aft", "cp_fwd")):
        for label, jp in (("all", "全部"), ("no_magnus", "マグヌスなし"), ("no_magnus_lift", "マグヌス・揚力なし")):
            sub = table[table.group == f"rpm/{base}/{label}"].sort_values("rpm")
            ax.plot(sub["rpm"], sub["y_mm@x3.0m"], marker="o", label=jp)
        ax.set_xscale("log")
        ax.set_title("圧力中心 " + ("開口側" if base == "cp_aft" else "天面側") + " 2.5mm", fontsize=10)
        ax.set_xlabel("回転数 [rpm]（面法線まわり）")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("前方 3m での横変位 [mm]（+ = 投手から見て左）")
    axes[0].legend(fontsize=8)
    fig.suptitle("ナックル姿勢で回転をかけたときの横変位の内訳\n" + NOTE, fontsize=10)
    return _save(fig, out, "fig4_rpm_lateral_attribution.png")


def write_figures(table: pd.DataFrame, series: dict[str, Any], out: Path) -> list[Path]:
    return [fig_cp_sweep(table, series, out), fig_perturbation(series, out), fig_spin(table, out),
            fig_rpm_attribution(table, out)]
