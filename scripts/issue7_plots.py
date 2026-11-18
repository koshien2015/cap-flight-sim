"""issue7_wake_study.py の結果から図を描く（計算はしない）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cap_flight_sim.outputs import plots as _style  # noqa: E402,F401  日本語フォント設定を流用

NOTE = "（係数はすべて仮定値。実測と未照合）"


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    path = out / name
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_flip(series: dict[str, Any], out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = {"face_up/no_moment": "面が上向き・モーメントなし", "face_up/cp_on": "面が上向き・復原モーメント ON",
              "face_forward/cp_on": "天面を打者へ正対・復原モーメント ON"}
    for name, label in labels.items():
        s = series[name]
        axes[0].plot(s["t"], np.array(s["pitch"]) - s["pitch"][0], label=label)
        axes[1].plot(s["t"], s["normal_to_wind"], label=label)
        axes[2].plot(s["t"], s["omega_inplane"], label=label)
    axes[0].set_ylabel("面法線の縦の回転量 [deg]（リリース時から）")
    axes[1].set_ylabel("面法線と相対風のなす角 [deg]（0 = 天面が風に正対）")
    axes[2].set_ylabel("面内軸まわりの角速度 [rad/s]")
    for ax in axes:
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("復原モーメント（圧力中心のずれ）を入れたときの姿勢変化\n" + NOTE, fontsize=10)
    return _save(fig, out, "fig1_flip_timeseries.png")


def fig_static_moment(checks: dict[str, Any], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    static = checks["static_pitch_moment"]
    x = np.array(static["offset_deg"])
    names = {"cp_on": "復原モーメント ON（空力中心 = 幾何中心）", "cm_alpha_-0.05": "ピッチ係数 cm_pitch_alpha = −0.05 /rad"}
    for name, label in names.items():
        line, = ax.plot(x, np.array(static[name]) * 1e6, label=label)
        for eq in checks["equilibria"][name]:
            ax.plot(eq["offset_deg"], 0.0, "o" if eq["stable"] else "x", color=line.get_color(), markersize=8)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(90.0, color="gray", ls=":", lw=0.8)
    ax.text(91, ax.get_ylim()[1] * 0.85, "面が上向き", fontsize=8, color="gray")
    ax.set_xlabel("面法線を進行方向から上へ傾けた角 [deg]（0 = 天面が前、90 = 面が上、180 = 開口が前）")
    ax.set_ylabel("ピッチモーメント [μN·m]（+ = 天面を前へ向ける）")
    ax.set_title("姿勢ごとの静的なピッチモーメント（40 km/h・無回転）。○ = 安定な平衡、× = 不安定な平衡\n" + NOTE, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out, "fig2_static_moment.png")


def fig_damping(table: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    damping = table[table.group == "damping"]
    for cp_z, rows in damping.groupby("cp_z_mm"):
        rows = rows.sort_values("tumble_damping")
        x = rows.tumble_damping.replace(0.0, 0.02)  # 対数軸に 0 を置けないので左端に寄せる
        ax.plot(x, rows.max_normal_change_deg, "o-", label=f"空力中心 {cp_z:+.1f} mm（重心から {cp_z - 2.5:+.1f} mm）")
    ax.axhline(90.0, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("tumble_damping（左端は 0）")
    ax.set_ylabel("面法線の最大変化 [deg]")
    ax.set_title("面が上向き・無回転: 減衰と腕の長さを分けて振ったときの面の振れ\n" + NOTE, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out, "fig3_damping_sweep.png")


def fig_wake_seeds(series: dict[str, Any], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    titles = {"face_up": "面が上向き（前縁 10° 上げ・40 km/h）", "face_forward": "天面を打者へ正対（前縁 30° 上げ・50 km/h）"}
    for ax, (base, title) in zip(axes, titles.items()):
        for name, s in series.items():
            if name.startswith(f"wake_force/{base}/"):
                ax.plot(s["x"], np.array(s["y"]) * 1e3, label=name.split("/")[-1])
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("前方距離 x [m]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("横変位 y [mm]（+ = 投手から見て左）")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("後流の非定常な横力だけを足したときの横変位（C_w = 0.15、無回転。破線 = 項なし）\n" + NOTE, fontsize=10)
    return _save(fig, out, "fig4_wake_force_seeds.png")


def fig_wake_terms(series: dict[str, Any], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = {"force_only": "横力だけ", "moment_only_td2": "モーメントだけ（減衰 2）", "both_td2": "両方（減衰 2）",
              "both_td0": "両方（減衰なし）", "both_cp_on_td2": "両方 + 復原モーメント ON（減衰 2）",
              "both_cm-0.05_td2": "両方 + ピッチ係数 −0.05（減衰 2）"}
    for variant, label in labels.items():
        s = series[f"wake_terms/{variant}/seed=3"]
        axes[0].plot(s["t"], np.array(s["y"]) * 1e3, label=label)
        axes[1].plot(s["t"], s["tilt"], label=label)
    axes[0].set_ylabel("横変位 y [mm]")
    axes[1].set_ylabel("面法線のリリース時からの傾き [deg]")
    for ax in axes:
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("横力 F_w・モーメント M_w の切り分け（面が上向き・無回転・シード 3、C_w = 0.15、C_mw = 0.001）\n" + NOTE, fontsize=10)
    return _save(fig, out, "fig5_wake_terms.png")


def write_figures(table: pd.DataFrame, series: dict[str, Any], checks: dict[str, Any], out: Path) -> list[Path]:
    return [fig_flip(series, out), fig_static_moment(checks, out), fig_damping(table, out),
            fig_wake_seeds(series, out), fig_wake_terms(series, out)]
