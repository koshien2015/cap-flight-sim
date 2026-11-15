"""グラフ出力（result.json の timeseries から描画。物理計算には依存しない）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib import font_manager  # noqa: E402

_JP_FONT_CANDIDATES = ("Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic", "Yu Gothic")
_available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in _JP_FONT_CANDIDATES if f in _available] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

Y_NOTE = "Y [m]（+ = 投手から見て左）"


def _series(doc: dict[str, Any], key: str) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in doc["timeseries"][key]], dtype=float)


def _label(doc: dict[str, Any]) -> str:
    return f"{doc['name']} [{doc['summary']['terminationReason']}]"


def _overlay(docs, out: Path, filename: str, title: str, xlabel: str, ylabel: str, draw, equal: bool = False, extra=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for doc in docs:
        draw(ax, doc)
    if extra:
        extra(ax)
    ax.set_title(title + "\n（仮定係数に基づく計算値）", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_runs(docs: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pitch = docs[0]["config"]["si"]["field"]["pitch_distance_m"]

    def catcher_line(ax):
        ax.axvline(pitch, color="gray", ls="--", lw=1, label=f"捕手面 X={pitch}m")

    paths = [
        _overlay(docs, out, "01_top_xy.png", "上から見た軌道 X-Y", "X [m]", Y_NOTE,
                 lambda ax, d: ax.plot(_series(d, "x"), _series(d, "y"), label=_label(d)), extra=catcher_line),
        _overlay(docs, out, "02_side_xz.png", "横から見た軌道 X-Z", "X [m]", "Z [m]",
                 lambda ax, d: ax.plot(_series(d, "x"), _series(d, "z"), label=_label(d)), extra=catcher_line),
        _plot_catcher_plane(docs, out),
        _overlay(docs, out, "04_speed.png", "速度", "t [s]", "|v| [m/s]",
                 lambda ax, d: ax.plot(_series(d, "time"), _series(d, "speed"), label=_label(d))),
        _overlay(docs, out, "05_rpm.png", "回転数", "t [s]", "rpm",
                 lambda ax, d: ax.plot(_series(d, "time"), _series(d, "rpm"), label=_label(d))),
        _plot_normals(docs, out),
        _plot_forces(docs, out),
        _plot_angles(docs, out),
    ]
    # Issue #1 Phase 1 の診断図。列が無い古い result.json は描かない
    paths += [p for p in (_plot_attitude(docs, out), _plot_moments(docs, out), _plot_reynolds(docs, out)) if p]
    return paths


def _with(docs, *keys: str) -> list[dict[str, Any]]:
    return [d for d in docs if all(k in d["timeseries"] for k in keys)]


def _save(fig, out: Path, filename: str) -> Path:
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_catcher_plane(docs, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for d in docs:
        cp = d["summary"].get("catcherPlane")
        if cp:
            ax.plot(cp["yM"], cp["zM"], "o", label=f"{d['name']}: Y={cp['yM']:.3f}, Z={cp['zM']:.3f}")
        else:
            ax.plot([], [], "x", label=f"{d['name']}: 未到達 ({d['summary']['terminationReason']})")
    ax.invert_xaxis()  # 捕手側から投手を見る視点では +Y(投手の左) が画面右
    ax.axhline(0, color="saddlebrown", lw=1)
    ax.set_title("捕手面 Y-Z（捕手側から投手方向を見る）\n（仮定係数に基づく計算値）", fontsize=11)
    ax.set_xlabel(Y_NOTE + "  ※X軸反転")
    ax.set_ylabel("Z [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "03_catcher_plane_yz.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_normals(docs, out: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for d in docs:
        t = _series(d, "time")
        for ax, key in zip(axes, ("normal_x", "normal_y", "normal_z")):
            ax.plot(t, _series(d, key), label=d["name"])
            ax.set_ylabel(key)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(alpha=0.3)
    axes[0].set_title("キャップ面法線（world 成分）\n（仮定係数に基づく計算値）", fontsize=11)
    axes[-1].set_xlabel("t [s]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = out / "06_face_normal.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_forces(docs, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"drag": ("-", "抗力"), "lift": ("--", "揚力"), "magnus": (":", "マグヌス")}
    for d in docs:
        t = _series(d, "time")
        color = None
        for key, (ls, jp) in styles.items():
            mag = np.linalg.norm(np.column_stack([_series(d, f"{key}_{c}") for c in "xyz"]), axis=1)
            (line,) = ax.plot(t, mag, ls=ls, color=color, label=f"{d['name']} {jp}")
            color = line.get_color()
    weight = docs[0]["summary"]["weightN"]
    ax.axhline(weight, color="black", lw=0.8, alpha=0.5, label=f"重量 {weight:.4f} N")
    ax.set_title("空気力の大きさ\n（仮定係数に基づく計算値）", fontsize=11)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|F| [N]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "07_forces.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_angles(docs, out: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for d in docs:
        t = _series(d, "time")
        axes[0].plot(t, _series(d, "angle_of_attack"), label=d["name"])
        axes[1].plot(t, _series(d, "sideslip"), label=d["name"])
    axes[0].set_ylabel("迎角 α [deg]")
    axes[1].set_ylabel("横滑り角 β [deg]")
    axes[1].set_xlabel("t [s]")
    axes[0].set_title("迎角・横滑り角（β は面内方位。軸対称キャップでは自転とともに回る）\n（仮定係数に基づく計算値）", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = out / "08_angles.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_attitude(docs, out: Path) -> Path | None:
    docs = _with(docs, "omega_normal", "omega_inplane", "spin_axis_tilt_deg")
    if not docs:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for d in docs:
        t = _series(d, "time")
        (line,) = axes[0].plot(t, _series(d, "omega_normal"), label=f"{d['name']} 面法線まわり")
        axes[0].plot(t, _series(d, "omega_inplane"), ls="--", color=line.get_color(), label=f"{d['name']} 面内（タンブリング）")
        axes[1].plot(t, _series(d, "spin_axis_tilt_deg"), label=d["name"])
        axes[2].plot(t, _series(d, "normal_to_wind_deg"), label=d["name"])
    axes[0].set_ylabel("ω [rad/s]")
    axes[1].set_ylabel("回転軸と面法線の角 [deg]")
    axes[2].set_ylabel("面法線と相対風の角 [deg]")
    axes[2].set_xlabel("t [s]")
    axes[0].set_title("回転と姿勢（回転軸の傾きは無回転では未定義）\n（仮定係数に基づく計算値）", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    return _save(fig, out, "09_attitude.png")


def _moment_components(doc) -> list[str]:
    names = {k[len("moment_"):-2] for k in doc["timeseries"] if k.startswith("moment_") and k[-2:] in ("_x", "_y", "_z")}
    return sorted(n for n in names if n and all(f"moment_{n}_{a}" in doc["timeseries"] for a in "xyz"))


def _plot_moments(docs, out: Path) -> Path | None:
    docs = [d for d in docs if _moment_components(d)]
    if not docs:
        return None
    labels = {"center_of_pressure": "圧力中心のずれ", "pitch": "ピッチ係数", "damping": "回転減衰", "table": "係数表"}
    styles = ("-", "--", ":", "-.")
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in docs:
        t = _series(d, "time")
        color = None
        for name, ls in zip(_moment_components(d), styles):
            mag = np.linalg.norm(np.column_stack([_series(d, f"moment_{name}_{a}") for a in "xyz"]), axis=1)
            if not np.any(mag > 0.0):
                continue  # 無効な項（係数 0）は凡例を埋めないよう描かない
            (line,) = ax.plot(t, mag, ls=ls, color=color, label=f"{d['name']} {labels.get(name, name)}")
            color = line.get_color()
    ax.set_title("空力モーメントの内訳（大きさ。0 の項は省略）\n（仮定係数に基づく計算値）", fontsize=11)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|M| [N·m]")
    ax.grid(alpha=0.3)
    if ax.get_lines():
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "空力モーメントは全時刻で 0（モーメント項の係数がすべて 0）",
                transform=ax.transAxes, ha="center", va="center")
    return _save(fig, out, "10_moments.png")


def _plot_reynolds(docs, out: Path) -> Path | None:
    docs = _with(docs, "reynolds_number")
    if not docs:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    for d in docs:
        ax.plot(_series(d, "time"), _series(d, "reynolds_number"), label=d["name"])
    ax.set_title("Reynolds 数 Re = ρ|u|D/μ（診断値。係数には使っていない）", fontsize=11)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Re")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, out, "11_reynolds.png")
