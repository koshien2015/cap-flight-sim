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
    return paths


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
