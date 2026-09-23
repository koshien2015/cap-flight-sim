"""コマンドライン: simulate / sweep / compare / fit / synthesize / tabulate。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .outputs.plots import plot_runs
    from .outputs.serializers import build_result_document, write_result_bundle
    from .simulation import run_simulation

    config = load_config(args.config)
    result = run_simulation(config)
    paths = write_result_bundle(result, args.output)
    doc = build_result_document(result)
    if not args.no_plots:
        plot_runs([doc], Path(args.output) / "plots")
    _print_summary(doc)
    print(f"出力: {', '.join(str(p) for p in paths.values())}")
    return 0


def _print_summary(doc: dict) -> None:
    s = doc["summary"]
    print(f"[{doc['name']}] 終了理由: {s['terminationReason']}  飛行時間 {s['flightTimeSec']:.3f}s")
    cp = s.get("catcherPlane")
    if cp:
        print(f"  捕手面: Y={cp['yM']:+.3f} m (＋=左)  Z={cp['zM']:.3f} m  速度 {cp['speedMS']:.2f} m/s  {cp['rpm']:.0f} rpm")
    else:
        x, y, z = s["finalPositionWorldM"]
        print(f"  捕手面未到達: 最終位置 X={x:.3f} Y={y:+.3f} Z={z:.3f}")
    print(f"  横変化(初速方向の直線から) {s['lateralBreakFromLaunchLineM']:+.3f} m  落下量(同) {s['dropFromLaunchLineM']:.3f} m")
    for w in doc["warnings"]:
        print(f"  ⚠ {w}")


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .experiments import run_sweep

    table = run_sweep(args.config, args.output)
    _print_table(table)
    print(f"出力: {args.output}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .experiments import compare_runs

    table = compare_runs(args.runs, args.output)
    _print_table(table)
    return 0


def _print_table(table) -> None:
    cols = [
        "name", "terminationReason", "flightTimeSec", "catcherY_m", "catcherZ_m",
        "lateralBreakFromLaunchLineM", "dropFromLaunchLineM", "meanAeroLateralAccelMS2", "maxFaceNormalChangeDeg",
    ]
    with_pd = table[[c for c in cols if c in table.columns]]
    print(with_pd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _cmd_fit(args: argparse.Namespace) -> int:
    from .fitting.optimizer import run_fit_from_files

    report = run_fit_from_files(args.config, args.trajectory, args.output, args.test_trajectory or [])
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    from .fitting.trajectory_loader import synthesize_observation

    path = synthesize_observation(args.config, args.output, args.fps, args.noise_m, args.seed)
    print(f"合成観測軌道（仮想データ）: {path}")
    return 0


def _cmd_tabulate(args: argparse.Namespace) -> int:
    from .aerodynamics.tabulate import tabulate_simple_model

    path = tabulate_simple_model(args.config, args.output)
    print(f"係数表: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cap_flight_sim", description="キャップ野球 6DoF フライトシミュレーター")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="1球を計算")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("sweep", help="実験設定のパラメータスイープ")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("compare", help="result.json を重ねて比較")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("fit", help="観測軌道 CSV へ係数をフィット")
    p.add_argument("--trajectory", nargs="+", required=True, help="学習用軌道 CSV（複数可）")
    p.add_argument("--test-trajectory", nargs="*", help="評価用軌道 CSV（学習に使わない）")
    p.add_argument("--config", required=True, help="フィッティング設定 YAML")
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("synthesize", help="シミュレーションから合成観測軌道 CSV を作る（検証用の仮想データ）")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--noise-m", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_synthesize)

    p = sub.add_parser("tabulate", help="簡易モデルを係数表スキーマの CSV へ書き出す（CFD 交換形式の例）")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_tabulate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
